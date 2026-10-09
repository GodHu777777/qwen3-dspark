"""Synchronous packed stochastic reference: shadow proposal and verify/commit.

Model/backbone/head operations are batched; probability draws and cache crops
remain synchronous request loops. This is not an asynchronous serving engine.
"""
import copy
from contextlib import nullcontext
from dataclasses import dataclass

import torch

from .tensor_sampling import (PROBABILITY_POLICY, TensorProposal, logits_to_probabilities,
    sample_categorical, validate_stops, validate_temperature, verify_proposal)


@dataclass(frozen=True)
class RequestSpec:
    input_ids: torch.Tensor
    max_new_tokens: int
    rng: object


@dataclass(frozen=True)
class IssuedProposal:
    request: str
    incarnation: int
    epoch: int
    cache_length: int
    nonce: int
    proposal_limit: int
    mode: str
    proposal: TensorProposal  # observation copy; modifying it cannot replace actual q


@dataclass(frozen=True)
class ProposalBatch:
    proposals: dict
    work: dict


def _copy_proposal(proposal):
    return TensorProposal(*(t.detach().clone() for t in
        (proposal.tokens, proposal.draft_probs, proposal.confidence_logits)))


@torch.no_grad()
def propose_packed(packed, anchors, limits, rngs, *, temperature):
    """One full backbone, one base head, one Markov/confidence head per position.

    Limits are fixed before this helper's draws. Zero limits explicitly skip that
    request's backbone; callers use this only in a declared fixed-budget mode.
    No request loop invokes a model/head. Sampling itself remains per-request.
    """
    temperature = validate_temperature(temperature)
    if not anchors or set(anchors) != set(limits) or set(anchors) != set(rngs):
        raise ValueError('Matching nonempty request/limit/RNG mappings required')
    size = packed.draft.spec.block_size
    if any(type(n) is not int or not 0 <= n <= size for n in limits.values()):
        raise ValueError('Proposal limits must be in [0, block_size]')
    model = packed.draft; device = packed.device; vocab = model.lm_head.out_features
    active = {r:t for r,t in anchors.items() if limits[r]}
    empty = lambda: TensorProposal(torch.empty(0,dtype=torch.long,device=device),
        torch.empty((0,vocab),dtype=torch.float64,device=device),torch.empty(0,device=device))
    result = {r:empty() for r in anchors if not limits[r]}
    work = dict(active_requests=len(anchors),draft_requests=len(active),target_only_skipped_requests=len(anchors)-len(active),
        backbone_calls=0,base_lm_head_calls=0,markov_projection_calls=0,confidence_head_calls=0,
        proposal_positions=sum(limits.values()),base_lm_head_rows=len(active)*size,
        markov_head_rows=sum(limits.values()),probability_policy=PROBABILITY_POLICY)
    if not active:
        return result, work
    features = packed.backbone(active)
    base = model.lm_head(features.hidden).reshape(len(active),size,vocab)
    work.update(backbone_calls=1,base_lm_head_calls=1,backbone=features.work)
    index = {r:i for i,r in enumerate(active)}
    previous = {r:t.reshape(1) for r,t in active.items()}
    tokens = {r:[] for r in active}; laws = {r:[] for r in active}; confidence = {r:[] for r in active}
    for position in range(max(limits.values())):
        requests = [r for r in active if position < limits[r]]
        selected = torch.tensor([index[r] for r in requests],device=device,dtype=torch.long)
        hidden = features.hidden.reshape(len(active),size,-1)[selected,position]
        embedding = model.markov_embedding(torch.cat([previous[r] for r in requests]))
        raw = model.confidence(torch.cat((hidden,embedding),dim=-1)).flatten()
        if not bool(torch.isfinite(raw).all()):raise ValueError('Nonfinite pre-token confidence')
        logits = base[selected,position] + model.markov_projection(embedding)
        probabilities = logits_to_probabilities(logits,temperature)
        work['markov_projection_calls'] += 1; work['confidence_head_calls'] += 1
        for row, request in enumerate(requests):
            # Confidence and q are computed before the current token is sampled.
            token = sample_categorical(probabilities[row],rngs[request]).reshape(1)
            previous[request] = token; tokens[request].append(token)
            laws[request].append(probabilities[row]); confidence[request].append(raw[row].float())
    for request in active:
        result[request] = TensorProposal(torch.cat(tokens[request]),torch.stack(laws[request]),torch.stack(confidence[request]))
    return result, work


class PackedSpeculativeSession:
    """Owns target/draft mutable caches and request RNGs; any execution error poisons it.

    Epoch identity invalidates outstanding proposals after every cache mutation.
    External allocation policies must satisfy nonanticipation themselves; this
    class verifies capability identity and legal prefixes, not planner causality.
    """
    def __init__(self, target, packed_draft, *, temperature=1., eos_ids=(), amp=False):
        if target.lengths or packed_draft.lengths:
            raise ValueError('Session requires initially empty request pools')
        if target.device != packed_draft.device or tuple(target.layer_ids) != tuple(packed_draft.draft.spec.layer_ids):
            raise ValueError('Target/draft device and selected feature layers must match')
        if target.device.type == 'cuda':
            from .rocm_varlen import BACKEND
            from .packed_draft import DRAFT_BACKEND
            if (getattr(getattr(target,'_varlen_kernel',None),'backend_name',None) != BACKEND or
                    getattr(packed_draft.kernel,'backend_name',None) != DRAFT_BACKEND):
                raise ValueError('GPU session requires explicit pinned target and noncausal draft backends')
        self.target = target; self.draft = packed_draft
        self.temperature = validate_temperature(temperature)
        self.eos_ids = validate_stops(eos_ids,target.model.config.vocab_size)
        self.amp = amp; self.requests = {}; self.epoch = 0; self._incarnation = 0; self._nonce = 0
        self._pending = {}; self.failed = False; self.last_prefill_work = None

    def _autocast(self):
        return torch.autocast(self.draft.device.type,dtype=torch.bfloat16) if self.amp else nullcontext()

    def _healthy(self):
        if self.failed:raise RuntimeError('Session invalidated by prior execution failure; create a new session')

    def _invalidate(self):
        # Failure may occur after partial HF layer extension or random draws.
        # Do not promise rollback/replay. No public method may reuse these caches.
        self.failed = True; self._pending.clear(); self.target.reset()
        self.draft.layers=[]; self.draft._lengths={}; self.draft._markers={}
        self.draft.key_requests=self.draft.key_requests[:0]
        self.draft.key_positions=self.draft.key_positions[:0]
        self.draft.last_projection_work=None

    def _advance(self):
        self.epoch += 1; self._pending.clear()

    def _check_lengths(self):
        target = self.target.lengths; draft = self.draft.lengths
        expected = {r:x['prompt_length']+len(x['output'])-1 for r,x in self.requests.items()}
        if target != expected or draft != expected:
            raise RuntimeError('Committed target/draft prefixes must exclude the newest emitted anchor')

    @torch.no_grad()
    def admit(self, requests):
        self._healthy()
        if not requests:raise ValueError('Need new requests')
        for r,spec in requests.items():
            if not isinstance(r,str) or not r or r in self.requests or not isinstance(spec,RequestSpec):raise ValueError('Fresh request names/specs required')
            t=spec.input_ids
            if (not isinstance(t,torch.Tensor) or t.ndim!=2 or t.shape[0]!=1 or t.shape[1]<1 or t.dtype!=torch.long or t.device!=self.target.device or
                not bool(((t>=0)&(t<self.target.model.config.vocab_size)).all()) or type(spec.max_new_tokens) is not int or spec.max_new_tokens<1 or not callable(getattr(spec.rng,'uniform',None))):
                raise ValueError('Valid prompt, positive budget and per-request RNG required')
        try:
            for r in requests:self.target.add_request(r);self.draft.add_request(r)
            features=self.target.append({r:x.input_ids for r,x in requests.items()})
            # PackedTarget.predict(last_only=True) projects every prompt row.
            # Select final hidden rows first, then do one R-row LM-head projection.
            final_hidden=torch.cat([features.for_request(r).last[:,-1:] for r in requests],dim=1)
            final_logits=self.target.model.get_output_embeddings()(final_hidden)[0]
            additions={}
            for index,(r,spec) in enumerate(requests.items()):
                token=int(sample_categorical(logits_to_probabilities(final_logits[index],self.temperature),spec.rng).item())
                additions[r]=dict(prompt_length=spec.input_ids.shape[1],budget=spec.max_new_tokens,rng=spec.rng,output=[token],
                    finished=token in self.eos_ids or spec.max_new_tokens==1,incarnation=self._incarnation)
                self._incarnation+=1
            with self._autocast():self.draft.append_committed({r:features.for_request(r).context for r in requests})
            self.requests.update(additions);self._advance();self._check_lengths()
            self.last_prefill_work=dict(target=features.work,draft_context_projection=self.draft.last_projection_work,
                prefill_lm_head_calls=1,prefill_lm_head_rows=len(requests),projection_scope='Only one final hidden row per request, batched once; not whole-prompt projection')
            return {r:list(self.requests[r]['output']) for r in requests}
        except Exception:
            self._invalidate();raise

    @torch.no_grad()
    def propose(self, active, proposal_limits=None, *, mode='shadow'):
        self._healthy();active=tuple(active)
        if not active or len(set(active))!=len(active) or any(r not in self.requests or self.requests[r]['finished'] for r in active):
            raise ValueError('Unique unfinished active requests required')
        if mode not in ('shadow','fixed_budget'):raise ValueError('Explicit shadow or fixed-budget proposal mode required')
        size=self.draft.draft.spec.block_size
        limits={r:size for r in active} if proposal_limits is None else dict(proposal_limits)
        if set(limits)!=set(active) or any(type(n) is not int or not 0<=n<=size for n in limits.values()):raise ValueError('Invalid declared proposal limits')
        if mode=='shadow' and any(n==0 for n in limits.values()):raise ValueError('Shadow collection must not silently skip history refresh')
        if mode=='fixed_budget':limits={r:min(n,self.requests[r]['budget']-len(self.requests[r]['output'])) for r,n in limits.items()}
        self._check_lengths()
        try:
            anchors={r:torch.tensor([[self.requests[r]['output'][-1]]],device=self.target.device,dtype=torch.long) for r in active}
            with self._autocast():
                proposals,work=propose_packed(self.draft,anchors,limits,{r:self.requests[r]['rng'] for r in active},temperature=self.temperature)
            issued={}
            for r in active:
                state=self.requests[r];self._nonce+=1
                handle=IssuedProposal(r,state['incarnation'],self.epoch,self.target.lengths[r],self._nonce,limits[r],mode,_copy_proposal(proposals[r]))
                # Retain independent tensors: observation edits cannot forge actual q.
                self._pending[r]=(handle,proposals[r],work);issued[r]=handle
            work['observation_copy_elements']=sum(t.numel() for h in issued.values() for t in (h.proposal.tokens,h.proposal.draft_probs,h.proposal.confidence_logits))
            return ProposalBatch(issued,copy.deepcopy(work))
        except Exception:
            self._invalidate();raise

    @torch.no_grad()
    def verify_commit(self, proposals, allocations, *, allocation_policy):
        self._healthy()
        typed_policy=callable(getattr(allocation_policy,'validate_allocation',None))
        if not typed_policy and allocation_policy not in ('fixed_before_proposal','external_nonanticipating'):
            raise ValueError('Caller must explicitly declare or validate a nonanticipating allocation policy')
        if not proposals or set(proposals)!=set(allocations):raise ValueError('Matching proposal/allocation mappings required')
        selected={};before=self.target.lengths;self._check_lengths()
        for r,handle in proposals.items():
            pending=self._pending.get(r)
            if (pending is None or pending[0] is not handle or r not in self.requests or handle.request!=r or
                handle.epoch!=self.epoch or handle.incarnation!=self.requests[r]['incarnation'] or handle.cache_length!=before[r]):
                raise ValueError('Stale, reused or foreign proposal capability')
            n=allocations[r];remaining=self.requests[r]['budget']-len(self.requests[r]['output'])
            if type(n) is not int or not 0<=n<=min(handle.proposal_limit,remaining):raise ValueError('Allocation must be a legal proposal prefix within output budget')
            if allocation_policy=='fixed_before_proposal' and (handle.mode!='fixed_budget' or n!=handle.proposal_limit):
                raise ValueError('Fixed policy must use its predeclared proposal length')
            p=pending[1];selected[r]=TensorProposal(p.tokens[:n],p.draft_probs[:n],p.confidence_logits[:n])
        policy_record=None;policy_host_values=0
        if typed_policy:
            original_confidence={r:tuple(self._pending[r][1].confidence_logits.detach().cpu().tolist()) for r in proposals}
            policy_host_values=sum(len(values) for values in original_confidence.values())
            context=dict(epoch=self.epoch,roster=[dict(request=r,incarnation=x['incarnation'],cache_length=before[r],
                remaining_output_budget=x['budget']-len(x['output'])) for r,x in self.requests.items() if not x['finished']],
                proposal_confidence_logits=original_confidence,
                proposal_identity={r:dict(request=h.request,nonce=h.nonce,incarnation=h.incarnation,epoch=h.epoch,cache_length=h.cache_length,proposal_limit=h.proposal_limit,mode=h.mode)
                    for r,h in proposals.items()})
            policy_record=allocation_policy.validate_allocation(context,proposals,dict(allocations))
            if not isinstance(policy_record,dict):raise ValueError('Typed allocation validator must return a proof record')
        proposal_work={id(self._pending[r][2]):copy.deepcopy(self._pending[r][2]) for r in proposals}
        try:
            chunks={r:torch.cat((torch.tensor([self.requests[r]['output'][-1]],device=self.target.device),p.tokens))[None] for r,p in selected.items()}
            verified=self.target.append(chunks);logits=self.target.predict(verified)
            decisions={};probabilities={};contexts={};record={}
            for r,p in selected.items():
                state=self.requests[r];remaining=state['budget']-len(state['output'])
                probabilities[r]=logits_to_probabilities(logits[r][0],self.temperature)
                decisions[r]=verify_proposal(p,probabilities[r],state['rng'],max_new_tokens=remaining,eos_ids=self.eos_ids)
                committed=decisions[r].tokens.numel();contexts[r]=verified.for_request(r).context[:,:committed]
                record[r]=dict(accepted=decisions[r].accepted_draft_tokens,proposed=p.tokens.numel(),committed=committed,
                    rejected_index=decisions[r].rejected_index,extra_token_kind=decisions[r].extra_token_kind,stop_reason=decisions[r].stop_reason,
                    cache_before=before[r],cache_after=before[r]+committed)
            for r in selected:self.target.crop(r,record[r]['cache_after'])
            with self._autocast():self.draft.append_committed(contexts)
            for r,decision in decisions.items():
                state=self.requests[r];state['output'].extend(decision.tokens.tolist())
                state['finished']=decision.stop_reason in ('eos','budget')
            self._advance();self._check_lengths()
            return dict(requests=record,decisions=decisions,proposals=selected,target_probs=probabilities,
                work=dict(target=verified.work,draft_context_projection=self.draft.last_projection_work,
                    proposal_batches=list(proposal_work.values()),policy_confidence_host_values=policy_host_values),
                allocation_policy='typed_validator' if typed_policy else allocation_policy,policy_validation=policy_record,
                planner_causality_proven=False,probability_policy=PROBABILITY_POLICY)
        except Exception:
            self._invalidate();raise

    def step(self, allocations):
        """Fixed-budget baseline convenience; explicitly allows zero-draft skip."""
        issued=self.propose(allocations,allocations,mode='fixed_budget')
        actual={r:h.proposal_limit for r,h in issued.proposals.items()}
        result=self.verify_commit(issued.proposals,actual,allocation_policy='fixed_before_proposal')
        result['work']['proposal']=issued.work
        return result

    def remove(self, request):
        self._healthy()
        if request not in self.requests:raise KeyError(request)
        try:
            self.target.remove_request(request);self.draft.remove_request(request)
            del self.requests[request];self._advance();self._check_lengths()
        except Exception:
            self._invalidate();raise

    def outputs(self):
        self._healthy()
        return {r:list(state['output']) for r,state in self.requests.items()}
