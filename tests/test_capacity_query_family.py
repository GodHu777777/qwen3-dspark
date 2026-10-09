"""Synchronous real full-shadow bridge with synthetic identity-B cost curves.

No measured SPS or trained confidence/acceptance claim. The zero-score fixture
fixes the actual confidence head to finite -1000 logits before any admission.
"""
import copy
from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from dspark_qwen.async_capacity import Profile, TwoStepCapacity, search_capacity
from dspark_qwen.async_round import CapacityRoundDriver
from dspark_qwen.calibration import calibrated_probabilities
from dspark_qwen.cached_target import CachedTarget
from dspark_qwen.config import DraftConfig
from dspark_qwen.model import DSparkDraft
from dspark_qwen.packed_draft import PackedDraft, test_only_noncausal_varlen
from dspark_qwen.packed_sampling import PackedSpeculativeSession, RequestSpec
from dspark_qwen.persistent_qwen_graph import CPUReplayEmulator
from dspark_qwen.persistent_qwen_target import PersistentQwenTarget, test_only_persistent_sdpa
from dspark_qwen.persistent_target_kv import Bucket, QueryFamily
from dspark_qwen.target_strategy import FiniteTargetBuckets, PersistentTargetStrategy
from tests.test_persistent_qwen_graph import model
from tests.test_persistent_sampling import RecordedRandom


class CheckedRandom(RecordedRandom):
    def __init__(self,seed):super().__init__(seed);self.check=None
    def uniform(self,reference):
        if self.check is not None:self.check()
        return super().uniform(reference)


class CapacityQueryFamilyTests(unittest.TestCase):
    def setUp(self):self.round_records=[]

    def session(self,*,family,zero=False,missing_b=None):
        m=model();reference=copy.deepcopy(m)
        draft=DSparkDraft(m,DraftConfig(layer_ids=(0,2),num_layers=2,block_size=3,
            markov_rank=8,mask_token_id=127)).eval()
        if zero:
            with torch.no_grad():draft.confidence.weight.zero_();draft.confidence.bias.fill_(-1000.)
        family_values=tuple(QueryFamily(r,b,min(4,b-r+1),64*r,64,'synthetic finite CPU family')
            for r in (1,2) for b in range(r,4*r+1) if (r,b)!=(2,missing_b))
        exact_values=tuple(Bucket(q,(60,)*len(q),'independent exact eager')
            for q in [(i,) for i in range(1,5)]+[(i,j) for i in range(1,5) for j in range(1,5)])
        prefill=(Bucket((3,4),(0,0),'two request admission'),Bucket((3,),(0,),'new incarnation admission'))
        opts=dict(graph_backend=CPUReplayEmulator(),max_graph_buckets=1,graph_byte_budget=1024**2) if family else {}
        target=PersistentQwenTarget(m,(0,2),slots=2,context_capacity=64,max_query_tokens=8,
            test_kernel=test_only_persistent_sdpa,**opts)
        strategy=PersistentTargetStrategy(target,FiniteTargetBuckets(prefill,family_values if family else exact_values))
        if family:
            hot=next((f for f in family_values if (f.request_count,f.query_tokens)==(2,5)),None)
            if hot is not None:target.register_graph_bucket(hot,reserve_bytes=4096)
        session=PackedSpeculativeSession(target,PackedDraft(draft,test_kernel=test_only_noncausal_varlen),target_strategy=strategy)
        session.admit(dict(a=RequestSpec(torch.tensor([[1,2,3]]),30,CheckedRandom(91)),
                           b=RequestSpec(torch.tensor([[4,5,6,7]]),30,CheckedRandom(92))))
        if family and hot is not None:
            target.capture_graph(dict(a=torch.tensor([[8]]),b=torch.tensor([[9,10,11,12]])),bucket=hot)
        return session,reference

    def fixture(self,zero=False,missing_b=None):
        family,reference=self.session(family=True,zero=zero,missing_b=missing_b)
        exact,_=self.session(family=False,zero=zero)
        planner=TwoStepCapacity(3)
        values=tuple(float(b) for b in range(1,9)) if zero else (1.,1.,1.,1.,100.,1.,1.,1.)
        profile=Profile('SYNTHETIC_CPU_ONLY_identity_B_' + ('rising_zero_fixture' if zero else 'B5_cliff'),values,tuple(range(1,9)),0,64)
        return family,exact,reference,planner,CapacityRoundDriver(family,planner,profile)

    def assert_equal(self,family,exact,left,right,reference):
        self.assertEqual(family.outputs(),exact.outputs());self.assertEqual(left['requests'],right['requests'])
        for r in left['proposals']:
            for field in ('tokens','draft_probs','confidence_logits'):
                self.assertTrue(torch.equal(getattr(left['proposals'][r],field),getattr(right['proposals'][r],field)))
            self.assertTrue(torch.equal(left['target_probs'][r],right['target_probs'][r]))
        for r,state in family.requests.items():
            self.assertEqual(state['rng'].values,exact.requests[r]['rng'].values)
            for owner in ('target','draft'):
                for x,y in zip(getattr(family,owner).request_kv(r),getattr(exact,owner).request_kv(r)):
                    for a,b in zip(x,y):torch.testing.assert_close(a,b,atol=0,rtol=0)
            # Separate full-prefix HF cache, not the family layout/gather path.
            prompt=torch.tensor([[1,2,3]]) if r=='a' else torch.tensor([[4,5,6,7]])
            ids=torch.cat((prompt,torch.tensor([state['output'][:-1]],dtype=torch.long)),dim=1)
            cached=CachedTarget(copy.deepcopy(reference),(0,2));features=cached.prefill(ids)
            for actual,expected in zip(family.target.request_kv(r),cached.cache.layers):
                torch.testing.assert_close(actual[0],expected.keys,atol=2e-6,rtol=2e-5)
                torch.testing.assert_close(actual[1],expected.values,atol=2e-6,rtol=2e-5)
            projected=family.draft.draft.project_context_kv(features.context,0)
            for actual,expected in zip(family.draft.request_kv(r),projected):
                for a,b in zip(actual,expected):torch.testing.assert_close(a,b,atol=3e-6,rtol=3e-5)

    def round(self,family,exact,reference,planner,driver,*,tamper_observation=False):
        epoch=len(planner.history);events=[];captured={};before={r:len(s['rng'].values) for r,s in family.requests.items()}
        freeze,propose,bind=planner.freeze,family.propose,planner.bind
        def frozen(**kw):
            events.append('freeze');ticket=freeze(**kw);captured['ticket']=ticket;return ticket
        def drawn(*args,**kw):
            self.assertEqual(events[-1],'freeze');self.assertEqual(kw,{'mode':'shadow'})
            events.append('propose');batch=propose(*args,**kw);captured['batch']=batch
            self.assertEqual(batch.work['proposal_positions'],len(captured['ticket'].roster)*3)
            for h in batch.proposals.values():
                self.assertEqual((h.mode,h.proposal_limit),('shadow',3))
            if tamper_observation:
                for h in batch.proposals.values():h.proposal.tokens.zero_();h.proposal.draft_probs.zero_()
            return batch
        def bound(ticket,handles):
            events.append('bind');result=bind(ticket,handles);captured['bound']=result;return result
        def check_draw():
            self.assertIn('ticket',captured)
            self.assertIs(planner._pending,captured['ticket'])
        for state in family.requests.values():state['rng'].check=check_draw
        with patch.object(planner,'freeze',side_effect=frozen),patch.object(family,'propose',side_effect=drawn),patch.object(planner,'bind',side_effect=bound):
            result=driver.step()
        for state in family.requests.values():state['rng'].check=None
        self.assertEqual(events,['freeze','propose','bind'])
        ticket=captured['ticket'];bound=captured['bound'];names=tuple(r.request for r in ticket.roster)
        oracle_batch=exact.propose(names,mode='shadow')
        for r,h in oracle_batch.proposals.items():
            self.assertGreaterEqual(len(family.requests[r]['rng'].values)-before[r],4)  # gamma shadow + at least one target draw
            private=calibrated_probabilities(h.proposal.confidence_logits.tolist(),ticket.calibration.temperatures)[0]
            self.assertEqual(tuple(private),bound.conditional_scores[names.index(r)])
        oracle=exact.verify_commit(oracle_batch.proposals,bound.allocations,allocation_policy='external_nonanticipating')
        self.assert_equal(family,exact,result,oracle,reference)
        proof=result['policy_validation'];work=result['work']['target'];actual=len(names)+sum(bound.allocations.values())
        for value in (proof['logical_b'],proof['physical_b'],work['logical_query_tokens'],work['physical_query_tokens']):self.assertEqual(value,actual)
        self.assertLessEqual(actual,proof['reserved_k'])
        self.assertEqual(proof['reservation_physical_b'],proof['reserved_k'])
        self.assertEqual(work['query_lengths'],[1+bound.allocations[r] for r in names])
        self.assertEqual(work['execution_kind'],'cpu_replay_emulator_not_gpu_graph' if (len(names),actual)==(2,5) else 'explicit_eager')
        self.assertEqual(planner.history[epoch].shadow_positions,len(names)*3)
        self.assertEqual(sum(x['proposal_positions'] for x in result['work']['proposal_batches']),len(names)*3)
        self.assertEqual(result['work']['policy_confidence_host_values'],len(names)*3)
        self.assertEqual(proof['history_epoch'],epoch-2 if epoch>=2 else None)
        self.assertEqual(proof['profile_identity'],ticket.profile.identity)
        self.assertEqual(proof['score_digest'],bound.allocation.score_digest)
        self.assertEqual(planner.history[epoch].scores,tuple(tuple(torch.tensor(row,dtype=torch.float64).cumprod(0).tolist()) for row in bound.conditional_scores))
        if epoch>=2:
            self.assertEqual(proof['history_digest'],planner.history[epoch-2].digest)
            self.assertEqual(proof['historical_k'],search_capacity(planner.history[epoch-2]))
        self.assertIsNone(family.target._feature_lease)
        self.round_records.append(dict(planner_epoch=epoch,
            fixture_profile=ticket.profile.name,profile_identity=ticket.profile.identity,
            roster=[dict(request=r.request,incarnation=r.incarnation,remaining=r.remaining_output_budget) for r in ticket.roster],
            historical_k=ticket.historical_k,reserved_k=ticket.reserved_k,actual_b=actual,
            actual_query_lengths=work['query_lengths'],execution_kind=work['execution_kind'],
            history_epoch=proof['history_epoch'],history_digest=proof['history_digest'],
            conditional_scores=bound.conditional_scores,score_digest=proof['score_digest'],
            shadow_positions=planner.history[epoch].shadow_positions,
            rng_draws={r:len(family.requests[r]['rng'].values)-before[r] for r in names},
            q_p_rng_outputs_target_draft_kv_match_exact_full_shadow=True,
            full_prefix_hf_target_draft_kv_close=True,replanning=list(ticket.replanning)))
        return result,bound,ticket

    def test_full_shadow_current_scores_nonuniform_hot_graph_churn_and_last_budget(self):
        family,exact,reference,planner,driver=self.fixture()
        for epoch in range(3):
            result,bound,ticket=self.round(family,exact,reference,planner,driver,tamper_observation=epoch==1)
            self.assertEqual(result['policy_validation']['reserved_k'],2 if epoch<2 else 5)
        self.assertEqual(sum(bound.allocations.values()),3)
        self.assertEqual(len(set(result['work']['target']['query_lengths'])),2)
        # Current R changes; historical full-shadow population still has departed b.
        for session in (family,exact):session.remove('b')
        result,_,_=self.round(family,exact,reference,planner,driver)
        self.assertEqual((result['policy_validation']['historical_k'],result['policy_validation']['reserved_k']),(5,4))
        self.assertIn('roster_churn',result['policy_validation']['replanning'])
        old=family.requests['a']['incarnation']
        for session in (family,exact):
            session.remove('a');session.admit({'a':RequestSpec(torch.tensor([[1,2,3]]),2,CheckedRandom(191))})
        self.assertGreater(family.requests['a']['incarnation'],old)
        result,bound,ticket=self.round(family,exact,reference,planner,driver)
        self.assertEqual(ticket.roster[0].remaining_output_budget,1)
        self.assertEqual((ticket.historical_k,ticket.reserved_k),(5,1));self.assertEqual(bound.allocations,{'a':0})
        self.assertTrue(family.requests['a']['finished']);self.assertEqual(result['work']['target']['physical_query_tokens'],1)
        family.target.close();exact.target.close()

    def test_finite_real_zero_head_underfills_reservation_but_keeps_full_shadow(self):
        family,exact,reference,planner,driver=self.fixture(zero=True)
        for epoch in range(4):
            result,bound,ticket=self.round(family,exact,reference,planner,driver)
            self.assertEqual(bound.conditional_scores,((0.,0.,0.),(0.,0.,0.)))
            self.assertEqual(bound.allocations,{'a':0,'b':0})
            self.assertEqual((ticket.reserved_k,result['work']['target']['physical_query_tokens']),(2 if epoch<2 else 8,2))
        family.target.close();exact.target.close()

    def test_nonidentity_profile_rejected_before_draw_including_reassignment(self):
        family,exact,_,planner,driver=self.fixture()
        bad=replace(driver.profile,physical=(2,2,3,4,5,6,7,8))
        before={r:list(s['rng'].values) for r,s in family.requests.items()}
        with self.assertRaisesRegex(ValueError,'identity physical-B'):CapacityRoundDriver(family,planner,bad)
        with self.assertRaises(ValueError):CapacityRoundDriver(family,planner,None)
        driver.profile=bad
        with self.assertRaisesRegex(ValueError,'identity physical-B'):driver.step()
        self.assertFalse(driver.failed);self.assertIsNone(planner._pending);self.assertFalse(planner.history)
        for r,state in family.requests.items():self.assertEqual(state['rng'].values,before[r])
        family.target.close();exact.target.close()

    def test_missing_actual_family_fails_after_shadow_before_target_and_cannot_retry(self):
        family,exact,reference,planner,driver=self.fixture(missing_b=5)
        for _ in range(2):self.round(family,exact,reference,planner,driver)
        before={r:len(s['rng'].values) for r,s in family.requests.items()};lengths=dict(family.target.lengths)
        history=planner.history
        with patch.object(family.target,'verify',side_effect=AssertionError('target must not run')):
            with self.assertRaisesRegex(ValueError,'No finite'):driver.step()
        self.assertEqual(family.target.lengths,lengths);self.assertEqual(planner.history,history)
        self.assertEqual(len(planner.history),2);self.assertEqual(planner._pending.reserved_k,5)
        for r,state in family.requests.items():self.assertEqual(len(state['rng'].values)-before[r],3)
        self.assertTrue(driver.failed)
        with self.assertRaisesRegex(RuntimeError,'no retry'):driver.step()
        family.target.close();exact.target.close()

    def test_public_confidence_and_foreign_bound_reject_before_target_or_acceptance_draws(self):
        for corruption in ('confidence','foreign_handle','foreign_ticket'):
            family,exact,_,planner,driver=self.fixture();original=family.propose;bind=planner.bind
            before={r:len(s['rng'].values) for r,s in family.requests.items()};lengths=dict(family.target.lengths)
            def propose(*args,**kw):
                batch=original(*args,**kw)
                if corruption=='confidence':
                    for h in batch.proposals.values():h.proposal.confidence_logits.fill_(100.)
                return batch
            def bind_corrupt(ticket,handles):
                if corruption=='foreign_ticket':return bind(replace(ticket),handles)
                if corruption=='foreign_handle':handles={r:copy.copy(h) for r,h in handles.items()}
                return bind(ticket,handles)
            with patch.object(family,'propose',side_effect=propose),patch.object(planner,'bind',side_effect=bind_corrupt),patch.object(family.target,'verify',side_effect=AssertionError('target must not run')):
                with self.assertRaises(ValueError):driver.step()
            self.assertTrue(driver.failed);self.assertEqual(family.target.lengths,lengths);self.assertFalse(planner.history)
            for r,state in family.requests.items():self.assertEqual(len(state['rng'].values)-before[r],3)
            family.target.close();exact.target.close()



if __name__=='__main__':unittest.main()
