"""Persistent target KV and speculative scratch, independent of model/sampling.

This store does not execute attention or capture graphs. Metadata preparation and
commit stay outside captured execution. A registered external-write capability
can publish scratch readiness after backend-owned completion; it trusts that
program's all-layer signature rather than proving device execution independently.
"""
from dataclasses import dataclass
from typing import Mapping

import torch


@dataclass(frozen=True)
class Slot:
    request: str
    index: int
    incarnation: int


@dataclass(frozen=True)
class Bucket:
    """Exact ordered query shape, declared growing-context bounds, provenance.

    Context ceilings bound each active entry, not global resident slot indices.
    They are storage applicability bounds, not a measured speed prediction.
    """
    query_lengths: tuple
    context_ceilings: tuple
    source: str

    def __post_init__(self):
        if (not self.query_lengths or len(self.query_lengths) != len(self.context_ceilings)
                or any(type(n) is not int or n < 1 for n in self.query_lengths)
                or any(type(n) is not int or n < 0 for n in self.context_ceilings)
                or not isinstance(self.source, str) or not self.source.strip()):
            raise ValueError('Bucket requires exact positive Q lengths, nonnegative C ceilings and provenance')
        if not isinstance(self.query_lengths, tuple) or not isinstance(self.context_ceilings, tuple):
            raise ValueError('Immutable tuple shapes required')

    @property
    def query_tokens(self):
        return sum(self.query_lengths)

    @property
    def key_capacity(self):
        return sum(self.query_lengths) + sum(self.context_ceilings)


@dataclass(frozen=True)
class Transaction:
    sequence: int
    slots: tuple
    context_lengths: tuple
    bucket: Bucket


@dataclass(frozen=True)
class ExternalWriter:
    """Registered backend signature; authority is exact object identity.

    Completion attests that this trusted fixed program finished. It is not an
    independent proof that the device executed every intended layer correctly.
    """
    bucket: Bucket
    layers: tuple
    scratch_pointers: tuple


@dataclass(frozen=True)
class ExternalWrite:
    writer: ExternalWriter
    transaction: Transaction
    generation: int


class _Workspace:
    def __init__(self, bucket, heads, dim, dtype, device):
        self.bucket = bucket
        n = bucket.key_capacity
        self.committed_indices = torch.zeros(n, dtype=torch.long, device=device)
        self.scratch_indices = torch.zeros_like(self.committed_indices)
        self.from_scratch = torch.zeros((n, 1, 1), dtype=torch.bool, device=device)
        self.valid = torch.zeros((n, 1, 1), dtype=torch.bool, device=device)
        self.cu_query = torch.zeros(len(bucket.query_lengths)+1, dtype=torch.int32, device=device)
        self.cu_key = torch.zeros_like(self.cu_query)
        self.positions = torch.zeros(bucket.query_tokens, dtype=torch.long, device=device)
        # A deterministic layer can reuse these allocations. Its attention must
        # consume them before the next layer overwrites them.
        shape = (n, heads, dim)
        self.old_k = torch.empty(shape, dtype=dtype, device=device)
        self.old_v = torch.empty_like(self.old_k)
        self.new_k = torch.empty_like(self.old_k)
        self.new_v = torch.empty_like(self.old_k)
        self.keys = torch.empty_like(self.old_k)
        self.values = torch.empty_like(self.old_k)

    def pointers(self):
        return {k: v.data_ptr() for k, v in vars(self).items() if isinstance(v, torch.Tensor)}


class PersistentTargetKV:
    """Single-owner transaction store with stable resident and bucket addresses.

    KV layout: [layers, resident slots, context capacity, KV heads, head dim].
    Scratch: [layers, maximum query tokens, KV heads, head dim].
    Request handles bind incarnation as well as slot; stale handles never regain
    authority after reuse. No mutation/admission/removal during a transaction.
    Methods accept already-produced RoPE-applied K/V; no model or probability
    calculation is added here. Tokens accepted by the sampling owner are the only
    scratch prefix copied to committed memory.
    """
    def __init__(self, *, layers, slots, context_capacity, max_query_tokens,
                 heads, dim, dtype=torch.float32, device='cpu'):
        dims = (layers, slots, context_capacity, max_query_tokens, heads, dim)
        if any(type(n) is not int or n < 1 for n in dims):
            raise ValueError('All capacities and dimensions must be positive integers')
        if not dtype.is_floating_point:
            raise ValueError('Floating point KV dtype required')
        self.layers, self.slot_capacity, self.context_capacity = layers, slots, context_capacity
        self.max_query_tokens, self.heads, self.dim = max_query_tokens, heads, dim
        self.device, self.dtype = torch.device(device), dtype
        self.keys = torch.zeros((layers, slots, context_capacity, heads, dim), dtype=dtype, device=self.device)
        self.device = self.keys.device  # Canonicalize aliases such as cuda or cpu:0.
        self.values = torch.zeros_like(self.keys)
        self.scratch_keys = torch.zeros((layers, max_query_tokens, heads, dim), dtype=dtype, device=self.device)
        self.scratch_values = torch.zeros_like(self.scratch_keys)
        self._owners = [None] * slots
        self._handles = [None] * slots
        self._incarnations = [0] * slots
        self._lengths = [0] * slots
        self._workspaces = {}
        self._sequence = 0
        self._pending = None
        self._staged = set()
        self._writers = {}
        self._external = None
        self._submission_generation = 0
        self.failed = False

    def _healthy(self):
        if self.failed:
            raise RuntimeError('Store invalidated by copy failure; construct a new store')

    def _idle(self):
        self._healthy()
        if self._pending is not None:
            raise RuntimeError('Outstanding verification transaction')

    def _slot(self, handle):
        self._healthy()
        if (not isinstance(handle, Slot) or not 0 <= handle.index < self.slot_capacity
                or self._handles[handle.index] is not handle
                or self._owners[handle.index] != handle.request
                or self._incarnations[handle.index] != handle.incarnation):
            raise ValueError('Stale or foreign request incarnation')
        return handle.index

    def add_request(self, request):
        self._idle()
        if not isinstance(request, str) or not request or request in self._owners:
            raise ValueError('New nonempty request identity required')
        if None not in self._owners:
            raise ValueError('Resident slot capacity exhausted')
        i = self._owners.index(None)
        self._incarnations[i] += 1
        self._owners[i], self._lengths[i] = request, 0
        handle = Slot(request, i, self._incarnations[i])
        self._handles[i] = handle
        return handle

    def remove_request(self, handle):
        self._idle()
        i = self._slot(handle)
        self._owners[i], self._lengths[i], self._handles[i] = None, 0, None
        # Old bytes may remain allocated; no new incarnation can address them.

    def length(self, handle):
        return self._lengths[self._slot(handle)]

    def _kv(self, keys, values, shape):
        for x in (keys, values):
            if (not isinstance(x, torch.Tensor) or tuple(x.shape) != shape
                    or x.device != self.device or x.dtype != self.dtype or x.requires_grad):
                raise ValueError('KV shape, device, dtype and no-autograd contract required')

    @torch.no_grad()
    def load_prefix(self, handle, keys, values):
        """Bootstrap exact real-model KV into an empty resident slot."""
        self._idle()
        i = self._slot(handle)
        if self._lengths[i] or not isinstance(keys, torch.Tensor) or keys.ndim != 4:
            raise ValueError('Prefix bootstrap requires empty slot and [L,C,H,D] KV')
        n = keys.shape[1]
        if not 0 <= n <= self.context_capacity:
            raise ValueError('Prefix exceeds resident context capacity')
        self._kv(keys, values, (self.layers, n, self.heads, self.dim))
        try:
            self.keys[:, i, :n].copy_(keys)
            self.values[:, i, :n].copy_(values)
            self._lengths[i] = n
        except Exception:
            self.failed = True
            raise

    def register_bucket(self, bucket):
        self._idle()
        if (not isinstance(bucket, Bucket) or len(bucket.query_lengths) > self.slot_capacity
                or bucket.query_tokens > self.max_query_tokens
                or any(c+q > self.context_capacity for c,q in zip(bucket.context_ceilings,bucket.query_lengths))):
            raise ValueError('Bucket exceeds resident/query capacity')
        if bucket not in self._workspaces:
            self._workspaces[bucket] = _Workspace(bucket, self.heads, self.dim, self.dtype, self.device)

    def workspace_bytes(self, bucket):
        """Exact tensor allocation size before registering a workspace."""
        n, r, q = bucket.key_capacity, len(bucket.query_lengths), bucket.query_tokens
        return 6*n*self.heads*self.dim*self.keys.element_size() + 18*n + 8*(r+1) + 8*q

    def register_external_writer(self, bucket, layers, completion_check):
        self._idle()
        if bucket not in self._workspaces or tuple(layers) != tuple(range(self.layers)) or not callable(completion_check):
            raise ValueError('Registered bucket, complete ordered layer signature and completion checker required')
        writer = ExternalWriter(bucket, tuple(layers),
            (self.scratch_keys.data_ptr(), self.scratch_values.data_ptr()))
        self._writers[id(writer)] = (writer, completion_check)
        return writer

    def prepare_external_write(self, tx, writer):
        self._transaction(tx)
        registered = self._writers.get(id(writer))
        if (registered is None or registered[0] is not writer or writer.bucket != tx.bucket
                or self._staged or self._external is not None):
            raise ValueError('Fresh transaction and exact registered external writer required')
        self._submission_generation += 1
        receipt = ExternalWrite(writer, tx, self._submission_generation)
        self._external = dict(receipt=receipt, state='prepared', event=None)
        return receipt

    def _external_write(self, receipt):
        if self._external is None or self._external['receipt'] is not receipt:
            raise ValueError('Stale, copied or foreign external write receipt')
        self._transaction(receipt.transaction)
        if receipt.writer.scratch_pointers != (self.scratch_keys.data_ptr(), self.scratch_values.data_ptr()):
            self.failed = True
            raise RuntimeError('Registered scratch allocation changed')
        return self._external

    def submit_external_write(self, receipt, event):
        state = self._external_write(receipt)
        if state['state'] != 'prepared':
            raise ValueError('External write must be submitted exactly once')
        state.update(state='submitted', event=event)

    def complete_external_write(self, receipt):
        state = self._external_write(receipt)
        if state['state'] != 'submitted':
            raise ValueError('Submitted, unpublished external write required')
        checker = self._writers[id(receipt.writer)][1]
        if not self._check_external_completion(checker,state['event'],receipt):
            raise RuntimeError('External write completion has not been observed')
        self._staged = set(receipt.writer.layers)
        state['state'] = 'completed'

    def _check_external_completion(self, checker, event, receipt):
        try:return bool(checker(event,receipt))
        except BaseException:
            self.failed=True
            raise

    def invalidate(self):
        """Fail closed after uncertain external/device writes; never reuse buffers."""
        self.failed = True

    @torch.no_grad()
    def begin(self, handles, bucket):
        """Prepare host metadata before any future captured layer execution.

        Exact Q and bounded actual C select a registered bucket. Physical Q is
        never rounded up. Tail K capacity is reported and excluded by cu_key.
        No transaction is published until every metadata copy succeeds. A failed
        preparation may be retried: begin rewrites all metadata, and cannot
        revive an earlier transaction capability.
        """
        self._idle()
        handles = tuple(handles)
        if bucket not in self._workspaces or len(handles) != len(bucket.query_lengths):
            raise ValueError('Registered exact-query bucket required')
        indices = [self._slot(h) for h in handles]
        if len(set(indices)) != len(indices):
            raise ValueError('Active request slots must be unique')
        contexts = tuple(self._lengths[i] for i in indices)
        if any(c > cap for c,cap in zip(contexts,bucket.context_ceilings)):
            raise ValueError('Actual context exceeds declared bucket applicability')
        old, new, choose, positions, cq, ck = [], [], [], [], [0], [0]
        qo = 0
        for i,c,q in zip(indices,contexts,bucket.query_lengths):
            old.extend(i*self.context_capacity+j for j in range(c))
            old.extend([0]*q)
            new.extend([0]*c)
            new.extend(range(qo,qo+q))
            choose.extend([False]*c+[True]*q)
            positions.extend(range(c,c+q))
            qo += q
            cq.append(qo)
            ck.append(ck[-1]+c+q)
        logical_k = len(old)
        tail = bucket.key_capacity-logical_k
        old.extend([0]*tail); new.extend([0]*tail); choose.extend([False]*tail)
        ws = self._workspaces[bucket]
        for dst, data in ((ws.committed_indices,old),(ws.scratch_indices,new),
                          (ws.from_scratch,choose),(ws.valid,[True]*logical_k+[False]*tail),
                          (ws.positions,positions),(ws.cu_query,cq),(ws.cu_key,ck)):
            dst.copy_(torch.tensor(data,device=self.device,dtype=dst.dtype).reshape(dst.shape))
        self._sequence += 1
        tx = Transaction(self._sequence,handles,contexts,bucket)
        self._pending, self._staged = tx, set()
        return tx

    def _transaction(self, tx):
        self._healthy()
        if tx is not self._pending:
            raise ValueError('Stale, copied or foreign transaction')
        for h,c in zip(tx.slots,tx.context_lengths):
            if self._lengths[self._slot(h)] != c:
                raise RuntimeError('Committed state changed during verification')
        return self._workspaces[tx.bucket]

    @torch.no_grad()
    def stage_layer(self, tx, layer, keys, values):
        """Copy real model's new KV only into independent scratch."""
        self._transaction(tx)
        if self._external is not None:
            raise ValueError('Cannot mix eager and registered external scratch writes')
        if type(layer) is not int or not 0 <= layer < self.layers or layer in self._staged:
            raise ValueError('Each model layer must stage exactly once')
        q = tx.bucket.query_tokens
        self._kv(keys, values, (q,self.heads,self.dim))
        self.scratch_keys[layer,:q].copy_(keys)
        self.scratch_values[layer,:q].copy_(values)
        self._staged.add(layer)

    @torch.no_grad()
    def prepare_attention(self, tx, layer):
        """Fixed-size gathers/selection into fixed-address per-bucket K/V.

        Host checks remain here: this method itself is not a graph-capture API.
        Its index_select/where tail is a candidate deterministic subgraph.
        Padding is neutral storage: a future native adapter MUST validate that
        cu_key excludes it; passing the entire capacity as actual K is incorrect.
        """
        ws = self._transaction(tx)
        if layer not in self._staged:
            raise ValueError('Layer scratch missing')
        for resident,scratch,old,new,out in (
                (self.keys,self.scratch_keys,ws.old_k,ws.new_k,ws.keys),
                (self.values,self.scratch_values,ws.old_v,ws.new_v,ws.values)):
            torch.index_select(resident[layer].reshape(-1,self.heads,self.dim),0,ws.committed_indices,out=old)
            torch.index_select(scratch[layer],0,ws.scratch_indices,out=new)
            torch.where(ws.from_scratch,new,old,out=out)
            # masked_fill makes invalid bytes neutral even if old slot data are NaN.
            out.masked_fill_(~ws.valid,0)
        return ws.keys, ws.values, ws.cu_query, ws.cu_key, ws.positions

    @torch.no_grad()
    def commit(self, tx, committed_query_lengths: Mapping):
        """Commit caller-decided verified prefix lengths (anchor included).

        No acceptance or probability law is implemented here. Validate the whole
        decision before any resident write. A device/copy failure poisons the pool
        rather than promising atomic rollback across all model layers.
        """
        self._transaction(tx)
        if set(committed_query_lengths) != set(tx.slots) or len(self._staged) != self.layers:
            raise ValueError('Complete layer scratch and one prefix count per active handle required')
        for h,c,q in zip(tx.slots,tx.context_lengths,tx.bucket.query_lengths):
            n = committed_query_lengths[h]
            if type(n) is not int or not 0 <= n <= q or c+n > self.context_capacity:
                raise ValueError('Commit must be a valid verified prefix')
        qo = 0
        try:
            for h,c,q in zip(tx.slots,tx.context_lengths,tx.bucket.query_lengths):
                n = committed_query_lengths[h]
                self.keys[:,h.index,c:c+n].copy_(self.scratch_keys[:,qo:qo+n])
                self.values[:,h.index,c:c+n].copy_(self.scratch_values[:,qo:qo+n])
                qo += q
            for h in tx.slots:
                self._lengths[h.index] += committed_query_lengths[h]
        except Exception:
            self.failed = True
            raise
        self._pending, self._staged, self._external = None, set(), None
        return dict(committed_tokens=sum(committed_query_lengths.values()),
                    copied_kv_bytes=2*self.layers*self.heads*self.dim*self.keys.element_size()*sum(committed_query_lengths.values()))

    def abort(self, tx):
        self._transaction(tx)
        if self._external is not None and self._external['state'] == 'submitted':
            receipt = self._external['receipt']
            if not self._check_external_completion(self._writers[id(receipt.writer)][1],self._external['event'],receipt):
                raise RuntimeError('Cannot abort buffers with an unfinished external write')
        self._pending, self._staged, self._external = None, set(), None

    def request_kv(self, handle):
        i = self._slot(handle); n = self._lengths[i]
        return self.keys[:,i,:n].detach().clone(), self.values[:,i,:n].detach().clone()

    def pointers(self, bucket):
        """Audit stable allocation identities, never a graph replay assertion."""
        return dict(committed_keys=self.keys.data_ptr(),committed_values=self.values.data_ptr(),
                    scratch_keys=self.scratch_keys.data_ptr(),scratch_values=self.scratch_values.data_ptr(),
                    **self._workspaces[bucket].pointers())

    def work(self, tx):
        self._transaction(tx)
        q = tx.bucket.query_tokens; actual_k = sum(tx.context_lengths)+q
        resident = sum(self._lengths)
        return dict(active_requests=len(tx.slots),resident_requests=sum(x is not None for x in self._owners),
                    logical_query_tokens=q,physical_query_tokens=q,logical_active_key_tokens=actual_k,
                    physical_key_capacity=tx.bucket.key_capacity,padded_key_capacity=tx.bucket.key_capacity-actual_k,
                    inactive_resident_key_tokens=resident-sum(tx.context_lengths),
                    gather_rows_per_kv_per_layer=2*tx.bucket.key_capacity,
                    staging_output_rows_per_layer=tx.bucket.key_capacity,
                    allowed_causal_pairs=sum(c*n+n*(n+1)//2 for c,n in zip(tx.context_lengths,tx.bucket.query_lengths)),
                    resident_allocated_kv_bytes=2*self.keys.numel()*self.keys.element_size(),
                    scratch_allocated_kv_bytes=2*self.scratch_keys.numel()*self.keys.element_size(),
                    bucket_workspace_bytes=sum(t.numel()*t.element_size() for t in vars(self._workspaces[tx.bucket]).values() if isinstance(t,torch.Tensor)),
                    bucket_source=tx.bucket.source,capture_or_replay_verified=False)
