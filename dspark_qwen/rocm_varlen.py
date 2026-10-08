"""Explicit pinned private-ATen reference backend; never an automatic fallback.

The immutable runtime identity is checked once per constructed kernel, not per
layer. Layout value checks synchronize once for a new layout object; cheap tensor
metadata checks remain per call. This is a checked reference, not a timing path.
No native tensor/full-Qwen gate pass is implied by selecting this backend.
"""
import math
import os

BACKEND = 'rocm_aten_no_window_pinned_v1'
PUBLIC_BACKEND = 'public'
SCHEMA = ('aten::_flash_attention_forward(Tensor query, Tensor key, Tensor value, Tensor? cum_seq_q, '
          'Tensor? cum_seq_k, SymInt max_q, SymInt max_k, float dropout_p, bool is_causal, '
          'bool return_debug_mask, *, float? scale=None, SymInt? window_size_left=None, '
          'SymInt? window_size_right=None, Tensor? seqused_k=None, Tensor? alibi_slopes=None, '
          'Tensor? block_table=None, int? num_splits=None) -> (Tensor output, Tensor softmax_logsumexp, '
          'Tensor rng_state, Tensor unused, Tensor debug_attn_mask)')
RUNTIME_PIN = dict(torch='2.12.0+rocm7.2', torch_git='7661cd9c6b841b62b7f411aa52ec51f05457263b',
    hip='7.2.53211', device_arch='gfx1201', flash_preference='_ROCmFABackend.AOTriton',
    flash_prefer_ck_env=None, aten_schema=SCHEMA)


def validate_runtime(runtime):
    """Pure identity validation, separately CPU-testable."""
    if runtime != RUNTIME_PIN:
        changed = sorted(k for k in set(runtime) | set(RUNTIME_PIN) if runtime.get(k) != RUNTIME_PIN.get(k))
        raise RuntimeError('Pinned ROCm varlen runtime mismatch: ' + ', '.join(changed))


def runtime_identity(device):
    import torch
    if device.type != 'cuda' or not torch.cuda.is_available() or torch.version.hip is None:
        raise RuntimeError('Pinned ROCm varlen requires an explicitly selected ROCm GPU; no fallback')
    return dict(torch=torch.__version__, torch_git=torch.version.git_version, hip=torch.version.hip,
        device_arch=getattr(torch.cuda.get_device_properties(device), 'gcnArchName', None),
        flash_preference=str(torch.backends.cuda.preferred_rocm_fa_library()),
        flash_prefer_ck_env=os.environ.get('TORCH_ROCM_FA_PREFER_CK'),
        aten_schema=str(torch.ops.aten._flash_attention_forward.default._schema))


class PinnedRocmVarlenKernel:
    """One explicit private entry, with fail-closed runtime/input contracts.

    Runtime/FA settings must stay unchanged while this object is in use. Layout
    tensors are immutable during append; only the latest layout is retained for
    validation reuse across model layers, so there is no growing cache. Inference
    tensors without version counters are value-checked each call. Autograd inputs
    are refused; this entry is only for a frozen inference target.
    """
    backend_name = BACKEND
    native_varlen_calls_per_layer = 1

    def __init__(self, device):
        import torch
        self.device = torch.device(device)
        if self.device.type == 'cuda' and self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        self.runtime = runtime_identity(self.device)
        validate_runtime(self.runtime)
        self._operator = torch.ops.aten._flash_attention_forward
        self._validated_layout = None
        self._layout_versions = None

    def _validate_inputs(self, query, key, value, layout, scale):
        import torch
        from .varlen_target import PackedLayout
        tensors = (query, key, value)
        if any(not isinstance(t, torch.Tensor) or not t.is_cuda or t.device != self.device or
               t.dtype != torch.bfloat16 or t.requires_grad or t.ndim != 3 or not t.is_contiguous() for t in tensors):
            raise ValueError('Pinned varlen requires no-autograd contiguous BF16 THD tensors on its selected GPU')
        if (query.shape[1:] != (16, 128) or key.shape[1:] != (8, 128) or
                value.shape != key.shape or query.shape[0] == 0 or key.shape[0] == 0):
            raise ValueError('Pinned varlen supports nonempty Hq16/Hkv8/D128 Qwen tensors only')
        if not isinstance(scale, (float, int)) or not math.isfinite(scale) or scale != 128**-0.5:
            raise ValueError('Pinned varlen requires the diagnosed 1/sqrt(128) scale')
        if not isinstance(layout, PackedLayout):
            raise ValueError('Pinned varlen requires PackedLayout')
        if query.shape[0] != layout.query_tokens or key.shape[0] != layout.gathered_key_tokens:
            raise ValueError('Tensor token dimensions differ from packed layout')
        cu = (layout.cu_query, layout.cu_key)
        if any(not isinstance(t, torch.Tensor) or t.device != self.device or t.dtype != torch.int32 or
               t.ndim != 1 or not t.is_contiguous() for t in cu):
            raise ValueError('Pinned varlen requires contiguous int32 cumulative lengths on its selected GPU')
        try:
            versions = tuple(t._version for t in cu)
        except RuntimeError:
            # inference_mode tensors have no version counter. Revalidate values
            # on every call rather than treating them as safely cached.
            versions = None
        if versions is not None and layout is self._validated_layout and versions == self._layout_versions:
            return
        qs, ks = layout.query_lengths, layout.key_lengths
        if (not qs or len(qs) != len(ks) or len(layout.request_markers) != len(qs) or
                len(set(layout.request_markers)) != len(qs) or
                any(type(q) is not int or type(k) is not int or not 0 < q <= k for q, k in zip(qs, ks))):
            raise ValueError('Invalid packed lengths/markers for bottom-right causal suffixes')
        def cumulative(lengths):
            result = [0]
            for n in lengths: result.append(result[-1] + n)
            if result[-1] >= 2**31:
                raise ValueError('Packed cumulative length exceeds int32')
            return result
        if layout.cu_query.tolist() != cumulative(qs) or layout.cu_key.tolist() != cumulative(ks):
            raise ValueError('Cumulative lengths differ from packed metadata')
        self._validated_layout, self._layout_versions = layout, versions

    def __call__(self, query, key, value, layout, *, scale):
        self._validate_inputs(query, key, value, layout, scale)
        # No try/except, fallback, backend-preference mutation or split tuning.
        output = self._operator(query, key, value, layout.cu_query, layout.cu_key,
            max(layout.query_lengths), max(layout.key_lengths), 0.0, True, False,
            scale=scale, window_size_left=None, window_size_right=None,
            seqused_k=None, alibi_slopes=None, block_table=None, num_splits=None)[0]
        if output.shape != query.shape or output.dtype != query.dtype or output.device != query.device:
            raise RuntimeError('Private ATen varlen returned incompatible output')
        return output
