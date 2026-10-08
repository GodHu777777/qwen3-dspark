#!/usr/bin/env python3
"""Independent alignment diagnostic. Default dry-run is stdlib-only.

Successful execution means all diagnostic observations were saved, not that the
production native adapter passed its original correctness gate.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = {
    'version': 1, 'seed': 20261009, 'dtype': 'bfloat16',
    'q_lengths': [1, 8], 'k_lengths': [17, 29], 'inactive_k': 13,
    'head_dim': 128, 'query_heads': 16, 'kv_heads': 8,
    'atol': 0.02, 'rtol': 0.02, 'max_rms_error': 0.005,
    'timeout_seconds': 120, 'native_calls': 6,
    'variants': ['public_gqa', 'public_repeated_kv_no_gqa', 'private_aten_mapping_control'],
    'inputs': ['original_random', 'zero_q_position_ramp'],
    'primary_oracles': ['top_left', 'bottom_right'],
    'ramp_fingerprint': 'full_attention',
    'relative_error_denominator_floor': 1e-30,
    'scope': 'Independent mapping diagnosis; does not replace failed native gate or change production adapter',
}


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def identity():
    paths = [Path(__file__).resolve(), *sorted((ROOT/'dspark_qwen').glob('*.py'))]
    return {'protocol': PROTOCOL,
            'protocol_sha256': hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest(),
            'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}


def prepare_inputs(device):
    """Preserve original first-case random generation order, including inactive KV."""
    import torch
    sys.path.insert(0, str(ROOT))
    from dspark_qwen.varlen_target import PackedLayout
    torch.manual_seed(PROTOCOL['seed'])
    pairs = [(m, p) for p in range(29) for m, n in [(1, 17), (2, 29), (3, 13)] if p < n]
    kr = torch.tensor([m for m, p in pairs], dtype=torch.long, device=device)
    kp = torch.tensor([p for m, p in pairs], dtype=torch.long, device=device)
    key = torch.randn(1, 8, 59, 128, dtype=torch.float32).to(device=device, dtype=torch.bfloat16)
    value = torch.randn(1, 8, 59, 128, dtype=torch.float32).to(device=device, dtype=torch.bfloat16)
    qr = torch.tensor([1]+[2]*8, dtype=torch.long, device=device)
    qp = torch.tensor([16]+list(range(21, 29)), dtype=torch.long, device=device)
    layout = PackedLayout.from_metadata(qr, qp, kr, kp)
    q = torch.randn(9, 16, 128, dtype=torch.float32).to(device=device, dtype=torch.bfloat16)
    k, v = layout.gather_kv(key, value)
    ramp = (kp.float()/32)[None, None, :, None].expand_as(value).to(torch.bfloat16)
    _, ramp_v = layout.gather_kv(key, ramp)
    return dict(q=q, k=k, v=v, ramp_v=ramp_v, key=key, value=value,
                qr=qr, qp=qp, kr=kr, kp=kp, layout=layout)


def references(q, k, v, layout):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    outputs = {}
    with sdpa_kernel(SDPBackend.MATH):
        for alignment in ['top_left', 'bottom_right']:
            rows, qo, ko = [], 0, 0
            for nq, nk in zip(layout.query_lengths, layout.key_lengths):
                ends = torch.arange(nq, device=q.device)
                if alignment == 'bottom_right':
                    ends = ends + nk-nq
                mask = (torch.arange(nk, device=q.device)[None] <= ends[:, None])[None, None]
                rows.append(torch.nn.functional.scaled_dot_product_attention(
                    q[qo:qo+nq].transpose(0, 1)[None].float(),
                    k[ko:ko+nk].transpose(0, 1)[None].float(),
                    v[ko:ko+nk].transpose(0, 1)[None].float(),
                    attn_mask=mask, dropout_p=0.0, is_causal=False,
                    scale=128**-0.5, enable_gqa=k.shape[1] != q.shape[1]).squeeze(0).transpose(0, 1))
                qo += nq
                ko += nk
            outputs[alignment] = torch.cat(rows)
    return outputs


def errors(actual, expected):
    import torch
    # Reduce diagnostic errors in FP64 so finite, wildly wrong BF16 outputs
    # cannot overflow the evidence serializer. Attention oracles remain FP32.
    error = (actual.double()-expected.double()).abs()
    allowed = PROTOCOL['atol'] + PROTOCOL['rtol']*expected.double().abs()
    failures = int((error > allowed).sum())
    rms = error.square().mean().sqrt().item()
    return {'elements': error.numel(), 'mismatched_elements': failures,
            'max_abs': error.max().item(), 'rms': rms,
            'max_relative': (error/expected.double().abs().clamp_min(PROTOCOL['relative_error_denominator_floor'])).max().item(),
            'passes_fixed_limits': failures == 0 and rms <= PROTOCOL['max_rms_error']}


def analytic_ramp(layout, device):
    import torch
    values = {'top_left': [], 'bottom_right': [], 'full_attention': []}
    for nq, nk in zip(layout.query_lengths, layout.key_lengths):
        values['top_left'].extend(i/64 for i in range(nq))
        values['bottom_right'].extend((nk-nq+i)/64 for i in range(nq))
        values['full_attention'].extend([(nk-1)/64]*nq)
    return {name: torch.tensor(v, dtype=torch.float32, device=device)[:, None, None].expand(9, 16, 128)
            for name, v in values.items()}


def persist_tensors(path, tensors):
    import torch
    values = {name: value.detach().cpu().contiguous() for name, value in tensors.items()}
    torch.save(values, path)
    return {name: {'shape': list(value.shape), 'dtype': str(value.dtype),
                   'sha256': hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()}
            for name, value in values.items()}


def worker(output):
    import torch
    sys.path.insert(0, str(ROOT))
    from dspark_qwen.varlen_target import native_varlen
    from torch.nn.attention.varlen import varlen_attn
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError('ROCm/CUDA unavailable; no fallback')
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    recorded_identity = identity()
    result = {'status': 'running', **recorded_identity,
              'torch': torch.__version__, 'hip': torch.version.hip,
              'device': torch.cuda.get_device_name(device),
              'device_arch': getattr(torch.cuda.get_device_properties(device), 'gcnArchName', None),
              'flash_preference': str(torch.backends.cuda.preferred_rocm_fa_library()) if torch.version.hip else None,
              'flash_prefer_ck_env': os.environ.get('TORCH_ROCM_FA_PREFER_CK'),
              'cases': [], 'production_gate_passed': False}
    native_source = Path(sys.modules['torch.nn.attention.varlen'].__file__)
    result['native_source_sha256'] = hashlib.sha256(native_source.read_bytes()).hexdigest()
    save(output/'diagnostic.json', result)
    inputs = prepare_inputs(device)
    layout = inputs['layout']
    input_tensors = {name: value for name, value in inputs.items() if isinstance(value, torch.Tensor)}
    input_tensors.update(cu_query=layout.cu_query, cu_key=layout.cu_key, gather_indices=layout.gather_indices)
    result['inputs'] = persist_tensors(output/'inputs.pt', input_tensors)
    save(output/'diagnostic.json', result)
    native_outputs = {}
    started = time.monotonic()
    for variant in PROTOCOL['variants']:
        for kind in PROTOCOL['inputs']:
            case_name = variant+'__'+kind
            q = inputs['q'] if kind == 'original_random' else torch.zeros_like(inputs['q'])
            k = inputs['k']
            v = inputs['v'] if kind == 'original_random' else inputs['ramp_v']
            if variant == 'public_repeated_kv_no_gqa':
                k, v = k.repeat_interleave(2, dim=1), v.repeat_interleave(2, dim=1)
            row = {'name': case_name, 'variant': variant, 'input': kind,
                   'status': 'prepared', 'q_shape': list(q.shape), 'kv_shape': list(k.shape),
                   'cu_query': layout.cu_query.tolist(), 'cu_key': layout.cu_key.tolist()}
            row['inputs'] = persist_tensors(output/(case_name+'__inputs.pt'), {'q': q, 'k': k, 'v': v})
            result['cases'].append(row)
            save(output/'diagnostic.json', result)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                if variant == 'public_gqa':
                    actual = native_varlen(q, k, v, layout, scale=128**-0.5)
                elif variant == 'public_repeated_kv_no_gqa':
                    actual = varlen_attn(q, k, v, layout.cu_query, layout.cu_key, 8, 29,
                                         scale=128**-0.5, window_size=(-1, 0), enable_gqa=False)
                else:
                    actual = torch.ops.aten._flash_attention_forward(q, k, v,
                        layout.cu_query, layout.cu_key, 8, 29, 0.0, True, False,
                        scale=128**-0.5, window_size_left=None, window_size_right=None)[0]
                torch.cuda.synchronize(device)
            row['operator_counts'] = {event.key: event.count for event in profile.key_averages()}
            row['raw_output'] = persist_tensors(output/(case_name+'.pt'), {'output': actual})
            row['status'] = 'native_output_saved'
            row['finite'] = bool(torch.isfinite(actual).all())
            row['shape_dtype_device_match'] = actual.shape == q.shape and actual.dtype == q.dtype and actual.device == q.device
            save(output/'diagnostic.json', result)
            if not row['finite'] or not row['shape_dtype_device_match']:
                raise AssertionError('Nonfinite or incompatible native output; raw output retained')
            if row['operator_counts'].get('aten::_flash_attention_forward', 0) != 1:
                raise AssertionError('Expected exactly one native flash op; event map retained')
            refs = references(q, k, v, layout)
            row['oracle_tensors'] = persist_tensors(output/(case_name+'__oracles.pt'), refs)
            row['errors'] = {name: errors(actual, expected) for name, expected in refs.items()}
            row['per_request_errors'] = {name: [errors(actual[:1], expected[:1]), errors(actual[1:], expected[1:])]
                                         for name, expected in refs.items()}
            if kind == 'zero_q_position_ramp':
                analytic = analytic_ramp(layout, device)
                row['analytic_errors'] = {name: errors(actual, expected) for name, expected in analytic.items()}
                row['oracle_analytic_errors'] = {name: errors(refs[name], analytic[name]) for name in refs}
            native_outputs[case_name] = actual
            row['status'] = 'observed'
            row['matches'] = [name for name, stats in row['errors'].items() if stats['passes_fixed_limits']]
            save(output/'diagnostic.json', result)
            # Persist every metric before asserting invariant oracle agreement.
            if kind == 'zero_q_position_ramp':
                for name in refs:
                    torch.testing.assert_close(refs[name], analytic[name], atol=2e-6, rtol=1e-5)
    result['gqa_vs_duplicated_kv'] = {kind: errors(native_outputs['public_gqa__'+kind],
        native_outputs['public_repeated_kv_no_gqa__'+kind]) for kind in PROTOCOL['inputs']}
    result['peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
    result['elapsed_seconds'] = time.monotonic()-started
    result['timing_scope'] = 'Six native calls, profiler, oracles, persistence and checks; not a benchmark'
    if identity() != recorded_identity:
        raise RuntimeError('Source or protocol changed during execution')
    result['status'] = 'diagnostic_completed'
    save(output/'diagnostic.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--execute', action='store_true')
    modes.add_argument('--dry-run', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if not args.execute or args.output is None:
            parser.error('Internal worker requires execute/output')
        try:
            worker(args.output)
        except Exception as exc:
            record = args.output/'diagnostic.json'
            result = json.loads(record.read_text()) if record.exists() else {}
            result.update(status='execution_failed', error_type=type(exc).__name__, error=str(exc))
            save(record, result)
            traceback.print_exc()
            return 1
        return 0
    ident = identity()
    if not args.execute:
        print(json.dumps(dict(ident, status='dry_run_no_backend_import_no_gpu'), indent=2))
        return 0
    if args.output is None:
        parser.error('--execute requires a fresh output directory')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    save(output/'identity.json', ident)
    command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--execute', '--output', str(output)]
    started, timed_out = time.monotonic(), False
    with (output/'stdout.log').open('w') as stdout, (output/'stderr.log').open('w') as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2'))
        save(output/'started.json', {'pid': process.pid, 'command': command})
        try:
            code = process.wait(timeout=PROTOCOL['timeout_seconds'])
        except subprocess.TimeoutExpired:
            timed_out = True
            try: os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            code = process.wait()
        except BaseException:
            try: os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            process.wait()
            raise
    valid = code == 0 and not timed_out and identity() == ident
    record = output/'diagnostic.json'
    valid = valid and record.exists() and json.loads(record.read_text()).get('status') == 'diagnostic_completed'
    save(output/'completion.json', {'exit_code': code, 'timed_out': timed_out,
         'wall_seconds': time.monotonic()-started,
         'status': 'diagnostic_completed' if valid else 'execution_failed',
         'scope': 'Execution completion is distinct from production correctness'})
    return 0 if valid else 1


if __name__ == '__main__':
    raise SystemExit(main())
