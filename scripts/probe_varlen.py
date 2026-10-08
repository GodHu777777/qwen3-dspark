#!/usr/bin/env python3
"""Fixed small-tensor native-varlen gate; default is stdlib-only dry run.

Execution requires an explicitly coordinated GPU window. No model/checkpoint is
loaded and no backend fallback, environment installation or tuning is performed.
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
    'query_heads': 16, 'kv_heads': 8, 'head_dim': 128,
    'window_size': [-1, 0], 'enable_gqa': True,
    'timeout_seconds': 120, 'atol': 0.02, 'rtol': 0.02,
    'max_rms_error': 0.005, 'isolation_atol': 0.0,
    'cases': [
        {'name': 'interleaved_cached_tail', 'markers': [1, 2],
         'q': [1, 8], 'k': [17, 29], 'inactive_k': 13},
        {'name': 'crop_exit_readd', 'markers': [1, 2, 4],
         'q': [3, 2, 1], 'k': [11, 5, 1], 'inactive_k': 0},
        {'name': 'crop_zero_then_append', 'markers': [1],
         'q': [2], 'k': [2], 'inactive_k': 6},
    ],
    'oracles': ['FP32 MATH per-request SDPA with explicit bottom-right mask',
                'FP32 MATH packed SDPA using PackedTarget._attention_payload'],
    'checks': ['finite shape/dtype/device', 'fixed elementwise and RMS limits',
               'zero-Q position-ramp causal sentinel', 'inactive KV poison',
               'other active request QKV poison with unchanged first request',
               'exactly one aten::_flash_attention_forward per native call'],
    'scope': 'Small tensor backend support/numerical gate only; not full model, losslessness, calibration or performance evidence',
}


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def source_identity():
    paths = [Path(__file__).resolve(), *sorted((ROOT / 'dspark_qwen').glob('*.py'))]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def worker(output):
    # All non-stdlib/backend imports live behind explicit execution.
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    sys.path.insert(0, str(ROOT))
    from dspark_qwen.packed_target import PackedTarget
    from dspark_qwen.varlen_target import PackedLayout, native_varlen

    torch.set_num_threads(2)
    torch.manual_seed(PROTOCOL['seed'])
    if not torch.cuda.is_available():
        raise RuntimeError('ROCm/CUDA unavailable; no fallback')
    device, dtype = torch.device('cuda:0'), torch.bfloat16
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    result = {'protocol': PROTOCOL, 'source_identity': source_identity(),
              'torch': torch.__version__, 'hip': torch.version.hip,
              'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name(device),
              'device_arch': getattr(torch.cuda.get_device_properties(device), 'gcnArchName', None),
              'rocm_flash_preferred_library': str(torch.backends.cuda.preferred_rocm_fa_library()) if torch.version.hip else None,
              'rocm_flash_prefer_ck_env': os.environ.get('TORCH_ROCM_FA_PREFER_CK'),
              'backend_identity_scope': 'Native ATen flash dispatch recorded; preferred library is a preference, not an independently traced device-kernel identity',
              'cases': [], 'status': 'running'}
    write_json(output / 'worker-result.json', result)
    started = time.monotonic()

    def metadata(items):
        return torch.tensor(items, dtype=torch.long, device=device)

    def tensor(count):
        # Generate on CPU before copying to make input-generation identity explicit.
        return torch.randn(1, 8, count, 128, dtype=torch.float32).to(device=device, dtype=dtype)

    # Physical cache is interleaved; every request's own positions remain ordered.
    pairs = [(m, p) for p in range(29) for m, n in [(1, 17), (2, 29), (3, 13)] if p < n]
    kr, kp = metadata([m for m, p in pairs]), metadata([p for m, p in pairs])
    key, value = tensor(len(pairs)), tensor(len(pairs))

    def native(q, k, v, layout):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            out = native_varlen(q, k, v, layout, scale=128 ** -0.5)
            torch.cuda.synchronize(device)
        events = {event.key: event.count for event in profile.key_averages()}
        if events.get('aten::_flash_attention_forward', 0) != 1:
            raise RuntimeError(f'Expected one native flash operator, got {events}')
        if out.shape != q.shape or out.dtype != q.dtype or out.device != q.device:
            raise AssertionError('Native output shape/dtype/device differs')
        if not torch.isfinite(out).all():
            raise AssertionError('Nonfinite native output')
        return out, events

    def dense(q, physical_k, physical_v, qr, qp):
        mask, _, _ = PackedTarget._attention_payload(None, qr, qp, kr, kp)
        with sdpa_kernel(SDPBackend.MATH):
            out = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(0, 1)[None].float(), physical_k.float(), physical_v.float(),
                attn_mask=mask['full_attention'], dropout_p=0.0, is_causal=False,
                scale=128 ** -0.5, enable_gqa=True)
        return out.squeeze(0).transpose(0, 1)

    def per_request(q, k, v, layout):
        rows, qo, ko = [], 0, 0
        with sdpa_kernel(SDPBackend.MATH):
            for nq, nk in zip(layout.query_lengths, layout.key_lengths):
                visible = (torch.arange(nk, device=device)[None] <=
                           torch.arange(nk-nq, nk, device=device)[:, None])[None, None]
                rows.append(torch.nn.functional.scaled_dot_product_attention(
                    q[qo:qo+nq].transpose(0, 1)[None].float(),
                    k[ko:ko+nk].transpose(0, 1)[None].float(),
                    v[ko:ko+nk].transpose(0, 1)[None].float(),
                    attn_mask=visible, dropout_p=0.0, is_causal=False,
                    scale=128 ** -0.5, enable_gqa=True).squeeze(0).transpose(0, 1))
                qo += nq
                ko += nk
        return torch.cat(rows)

    def compare(actual, expected):
        error = actual.float() - expected.float()
        stats = {'max_abs': error.abs().max().item(), 'rms': error.square().mean().sqrt().item()}
        torch.testing.assert_close(actual.float(), expected.float(),
                                   atol=PROTOCOL['atol'], rtol=PROTOCOL['rtol'])
        if stats['rms'] > PROTOCOL['max_rms_error']:
            raise AssertionError(f'RMS threshold exceeded: {stats}')
        return stats

    for index, case in enumerate(PROTOCOL['cases']):
        if index:
            if index == 1:
                # Crop requests 1 and 2 to accepted prefixes, exit marker3,
                # then re-add that external request with fresh marker4.
                keep = ((kr == 1) & (kp < 8)) | ((kr == 2) & (kp < 3))
                added = [(1, p) for p in range(8, 11)] + [(2, p) for p in range(3, 5)] + [(4, 0)]
            else:
                keep = kr != 1  # crop request1 to zero, retain inactive 2/4
                added = [(1, 0), (1, 1)]
            key, value = key[:, :, keep], value[:, :, keep]
            kr, kp = kr[keep], kp[keep]
            key = torch.cat([key, tensor(len(added))], dim=2)
            value = torch.cat([value, tensor(len(added))], dim=2)
            kr = torch.cat([kr, metadata([m for m, p in added])])
            kp = torch.cat([kp, metadata([p for m, p in added])])
        qr = metadata([m for m, nq in zip(case['markers'], case['q']) for _ in range(nq)])
        qp = metadata([p for nq, nk in zip(case['q'], case['k']) for p in range(nk-nq, nk)])
        layout = PackedLayout.from_metadata(qr, qp, kr, kp)
        assert list(layout.query_lengths) == case['q'] and list(layout.key_lengths) == case['k']
        assert layout.physical_key_tokens-layout.gathered_key_tokens == case['inactive_k']
        q = torch.randn(sum(case['q']), 16, 128, dtype=torch.float32).to(device=device, dtype=dtype)
        k, v = layout.gather_kv(key, value)
        actual, events = native(q, k, v, layout)
        ref, packed = per_request(q, k, v, layout), dense(q, key, value, qr, qp)
        torch.testing.assert_close(ref, packed, atol=2e-6, rtol=1e-5)
        row = {'name': case['name'], 'q_shape': list(q.shape), 'gathered_k_shape': list(k.shape),
               'physical_k': layout.physical_key_tokens, 'native_operators': events,
               'per_request_error': compare(actual, ref), 'packed_error': compare(actual, packed)}
        # Uniform attention makes each result the visible local-position mean:
        # directly distinguishes bottom-right from erroneous top-left causality.
        ramp = (kp.float()/32)[None, None, :, None].expand_as(value).to(dtype)
        rk, rv = layout.gather_kv(key, ramp)
        sentinel, _ = native(torch.zeros_like(q), rk, rv, layout)
        expected = (qp.float()/64)[:, None, None].expand_as(sentinel)
        row['causal_sentinel_error'] = compare(sentinel, expected)
        # Poison every other request (including inactive), keeping request1 fixed.
        poison_k, poison_v, poison_q = key.clone(), value.clone(), q.clone()
        poison_k[:, :, kr != case['markers'][0]] = 100
        poison_v[:, :, kr != case['markers'][0]] = -100
        poison_q[qr != case['markers'][0]] = 100
        pk, pv = layout.gather_kv(poison_k, poison_v)
        changed, _ = native(poison_q, pk, pv, layout)
        first = case['q'][0]
        torch.testing.assert_close(changed[:first], actual[:first], atol=0, rtol=0)
        row['other_request_isolation_exact'] = True
        if case['inactive_k']:
            inactive = ~torch.isin(kr, metadata(case['markers']))
            poison_k, poison_v = key.clone(), value.clone()
            poison_k[:, :, inactive] = 100
            poison_v[:, :, inactive] = -100
            pk, pv = layout.gather_kv(poison_k, poison_v)
            changed, _ = native(q, pk, pv, layout)
            torch.testing.assert_close(changed, actual, atol=0, rtol=0)
            row['inactive_isolation_exact'] = True
        result['cases'].append(row)
        write_json(output / 'worker-result.json', result)
    torch.cuda.synchronize(device)
    if source_identity() != result['source_identity']:
        raise RuntimeError('Probe source changed during execution')
    implementation = Path(sys.modules['torch.nn.attention.varlen'].__file__).resolve()
    result['native_implementation'] = {'path': str(implementation),
        'sha256': hashlib.sha256(implementation.read_bytes()).hexdigest()}
    result.update(status='passed', peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                  elapsed_seconds=time.monotonic()-started,
                  timing_scope='All tensor cases, CPU profiling, oracles and assertions; not a benchmark')
    write_json(output / 'worker-result.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--dry-run', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if not args.execute or args.output is None:
            parser.error('Internal worker requires execute/output')
        try:
            worker(args.output)
        except Exception as exc:
            write_json(args.output/'failure.json', {'type': type(exc).__name__, 'error': str(exc)})
            traceback.print_exc()
            return 1
        return 0
    protocol_hash = hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()
    identity = {'protocol': PROTOCOL, 'protocol_sha256': protocol_hash,
                'source_identity': source_identity()}
    if not args.execute:
        print(json.dumps(dict(identity, status='dry_run_no_torch_import_no_gpu'), indent=2))
        return 0
    if args.output is None:
        parser.error('--execute requires a fresh --output directory')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output/'identity.json', identity)
    started = time.monotonic()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2')
    command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--execute', '--output', str(args.output)]
    timed_out = False
    with (args.output/'stdout.log').open('w') as stdout, (args.output/'stderr.log').open('w') as stderr:
        process = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
        write_json(args.output/'started.json', {'pid': process.pid, 'command': command})
        try:
            code = process.wait(timeout=PROTOCOL['timeout_seconds'])
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            code = process.wait()
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise
    valid = code == 0 and not timed_out and source_identity() == identity['source_identity']
    report = args.output/'worker-result.json'
    valid = valid and report.exists() and json.loads(report.read_text()).get('status') == 'passed'
    write_json(args.output/'completion.json', {'exit_code': code, 'timed_out': timed_out,
               'wall_seconds': time.monotonic()-started, 'status': 'passed' if valid else 'failed'})
    return 0 if valid else 1


if __name__ == '__main__':
    raise SystemExit(main())
