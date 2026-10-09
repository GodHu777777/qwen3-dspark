#!/usr/bin/env python3
"""CPU-only audit of saved Q1/Q8 setup tensors; never loads a model."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(2)
    root = args.evidence_root
    complete = root / 'complete'
    runtime = json.loads((complete / 'run/worker/runtime.json').read_text())
    checks = 0
    def check(condition, label):
        nonlocal checks
        checks += 1
        if not condition:
            raise AssertionError(label)
    archive = root / 'evidence-verified.tar.gz'
    expected_archive = (root / 'archive-remote.log').read_text().split()[0]
    check(hashlib.sha256(archive.read_bytes()).hexdigest() == expected_archive, 'archive SHA')
    caps = runtime['capture']['captures']
    check([c['query_tokens'] for c in caps] == [1, 8], 'exact Q1/Q8 captures')
    rows = []
    for cap in caps:
        validation = cap['validation']
        artifact = validation['tensors']
        path = complete / 'run/worker/setup-validation' / artifact['file']
        check(hashlib.sha256(path.read_bytes()).hexdigest() == artifact['sha256'], 'tensor SHA')
        tensors = torch.load(path, map_location='cpu', weights_only=True)
        check(len(tensors) == 18, '18 saved tensors')
        q = cap['query_tokens']
        check(tensors['input_ids'].shape == (1, q), 'input geometry')
        check(tensors['positions'].shape == (q,), 'position geometry')
        check(tensors['cu_query'].tolist() == [0, q], 'query cumulative lengths')
        check(validation['limits'] == dict(atol=.02, rtol=.02, max_rms=.005), 'frozen limits')
        check(cap['status'] == 'validated' and validation['status'] == 'passed', 'worker validation')
        check(validation['execution']['actual_gpu_graph'] and validation['execution']['python_forward_deltas'] == [0]*29, 'actual replay')
        pairs = []
        for i, layer in enumerate([1, 7, 14, 21, 26]):
            pairs.append((f'selected_raw_layer_{layer}', tensors['replay_context'][..., i*1024:(i+1)*1024], tensors['eager_context'][..., i*1024:(i+1)*1024]))
        for key in ['final_norm', 'logits']:
            pairs.append((key, tensors['replay_'+key], tensors['eager_'+key]))
        for layer in range(28):
            for key in ['scratch_keys', 'scratch_values']:
                pairs.append((f'{key}_layer_{layer}', tensors['replay_'+key][layer], tensors['eager_'+key][layer]))
        comparisons = {}
        for label, actual, expected in pairs:
            check(actual.shape == expected.shape and actual.dtype == expected.dtype, label+' shape/dtype')
            finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
            difference = (actual.double()-expected.double()).abs()
            rms = float(difference.square().mean().sqrt())
            max_abs = float(difference.max())
            passed = finite and bool((difference <= .02+.02*expected.double().abs()).all()) and rms <= .005
            equal = torch.equal(actual, expected)
            byte_equal = torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))
            check(passed, label+' frozen numerical threshold')
            original = validation['comparisons'][label]
            check(original == dict(finite=finite, passed=passed, max_abs=max_abs, rms=rms, bit_equal=equal), label+' original metrics')
            comparisons[label] = dict(max_abs=max_abs, rms=rms, passed=passed, torch_equal=equal, storage_bytes_equal=byte_equal)
        resident = {}
        for key in ['keys', 'values']:
            before = tensors['resident_'+key+'_before']; after = tensors['resident_'+key+'_after']
            check(before.shape == after.shape == (28, 1, 384, 8, 128), 'resident geometry')
            check(torch.equal(before, after), 'resident unchanged')
            resident[key] = dict(torch_equal=True, storage_bytes_equal=torch.equal(before.contiguous().view(torch.uint8), after.contiguous().view(torch.uint8)))
        rows.append(dict(query_tokens=q, tensor_sha256=artifact['sha256'], comparisons=comparisons, resident=resident))
    report = dict(passed=True, assertions=checks, artifact_count=2, saved_tensor_count=36,
                  numerical_comparison_count=126, archive_sha256=expected_archive,
                  limits=dict(atol=.02, rtol=.02, max_rms=.005), captures=rows,
                  scope='Saved native eager versus first graph replay at setup only; CPU tensor audit, no model execution.',
                  exact_equality_note='torch.equal and byte equality are descriptive diagnostics, not tightened numerical pass thresholds; resident preservation is an existing exact invariant.')
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'captures'}))


if __name__ == '__main__':
    main()
