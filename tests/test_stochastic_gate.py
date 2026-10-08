"""Dry-run binding/failure persistence and CPU tiny-Qwen gate integration."""
import argparse
import builtins
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from dspark_qwen import eval_stochastic_gate as gate


def fixture(root):
    model, checkpoint = root / 'model', root / 'checkpoint'
    model.mkdir(); checkpoint.mkdir()
    (model / 'config.json').write_text(json.dumps(dict(model_type='qwen3', vocab_size=4)))
    (model / 'model.safetensors').write_bytes(b'identity-only fixture, never model-loaded')
    (checkpoint / 'draft.safetensors').write_bytes(b'identity-only checkpoint')
    rows = [dict(id='train-id', split='train', accepted=True, prompt_token_ids=[0, 1]),
            dict(id='dev1-id', split='validation', accepted=True, prompt_token_ids=[1, 2, 3]),
            dict(id='dev2-id', split='validation', accepted=True, prompt_token_ids=[2])]
    records = root / 'records.jsonl'
    records.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    metadata = dict(record_id='train-id', draft_config=dict(block_size=3),
        draft_weights_sha256=gate.sha256(checkpoint / 'draft.safetensors'),
        identity=dict(config=dict(model=str(model), records=str(records)),
            records_sha256=gate.sha256(records),
            target_fingerprint={p.name: gate.sha256(p) for p in model.iterdir()}))
    (checkpoint / 'metadata.json').write_text(json.dumps(metadata))
    prior = root / 'prior.json'
    prior.write_text(json.dumps(dict(identity=dict(args=dict(checkpoint=str(checkpoint)),
        checkpoint_weights_sha256=metadata['draft_weights_sha256'],
        checkpoint_metadata_sha256=gate.sha256(checkpoint / 'metadata.json')),
        examples=[dict(id=r['id'], split=r['split'], prompt_tokens=len(r['prompt_token_ids'])) for r in rows])))
    args = argparse.Namespace(prior_gate=str(prior), checkpoint=None, temperature=1.0,
        seeds=[20261009], max_new_tokens=4, probe_rounds=1, timeout_seconds=60)
    return args


class GateBindingTests(unittest.TestCase):
    def test_worker_launcher_captures_native_streams_and_times_out_owned_child(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            source = out / 'source' / 'dspark_qwen'
            source.mkdir(parents=True)
            (source / '__init__.py').write_text('')
            worker = source / 'eval_stochastic_gate.py'
            worker.write_text("import os\nprint('python stdout', flush=True)\nos.write(2, b'native stderr\\n')\n")
            self.assertEqual(gate.launch_worker(out, 5), 0)
            log = (out / 'stdout.log').read_text()
            self.assertIn('python stdout', log)
            self.assertIn('native stderr', log)
            worker.write_text("import time\nprint('before timeout', flush=True)\ntime.sleep(60)\n")
            with self.assertRaises(subprocess.TimeoutExpired):
                gate.launch_worker(out, 0.5)
            self.assertIn('before timeout', (out / 'stdout.log').read_text())

    def test_launcher_failure_keeps_worker_partial_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = fixture(root)
            out = root / 'run'
            def fail(output, timeout):
                report = json.loads((output / 'result.json').read_text())
                report.update(status='running', partial_marker='preserve this evidence')
                gate.save_report(output, report)
                raise subprocess.TimeoutExpired('owned worker', timeout)
            with patch.object(gate, 'launch_worker', side_effect=fail):
                self.assertEqual(gate.main(['--prior-gate', args.prior_gate, '--output', str(out)]), 1)
            report = json.loads((out / 'result.json').read_text())
            self.assertEqual(report['partial_marker'], 'preserve this evidence')
            self.assertEqual(report['status'], 'failed')
            self.assertEqual(report['error_type'], 'TimeoutExpired')

    def test_dry_run_never_imports_torch_or_launches_worker_and_copies_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = fixture(root)
            original_import = builtins.__import__
            def reject_gpu_import(name, *positional, **keywords):
                if name == 'torch' or name.startswith('torch.') or name == 'transformers':
                    raise AssertionError('Dry-run attempted a model/runtime import')
                return original_import(name, *positional, **keywords)
            output = root / 'dry-run'
            with patch('builtins.__import__', side_effect=reject_gpu_import), patch.object(gate.subprocess, 'Popen') as launch:
                result = gate.main(['--prior-gate', args.prior_gate, '--output', str(output), '--dry-run'])
                self.assertEqual(result, 0)
                launch.assert_not_called()
            binding = json.loads((output / 'run.json').read_text())
            self.assertEqual(len(binding['cases']), 3)
            gate.verify_binding(binding, output / 'source' / 'dspark_qwen')
            report = json.loads((output / 'result.json').read_text())
            self.assertEqual(report['status'], 'dry_run_complete')
            self.assertIsNone(report['execution_checks_passed'])
            aggregate = json.loads((output / 'aggregate.json').read_text())
            self.assertNotIn('cases', aggregate)
            self.assertNotIn('prompt_token_ids', json.dumps(aggregate))
            self.assertEqual(output.stat().st_mode & 0o777, 0o700)
            with self.assertRaises(FileExistsError):
                gate.main(['--prior-gate', args.prior_gate, '--output', str(output), '--dry-run'])
            snapshot = output / 'source' / 'dspark_qwen' / 'eval_stochastic_gate.py'
            snapshot.write_text(snapshot.read_text() + '\n# changed\n')
            with self.assertRaisesRegex(ValueError, 'Bound input changed'):
                gate.verify_binding(binding, snapshot.parent)

    def test_corrupt_inputs_and_changed_case_selection_fail_before_execution(self):
        for corruption in ('weights', 'target', 'records', 'prior_cases', 'checkpoint_identity'):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                args = fixture(root)
                if corruption == 'weights':
                    (root / 'checkpoint' / 'draft.safetensors').write_bytes(b'changed')
                elif corruption == 'target':
                    (root / 'model' / 'model.safetensors').write_bytes(b'changed')
                elif corruption == 'records':
                    with (root / 'records.jsonl').open('a') as stream:
                        stream.write(json.dumps(dict(id='test', split='test')) + '\n')
                else:
                    prior = json.loads(Path(args.prior_gate).read_text())
                    if corruption == 'prior_cases': prior['examples'].pop()
                    else: prior['identity']['checkpoint_metadata_sha256'] = '0' * 64
                    Path(args.prior_gate).write_text(json.dumps(prior))
                with self.assertRaises(ValueError): gate.bind_inputs(args)
                output = root / 'failed'
                self.assertEqual(gate.main(['--prior-gate', args.prior_gate, '--output', str(output), '--dry-run']), 1)
                self.assertEqual(json.loads((output / 'result.json').read_text())['status'], 'failed')
                self.assertTrue((output / 'stdout.log').exists())
                self.assertTrue((output / 'launcher-error.txt').exists())

    def test_bound_limits_and_aggregate_does_not_turn_numeric_differences_into_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            args = fixture(Path(directory))
            for field, value in [('max_new_tokens', 33), ('temperature', 0), ('seeds', [1, 1]),
                                 ('probe_rounds', 3), ('timeout_seconds', 0)]:
                changed = argparse.Namespace(**vars(args)); setattr(changed, field, value)
                with self.assertRaises(ValueError): gate.probability_protocol(changed)
        report = dict(status='completed', execution_checks_passed=True, runs=[], checks=[dict(passed=True)],
            numerical_probes=[dict(total_variation=0.1, max_absolute_logit_difference=0.5, argmax_equal=False)])
        summary = gate.aggregate(report)
        self.assertTrue(summary['execution_checks_passed'])
        self.assertEqual(summary['native_numerical_fidelity']['nonzero_tv_rows'], 1)
        self.assertNotIn('passed', summary['native_numerical_fidelity'])
        for tokens, budget, eos, prompt, length in [([], 2, (), 2, 2), ([1, 2], 2, (1,), 2, 3),
                                                   ([1], 2, (), 2, 2), ([1], 1, (), 2, 3)]:
            with self.assertRaises(RuntimeError): gate.check_output(tokens, budget, eos, prompt, length)


@unittest.skipUnless(importlib.util.find_spec('torch') is not None, 'torch CPU runtime required')
class GateRuntimeTests(unittest.TestCase):
    def setUp(self):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from dspark_qwen.config import DraftConfig
        from dspark_qwen.model import DSparkDraft
        torch.manual_seed(212)
        config = Qwen3Config(vocab_size=32, hidden_size=32, intermediate_size=64,
            num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, max_position_embeddings=128, attention_dropout=0.0, eos_token_id=None)
        config._attn_implementation = 'sdpa'
        self.model = Qwen3ForCausalLM(config).eval()
        self.draft = DSparkDraft(self.model, DraftConfig(layer_ids=(0, 2), num_layers=2,
            block_size=3, markov_rank=8, mask_token_id=31)).eval()
        self.binding = dict(protocol=dict(max_new_tokens=4, temperature=0.8, seeds=[19], probe_rounds=1),
            cases=[dict(index=i, split='train' if i == 0 else 'validation', prompt_token_ids=[1, 2, i + 3]) for i in range(3)])

    def test_tiny_real_qwen_complete_repeats_interleaving_and_probe_rows(self):
        import torch
        with tempfile.TemporaryDirectory() as directory, torch.no_grad():
            out = Path(directory)
            report = dict(status='running', runs=[], checks=[], numerical_probes=[])
            gate.exercise(self.model, self.draft, self.binding, out, report, device='cpu', amp=False)
            self.assertEqual(len(report['runs']), 9)
            self.assertEqual(len(report['checks']), 6)
            self.assertTrue(all(row['passed'] for row in report['checks']))
            self.assertEqual([row['case'] for row in report['runs'][-3:]], [1, 0, 2])
            self.assertEqual(len(report['numerical_probes']), 12)  # n=3 + bonus, three cases.
            self.assertLess(max(row['total_variation'] for row in report['numerical_probes']), 1e-6)
            self.assertLess(max(row['max_absolute_logit_difference'] for row in report['numerical_probes']), 1e-5)
            payloads = list(out.glob('private-probe-*.pt'))
            self.assertEqual(len(payloads), 3)
            for path in payloads:
                payload = torch.load(path, weights_only=True)
                self.assertEqual(payload['actual_q'].dtype, torch.float64)
                self.assertEqual(len(payload['sequential_probs']), 4)
                self.assertEqual(len(payload['semantic_committed_output_prefix']), 1)

    def test_probe_failure_preserves_execution_result_and_partial_rows(self):
        import torch
        original = gate.compare_probe
        def fail_after_one(model, ids, output, emitted, payload, temperature, on_row):
            def fail(metric):
                on_row(metric)
                raise RuntimeError('injected numerical probe failure')
            return original(model, ids, output, emitted, payload, temperature, fail)
        with tempfile.TemporaryDirectory() as directory, torch.no_grad():
            out = Path(directory)
            report = dict(status='running', runs=[], checks=[], numerical_probes=[])
            with patch.object(gate, 'compare_probe', side_effect=fail_after_one):
                with self.assertRaisesRegex(RuntimeError, 'injected numerical probe failure'):
                    gate.exercise(self.model, self.draft, self.binding, out, report, device='cpu', amp=False)
            saved = json.loads((out / 'result.json').read_text())
            self.assertEqual(len(saved['runs']), 1)
            self.assertEqual(len(saved['numerical_probes']), 1)
            self.assertTrue((out / 'private-rounds.jsonl').read_text())
            payload = torch.load(next(out.glob('private-probe-*.pt')), weights_only=True)
            self.assertEqual(len(payload['sequential_probs']), 1)


if __name__ == '__main__':
    unittest.main()
