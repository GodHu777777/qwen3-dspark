"""CPU-only diagnostic/controller tests. Native GPU functions are never called."""
import builtins
import contextlib
import io
import json
from pathlib import Path
import runpy
import signal
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

SCRIPT = Path(__file__).resolve().parents[1]/'scripts/diagnose_varlen.py'


class DiagnosticControllerTests(unittest.TestCase):
    def setUp(self):
        self.module = runpy.run_path(str(SCRIPT))

    def test_default_is_guarded_stdlib_only(self):
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split('.')[0] in {'torch', 'transformers', 'dspark_qwen'}:
                raise AssertionError('Unexpected backend import: '+name)
            return original(name, *args, **kwargs)
        stream = io.StringIO()
        with patch('sys.argv', [str(SCRIPT)]), patch('builtins.__import__', guarded), contextlib.redirect_stdout(stream):
            self.assertEqual(self.module['main'](), 0)
        result = json.loads(stream.getvalue())
        self.assertEqual(result['status'], 'dry_run_no_backend_import_no_gpu')
        self.assertEqual(result['protocol']['native_calls'], 6)
        self.assertEqual(result['protocol']['timeout_seconds'], 120)

    def test_timeout_and_interrupt_reap_only_own_child(self):
        for failure in [subprocess.TimeoutExpired('mock', 120), KeyboardInterrupt()]:
            with self.subTest(failure=type(failure).__name__), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)/'run'
                process = Mock(pid=123456)
                process.wait.side_effect = [failure, -9]
                with patch('sys.argv', [str(SCRIPT), '--execute', '--output', str(out)]), patch('subprocess.Popen', return_value=process) as launch, patch('os.killpg') as kill:
                    if isinstance(failure, KeyboardInterrupt):
                        with self.assertRaises(KeyboardInterrupt): self.module['main']()
                    else:
                        self.assertEqual(self.module['main'](), 1)
                        result = json.loads((out/'completion.json').read_text())
                        self.assertTrue(result['timed_out'])
                        self.assertEqual(result['status'], 'execution_failed')
                self.assertTrue(launch.call_args.kwargs['start_new_session'])
                kill.assert_called_once_with(123456, signal.SIGKILL)
                self.assertEqual(process.wait.call_count, 2)


class DiagnosticOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(2)
        cls.torch = torch
        cls.module = runpy.run_path(str(SCRIPT))

    def setUp(self):
        self.inputs = self.module['prepare_inputs']('cpu')

    def test_original_seed_and_draw_order_are_identical(self):
        t = self.torch
        t.manual_seed(20261009)
        key = t.randn(1, 8, 59, 128, dtype=t.float32).to(t.bfloat16)
        value = t.randn(1, 8, 59, 128, dtype=t.float32).to(t.bfloat16)
        query = t.randn(9, 16, 128, dtype=t.float32).to(t.bfloat16)
        for name, expected in [('key', key), ('value', value), ('q', query)]:
            self.assertTrue(t.equal(self.inputs[name], expected))
        layout = self.inputs['layout']
        self.assertEqual(layout.cu_query.tolist(), [0, 1, 9])
        self.assertEqual(layout.cu_key.tolist(), [0, 17, 46])
        self.assertEqual(layout.physical_key_tokens, 59)

    def test_ramp_oracles_distinguish_alignment(self):
        t, data = self.torch, self.inputs
        refs = self.module['references'](t.zeros_like(data['q']), data['k'], data['ramp_v'], data['layout'])
        analytic = self.module['analytic_ramp'](data['layout'], 'cpu')
        for name in refs:
            t.testing.assert_close(refs[name], analytic[name], atol=2e-6, rtol=1e-5)
            other = 'bottom_right' if name == 'top_left' else 'top_left'
            self.assertTrue(self.module['errors'](refs[name], analytic[name])['passes_fixed_limits'])
            self.assertFalse(self.module['errors'](refs[name], analytic[other])['passes_fixed_limits'])
        self.assertEqual(analytic['top_left'][0, 0, 0].item(), 0)
        self.assertEqual(analytic['bottom_right'][0, 0, 0].item(), .25)

    def test_gqa_and_repeated_kv_have_same_reference(self):
        t, data = self.torch, self.inputs
        base = self.module['references'](data['q'], data['k'], data['v'], data['layout'])
        repeated = self.module['references'](data['q'], data['k'].repeat_interleave(2, dim=1),
                                              data['v'].repeat_interleave(2, dim=1), data['layout'])
        for name in base:
            t.testing.assert_close(base[name], repeated[name], atol=2e-6, rtol=1e-5)

    def test_raw_tensors_and_metrics_survive_numerical_failure(self):
        t, data = self.torch, self.inputs
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            actual = t.zeros_like(data['q'])
            record = self.module['persist_tensors'](path/'raw.pt', {'output': actual})
            stats = self.module['errors'](actual, t.ones_like(actual))
            self.module['save'](path/'metrics.json', {'raw_output': record, 'errors': stats})
            self.assertFalse(stats['passes_fixed_limits'])
            self.assertEqual(stats['mismatched_elements'], actual.numel())
            saved = t.load(path/'raw.pt', weights_only=True)
            self.assertTrue(t.equal(saved['output'], actual))
            self.assertEqual(json.loads((path/'metrics.json').read_text())['errors']['max_abs'], 1)

    def test_large_finite_error_is_json_serializable(self):
        t = self.torch
        actual = t.full((2,), 1e30, dtype=t.bfloat16)
        stats = self.module['errors'](actual, t.zeros_like(actual))
        self.assertFalse(stats['passes_fixed_limits'])
        json.dumps(stats, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
