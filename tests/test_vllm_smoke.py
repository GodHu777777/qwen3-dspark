"""CPU stdlib lifecycle tests; no vLLM import, model loading or GPU execution."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


guard = load('guard_vllm_smoke')
worker = load('probe_vllm_offline')


class FakeGuard:
    def __init__(self, fail_running=False):
        self.fail_running = fail_running

    def snapshot(self, owned, *, before=False):
        if owned and self.fail_running:
            raise RuntimeError('fixture ASR health failure')
        return dict(cpu_fixture=True, ready=True, busy=False)


class SmokeInputTests(unittest.TestCase):
    def test_distribution_and_module_versions_are_separately_pinned(self):
        evidence = worker.version_identity('0.30.0+rocm723', '0.30.0')
        self.assertTrue(evidence['verified'])
        self.assertNotEqual(evidence['distribution_metadata']['actual'], evidence['module']['actual'])
        for distribution, module in [('0.30.0', '0.30.0'), ('0.30.0+rocm723', '0.30.0+rocm723'),
                                     ('0.31.0+rocm723', '0.30.0'), ('0.30.0+rocm723', '0.31.0')]:
            evidence = worker.version_identity(distribution, module)
            self.assertFalse(evidence['verified'])
            self.assertEqual(evidence['distribution_metadata']['actual'], distribution)
            self.assertEqual(evidence['module']['actual'], module)
            self.assertEqual(evidence['distribution_metadata']['expected'], '0.30.0+rocm723')
            self.assertEqual(evidence['module']['expected'], '0.30.0')

    def test_worker_preflight_exact_settings_and_invalid_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / 'model'; model.mkdir()
            (model / 'config.json').write_text('{"vocab_size": 100}')
            request = Path(tmp) / 'request.json'
            request.write_text('{"prompt_token_ids": [1,2,3], "seed": 123}')
            tokens, engine, sampling, identity = worker.prepare(model, request)
            self.assertEqual(tokens, [1, 2, 3])
            self.assertFalse(engine['enforce_eager'])
            self.assertEqual(engine['gpu_memory_utilization'], .18)
            self.assertEqual(engine['generation_config'], 'vllm')
            self.assertEqual(sampling['max_tokens'], 128)
            self.assertTrue(sampling['ignore_eos'])
            self.assertEqual(identity['prompt_tokens'], 3)
            request.write_text('{"prompt_token_ids": [100], "seed": 123}')
            with self.assertRaises(ValueError): worker.prepare(model, request)

    def test_binding_rejects_missing_weights_and_missing_index_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / 'model'; model.mkdir()
            paths = [model / name for name in ('config.json', 'tokenizer_config.json', 'tokenizer.json')]
            for path in paths: path.write_text('{}')
            request = Path(tmp) / 'request.json'; request.write_text('{}')
            paths += [request, Path(worker.__file__).resolve(), Path(guard.__file__).resolve()]
            def binding():
                return dict(files={str(p): guard.sha(p) for p in paths}, worker=str(Path(worker.__file__).resolve()),
                    python=sys.executable, model=str(model), request=str(request), asr_pid=1, asr_url='fixture')
            with self.assertRaisesRegex(ValueError, 'Actual model weights'):
                guard.verify_binding(binding())
            weights = model / 'model.safetensors'; weights.write_bytes(b'fixture'); paths.append(weights)
            guard.verify_binding(binding())
            index = model / 'model.safetensors.index.json'
            index.write_text('{"weight_map":{"x":"missing.safetensors"}}'); paths.append(index)
            with self.assertRaisesRegex(ValueError, 'indexed weight'):
                guard.verify_binding(binding())

    def test_reused_parent_pid_does_not_own_unrelated_descendant(self):
        own_pid = os.getpid()
        old = dict(pid=999991, ppid=own_pid, pgrp=999991, start_ticks=10, state='S')
        reused = dict(old, ppid=1, start_ticks=20)
        child = dict(pid=999992, ppid=999991, pgrp=999991, start_ticks=21, state='S')
        entries = [Path('/proc/999991'), Path('/proc/999992')]
        lookup = {999991: reused, 999992: child}
        owned = {999991: old}
        with patch.object(guard.Path, 'iterdir', return_value=iter(entries)), patch.object(guard, 'process_record', side_effect=lambda p: lookup.get(p)):
            guard.discover_owned(owned, 999991)
        self.assertNotIn(999992, owned)
        self.assertEqual(owned[999991]['start_ticks'], 10)


@unittest.skipUnless(sys.platform.startswith('linux'), 'Linux /proc lifecycle tests')
class SmokeLifecycleTests(unittest.TestCase):
    def run_worker(self, code, *, timeout=2., fail=False):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        output = Path(tmp.name) / 'run'
        result = guard.supervise([sys.executable, '-u', '-c', code], output, FakeGuard(fail),
            timeout=timeout, poll_seconds=.05, grace=.2)
        return result, output

    def test_success_and_real_nonzero_os_exit(self):
        result, output = self.run_worker('print("fixture complete")')
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['worker_os_exit'], 0)
        self.assertTrue(result['released'])
        self.assertIn('fixture complete', (output / 'worker.log').read_text())
        result, _ = self.run_worker('raise SystemExit(7)')
        self.assertEqual(result['status'], 'worker_failed')
        self.assertEqual(result['worker_os_exit'], 7)
        self.assertTrue(result['released'])

    def test_timeout_kills_owned_child_even_after_new_session(self):
        code = 'import subprocess,sys,time; subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"],start_new_session=True); time.sleep(60)'
        result, output = self.run_worker(code, timeout=.4)
        self.assertTrue(result['timed_out'])
        self.assertEqual(result['status'], 'timeout')
        self.assertNotEqual(result['worker_os_exit'], 0)
        self.assertTrue(result['released'])
        self.assertGreaterEqual(len(result['owned_processes']), 2)
        self.assertEqual(result['remaining_owned'], [])
        self.assertTrue(all(not guard.same_process(r) for r in result['owned_processes']))
        self.assertLess(result['elapsed_seconds'], 4.)
        self.assertTrue((output / 'worker-exit.json').is_file())

    def test_orphan_grandchild_is_adopted_and_cleaned_after_successful_parent(self):
        grandchild = 'import time; time.sleep(60)'
        child = 'import subprocess,sys; subprocess.Popen([sys.executable,"-c",' + repr(grandchild) + '],start_new_session=True)'
        code = 'import subprocess,sys,time; subprocess.run([sys.executable,"-c",' + repr(child) + ']); time.sleep(.2)'
        result, _ = self.run_worker(code)
        self.assertEqual(result['worker_os_exit'], 0)
        self.assertTrue(result['released'])
        self.assertGreaterEqual(len(result['owned_processes']), 2)
        self.assertEqual(result['remaining_owned'], [])
        self.assertTrue(result['cleanup_actions'])

    def test_guard_failure_stops_only_owned_tree_and_records_error(self):
        result, output = self.run_worker('import time; time.sleep(60)', fail=True)
        self.assertEqual(result['status'], 'failed')
        self.assertTrue(result['released'])
        self.assertIn('fixture ASR health failure', (output / 'controller-error.txt').read_text())
        self.assertEqual(result['remaining_owned'], [])


if __name__ == '__main__':
    unittest.main()
