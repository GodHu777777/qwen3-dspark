"""Stdlib-only controller checks; never import Torch or execute a GPU worker."""
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
from unittest.mock import patch, Mock


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/probe_varlen.py'


class ProbeControllerTests(unittest.TestCase):
    def setUp(self):
        self.main = runpy.run_path(str(SCRIPT))['main']

    def test_default_dry_run_does_not_import_backend(self):
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split('.')[0] in {'torch', 'transformers', 'dspark_qwen'}:
                raise AssertionError('Unexpected backend import: ' + name)
            return original(name, *args, **kwargs)
        stream = io.StringIO()
        with patch('sys.argv', [str(SCRIPT)]), patch('builtins.__import__', guarded), contextlib.redirect_stdout(stream):
            self.assertEqual(self.main(), 0)
        result = json.loads(stream.getvalue())
        self.assertEqual(result['status'], 'dry_run_no_torch_import_no_gpu')
        self.assertEqual(result['protocol']['timeout_seconds'], 120)

    def test_timeout_kills_only_owned_group_and_records_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'run'
            process = Mock(pid=123456)
            process.wait.side_effect = [subprocess.TimeoutExpired('mock', 120), -9]
            with patch('sys.argv', [str(SCRIPT), '--execute', '--output', str(output)]), patch('subprocess.Popen', return_value=process) as launch, patch('os.killpg') as kill:
                self.assertEqual(self.main(), 1)
            self.assertTrue(launch.call_args.kwargs['start_new_session'])
            kill.assert_called_once_with(123456, signal.SIGKILL)
            result = json.loads((output/'completion.json').read_text())
            self.assertTrue(result['timed_out'])
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['exit_code'], -9)

    def test_interrupt_reaps_owned_child_before_propagating(self):
        with tempfile.TemporaryDirectory() as tmp:
            process = Mock(pid=123456)
            process.wait.side_effect = [KeyboardInterrupt(), -9]
            with patch('sys.argv', [str(SCRIPT), '--execute', '--output', str(Path(tmp)/'run')]), patch('subprocess.Popen', return_value=process), patch('os.killpg') as kill:
                with self.assertRaises(KeyboardInterrupt):
                    self.main()
            kill.assert_called_once_with(123456, signal.SIGKILL)
            self.assertEqual(process.wait.call_count, 2)


if __name__ == '__main__':
    unittest.main()
