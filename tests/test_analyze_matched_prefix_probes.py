"""Synthetic CPU tests; never load the six historical probe payloads."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import torch

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/analyze_matched_prefix_probes.py'
spec = importlib.util.spec_from_file_location('matched_probe_analysis', SCRIPT)
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def fixture():
    p = torch.tensor([.125, .375, .5], dtype=torch.float64)
    q = torch.tensor([.25, .25, .5], dtype=torch.float64)
    seq = torch.tensor([.25, .5, .25], dtype=torch.float64)
    logits = torch.tensor([0., 1., 2.], dtype=torch.bfloat16)
    sl = torch.tensor([.25, 1., 1.5], dtype=torch.bfloat16)
    payload = dict(actual_q=q.repeat(7, 1), block_probs=p.repeat(8, 1),
        block_logits=logits.repeat(8, 1), proposal_tokens=torch.tensor([0,1,2,0,1,2,0]),
        sequential_probs=[seq.clone() for _ in range(8)],
        sequential_logits=[sl.clone() for _ in range(8)], semantic_committed_output_prefix=[2])
    probe = dict(step=128, ordinal=0, proposal_tokens=[0,1,2,0,1,2,0], initial_output=[2],
        selected_q0=.25, selected_p0=.125, historical_alpha=.5, first_prefix_label=0,
        numerical_control=dict(total_variation=.25, max_probability_difference=.25,
            max_absolute_logit_difference=.5, mean_absolute_logit_difference=.25, argmax_equal=False))
    return payload, probe


class PayloadTests(unittest.TestCase):
    def test_complete_payload_and_detached_row_copies(self):
        payload, probe = fixture()
        rows, private = a.validate_payload(payload, probe, vocab=3)
        self.assertEqual(private['probability_rows_validated'], 23)
        self.assertEqual(private['historical_alpha'], .5)
        payload['actual_q'][0, 0] = 0
        self.assertEqual(float(rows['q'][0]), .25)

    def test_invalid_schema_laws_logits_tokens_prefix_and_gather(self):
        def missing(p, b): p.pop('sequential_logits')
        def dtype(p, b): p['actual_q'] = p['actual_q'].float()
        def shape(p, b): p['block_probs'] = p['block_probs'][:7]
        def later_nan(p, b): p['actual_q'][6, 1] = float('nan')
        def later_mass(p, b): p['sequential_probs'][7][1] += 1e-8
        def negative(p, b): p['block_probs'][7, 0] = -.01
        def incomplete(p, b): p['sequential_probs'].pop()
        def bad_logits(p, b): p['sequential_logits'][7][0] = float('inf')
        def logits_dtype(p, b): p['block_logits'] = p['block_logits'].float()
        def tokens(p, b): p['proposal_tokens'][6] = 2
        def prefix(p, b): p['semantic_committed_output_prefix'] = [1]
        def gather(p, b): b['selected_q0'] += 1e-15
        for mutate in (missing, dtype, shape, later_nan, later_mass, negative,
                       incomplete, bad_logits, logits_dtype, tokens, prefix, gather):
            with self.subTest(mutation=mutate.__name__):
                p, b = fixture(); mutate(p, b)
                with self.assertRaises(ValueError): a.validate_payload(p, b, vocab=3)

    def test_historical_reductions_tolerant_but_maxima_exact(self):
        p, b = fixture()
        for key in ('total_variation', 'mean_absolute_logit_difference'):
            tolerant = copy.deepcopy(b); tolerant['numerical_control'][key] += 5e-13
            a.validate_payload(p, tolerant, vocab=3)
            tolerant['numerical_control'][key] += 2e-12
            with self.assertRaisesRegex(ValueError, 'reconciliation'):
                a.validate_payload(p, tolerant, vocab=3)
        for key in ('max_probability_difference', 'max_absolute_logit_difference'):
            exact = copy.deepcopy(b); exact['numerical_control'][key] += 1e-15
            with self.assertRaisesRegex(ValueError, 'exact numerical'):
                a.validate_payload(p, exact, vocab=3)
        b['numerical_control']['argmax_equal'] = True
        with self.assertRaises(ValueError): a.validate_payload(p, b, vocab=3)


class MathematicsTests(unittest.TestCase):
    def test_full_vocabulary_overlap_and_mass_identity(self):
        # Selected token alone says nothing about the remaining two coordinates.
        p = torch.tensor([.1, .6, .3], dtype=torch.float64)
        q = torch.tensor([.1, .2, .7], dtype=torch.float64)
        p[2] += 4e-13; q[2] -= 3e-13
        a.law(p, (3,)); a.law(q, (3,))
        result = a.overlap(p, q)
        self.assertAlmostEqual(result['overlap'], .6 + 4e-13, places=14)
        self.assertAlmostEqual(result['mass_residual'], 5e-14, places=15)
        self.assertLessEqual(abs(result['algebra_error']), a.TOL)
        self.assertNotEqual(result['mass_residual'], 0.)

    def test_alternate_teacher_sign_reversal_and_pair_bound(self):
        def t(x): return torch.tensor(x, dtype=torch.float64)
        rows = {128:dict(q=t([.8,.2]),seq=t([.9,.1]),block=t([.9,.1])),
                512:dict(q=t([.2,.8]),seq=t([.1,.9]),block=t([.1,.9])),
                1280:dict(q=t([.5,.5]),seq=t([.6,.4]),block=t([.6,.4]))}
        # Legal small mass defects must enter each teacher bound explicitly.
        rows[512]['seq'][0] += 4e-13
        result = a.compare_state(0, rows)
        self.assertFalse(result['all_sequential_laws_byte_equal'])
        change = result['changes'][0]
        self.assertEqual(change['sensitivity_status'], 'numerically_sensitive')
        self.assertEqual([x['sign'] for x in change['saved_sequential_teacher_sensitivity']], [-1,1,-1])
        self.assertAlmostEqual(change['common_reference_delta'], -.6)
        self.assertAlmostEqual(change['pair_controls']['block']['own_pair_delta'], 0.)
        correction = result['checkpoints'][1]['teacher_sensitivity']['seq']['mass_defect_correction']
        self.assertGreater(correction, 0.)
        for item in result['checkpoints']:
            for sensitivity in item['teacher_sensitivity'].values():
                self.assertEqual(sensitivity['bound'], sensitivity['tv_to_reference'] + sensitivity['mass_defect_correction'] + a.TOL)
        for change in result['changes']:
            for control in change['pair_controls'].values():
                self.assertLessEqual(control['absolute_delta_shift'], control['bound'])
        serialized = json.dumps(result)
        for forbidden in ('proposal_tokens','selected_q0','selected_p0','historical_alpha','seed','prompt_id'):
            self.assertNotIn(forbidden, serialized)

    def test_byte_equality_is_stricter_than_value_equality(self):
        p = torch.tensor([0., 1.], dtype=torch.float64)
        q = torch.tensor([-0., 1.], dtype=torch.float64)
        self.assertTrue(torch.equal(p,q)); self.assertFalse(a.byte_equal(p,q))


class ExecutionTests(unittest.TestCase):
    def test_default_cli_does_not_import_torch_or_execute(self):
        code = '''import builtins, importlib.util, pathlib, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'): raise AssertionError('torch imported')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
spec = importlib.util.spec_from_file_location('analyzer', sys.argv[1])
a = importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
a.bind = lambda *args: dict(probes=[dict(size_bytes=1)]*6, source_sha256={})
a.supervise = lambda *args: (_ for _ in ()).throw(AssertionError('execution'))
assert a.main(['--probe-root','unused','--quality-root','unused','--historical-source','unused']) == 0
'''
        result = subprocess.run([sys.executable, '-c', code, str(SCRIPT)], capture_output=True, text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads(result.stdout)['status'], 'dry_run_no_tensor_load')

    def test_gap_preserves_first_validated_payload_and_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp); probes=[]
            for index in range(3):
                payload, probe = fixture()
                if index == 1: payload['actual_q'][6,0] = float('nan')
                path = out/f'synthetic{index}.pt'; torch.save(payload, path)
                probe.update(path=str(path), sha256=a.sha(path)); probes.append(probe)
            validate = a.validate_payload
            with patch.object(a, 'validate_payload', side_effect=lambda p,b:validate(p,b,vocab=3)), patch.object(torch, 'load', wraps=torch.load) as loader:
                with self.assertRaisesRegex(ValueError, 'probability entries'):
                    a.analyze(dict(probes=probes),out)
            self.assertEqual(loader.call_count,2)
            self.assertTrue(all(c.kwargs == dict(map_location='cpu', weights_only=True) for c in loader.call_args_list))
            private = a.read(out/'private-validation.json'); public = a.read(out/'summary.json')
            self.assertEqual(private['validated_probes'],1)
            self.assertEqual(private['status'],'evidence_gap')
            self.assertEqual(len(private['rows']),1)
            self.assertEqual(public['validated_probes'],1)
            self.assertNotIn('selected_q0',json.dumps(public))

    def test_cooperative_deadline_saves_gap_before_load(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(torch, 'load') as loader:
            with self.assertRaises(TimeoutError):
                a.analyze(dict(probes=[{}]),tmp,deadline=time.monotonic()-1)
            loader.assert_not_called()
            self.assertEqual(a.read(Path(tmp)/'summary.json')['status'],'partial_deadline')

    def test_resource_caps(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(a, 'peak_rss',return_value=a.CAPS['rss_bytes']+1):
                with self.assertRaisesRegex(ValueError,'RSS'): a.checkpoint(time.monotonic()+5,tmp)
            (Path(tmp)/'extra').write_bytes(b'123')
            with patch.dict(a.CAPS,output_bytes=2):
                with self.assertRaisesRegex(ValueError,'output byte'): a.checkpoint(time.monotonic()+5,tmp)

    def test_supervisor_enforces_output_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'fresh'
            with patch.object(a.subprocess,'Popen') as popen, patch.object(a.subprocess,'check_output',return_value='1'), patch.dict(a.CAPS,output_bytes=0):
                child=popen.return_value; child.poll.return_value=None; child.pid=123; child.wait.return_value=-9
                self.assertEqual(a.supervise({},out),1)
                child.kill.assert_called_once(); child.wait.assert_called_once()
            self.assertEqual(a.read(out/'supervision.json')['limit_reason'],'output_limit')

    def test_supervisor_kills_and_reaps_only_its_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'fresh'
            with patch.object(a.subprocess,'Popen') as popen, patch.object(a.subprocess,'check_output',return_value='1'), patch.dict(a.CAPS,timeout_seconds=0):
                child=popen.return_value; child.poll.return_value=None; child.pid=123; child.wait.return_value=-9
                self.assertEqual(a.supervise({},out),1)
                child.kill.assert_called_once(); child.wait.assert_called_once()
                kwargs=popen.call_args.kwargs
                self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],'')
                self.assertEqual(kwargs['env']['HIP_VISIBLE_DEVICES'],'')
            record=a.read(out/'supervision.json')
            self.assertEqual(record['worker_os_exit'],-9)
            self.assertTrue(record['child_reaped'])
            self.assertEqual(record['limit_reason'],'timeout')


if __name__ == '__main__': unittest.main()
