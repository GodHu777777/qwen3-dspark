"""Frozen grouping, denominator semantics, dry-run isolation and tiny-Qwen collection."""
import builtins
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from dspark_qwen import rollout_protocol as protocol, collect_rollout as collector


def fixture(root):
    model, checkpoint = root / 'model', root / 'checkpoint'
    model.mkdir(); checkpoint.mkdir()
    (model / 'config.json').write_text(json.dumps(dict(model_type='qwen3', vocab_size=512)))
    (model / 'model.safetensors').write_bytes(b'fixture')
    (checkpoint / 'draft.safetensors').write_bytes(b'fixture checkpoint')
    rows = [dict(id=f'dev{i}', split='validation', accepted=True, prompt_token_ids=[1, i+2],
                 output_token_ids=[3], messages=[dict(role='user', content=f'question {i}')]) for i in range(119)]
    records = root / 'records.jsonl'
    records.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    generation = root / 'manifest.json'
    fingerprint = {p.name: protocol.sha256(p) for p in model.iterdir()}
    generation.write_text(json.dumps(dict(model_files_sha256=fingerprint)))
    (root / 'summary.json').write_text(json.dumps(dict(output_sha256={'records.jsonl': protocol.sha256(records)})))
    cfg = dict(model=str(model), records=str(records), generation_manifest=str(generation), draft=dict(block_size=3))
    (checkpoint / 'metadata.json').write_text(json.dumps(dict(draft_config=cfg['draft'],
        draft_weights_sha256=protocol.sha256(checkpoint / 'draft.safetensors'), identity=dict(config=cfg,
        records_sha256=protocol.sha256(records), target_fingerprint=fingerprint))))
    manifest = protocol.build_manifest(cfg)
    path = root / 'panel.json'
    protocol.write_json(path, manifest)
    return cfg, rows, manifest, path, checkpoint


class ProtocolTests(unittest.TestCase):
    def test_exact_partition_stable_seeds_and_checkpoint_independence(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, rows, manifest, path, checkpoint = fixture(Path(directory))
            self.assertEqual({k: len(v) for k,v in manifest['groups'].items()}, dict(quality=32,fit=44,eval=43))
            self.assertEqual([r['id'] for r in manifest['groups']['quality']], [r['id'] for r in rows[:32]])
            ids = [r['id'] for group in manifest['groups'].values() for r in group]
            self.assertEqual(len(set(ids)), 119)
            self.assertEqual(manifest, protocol.build_manifest(cfg))
            binding = collector.bind_inputs(checkpoint, path)
            self.assertEqual(binding['cases'][0]['seed'], protocol.prompt_seed('dev0'))
            self.assertEqual(manifest['tv_probes'], [dict(id='dev0',round=0),dict(id='dev1',round=0)])
            self.assertEqual(manifest['protocol']['max_new_tokens'], 128)

    def test_rejects_leakage_changes_and_missing_rows(self):
        mutations = [lambda r: r.pop(), lambda r: r[1].update(id=r[0]['id']),
            lambda r: r[0].update(split='test'), lambda r: r[1].update(prompt_token_ids=r[0]['prompt_token_ids']),
            lambda r: r[1].update(messages=r[0]['messages']),
            lambda r: r[1].update(messages=[dict(role='user', content='ｑｕｅｓｔｉｏｎ ０')])]
        for mutate in mutations:
            with self.subTest(mutate=mutate), tempfile.TemporaryDirectory() as directory:
                cfg, rows, manifest, path, checkpoint = fixture(Path(directory))
                mutate(rows)
                Path(cfg['records']).write_text(''.join(json.dumps(r)+'\n' for r in rows))
                with self.assertRaises(ValueError): protocol.read_rows(cfg['records'])
                with self.assertRaises(ValueError): collector.bind_inputs(checkpoint, path)
        with tempfile.TemporaryDirectory() as directory:
            cfg, rows, manifest, path, checkpoint = fixture(Path(directory))
            manifest['groups']['quality'][0]['seed'] += 1
            with self.assertRaises(ValueError): protocol.verify_manifest(manifest, cfg)

    def test_prefix_labels_use_verified_tail_and_accepted_eos_truncation(self):
        row = protocol.prefix_evidence(7,7,2,2,None)
        self.assertEqual(row['prefix_labels'], [1,1,0,0,0,0,0])
        self.assertEqual(row['attempted_positions'], 3)
        self.assertEqual(row['effective_positions'], 7)
        self.assertEqual(protocol.prefix_evidence(7,7,2,None,1)['prefix_labels'], [1,1])
        self.assertEqual(protocol.prefix_evidence(2,2,2,None,None)['prefix_labels'], [1,1])
        self.assertEqual(protocol.prefix_evidence(0,0,0,None,None)['prefix_labels'], [])
        with self.assertRaises(ValueError): protocol.prefix_evidence(7,7,2,2,1)
        with self.assertRaises(ValueError): protocol.prefix_evidence(7,7,2,None,None)
        # Residual/bonus EOS has no accepted_eos_position; its verified tail remains.
        self.assertEqual(protocol.prefix_evidence(7,7,0,0,None)['effective_positions'],7)
        self.assertEqual(protocol.prefix_evidence(7,7,7,None,None)['effective_positions'],7)

    def test_dry_run_is_stdlib_and_snapshot_binding_detects_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg, rows, manifest, path, checkpoint = fixture(root)
            real_import = builtins.__import__
            def guard(name, *args, **kwargs):
                if name.split('.')[0] in ('torch','transformers'):
                    raise AssertionError('Runtime imported in dry run')
                return real_import(name,*args,**kwargs)
            out = root / 'dry'
            with patch('builtins.__import__',side_effect=guard), patch.object(collector,'launch_worker') as launch:
                self.assertEqual(collector.main(['--checkpoint',str(checkpoint),'--manifest',str(path),'--output',str(out),'--dry-run']),0)
                launch.assert_not_called()
            binding = json.loads((out / 'run.json').read_text())
            source = out / 'source' / 'dspark_qwen'
            collector.verify_binding(binding, source)
            (source / 'collect_rollout.py').write_text('changed')
            with self.assertRaises(ValueError): collector.verify_binding(binding, source)
            self.assertEqual(out.stat().st_mode & 0o777,0o700)

    def test_timeout_keeps_partial_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg, rows, manifest, path, checkpoint = fixture(root)
            def timeout(out, seconds):
                report = json.loads((out / 'result.json').read_text())
                report['blocks'] = 4
                collector.save(out, report)
                raise subprocess.TimeoutExpired('fixture', seconds)
            with patch.object(collector,'launch_worker',side_effect=timeout):
                self.assertEqual(collector.main(['--checkpoint',str(checkpoint),'--manifest',str(path),'--output',str(root/'run')]),1)
            saved = json.loads((root/'run'/'result.json').read_text())
            self.assertEqual(saved['blocks'],4)
            self.assertEqual(saved['error_type'],'TimeoutExpired')


@unittest.skipUnless(importlib.util.find_spec('torch'), 'CPU Torch required')
class CollectionTests(unittest.TestCase):
    def setUp(self):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from dspark_qwen.config import DraftConfig
        from dspark_qwen.model import DSparkDraft
        torch.manual_seed(212)
        config = Qwen3Config(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=4,
            num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=128,eos_token_id=None)
        config._attn_implementation='sdpa'
        self.model=Qwen3ForCausalLM(config).eval()
        self.draft=DSparkDraft(self.model,DraftConfig(layer_ids=(0,2),num_layers=2,block_size=3,
                                                   markov_rank=8,mask_token_id=31)).eval()
        self.binding=dict(manifest=dict(protocol=dict(probability_policy=protocol.POLICY,temperature=1.,
            max_new_tokens=6,block_size=3),tv_probes=[dict(id='a',round=0)]),cases=[dict(index=0,id='a',
                seed=19,prompt_token_ids=[1,2,3])])

    def run_fixture(self, out):
        import torch
        report=collector.new_report()
        with torch.no_grad():
            collector.exercise(self.model,self.draft,self.binding,out,report,device='cpu',amp=False)
        return report

    def test_actual_q_bounded_probes_seed_repeat_budget_and_labels(self):
        import torch
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            first=self.run_fixture(Path(a)); second=self.run_fixture(Path(b))
            self.assertEqual(first,second)
            self.assertEqual(first['runs'][0]['output_tokens'],6)
            self.assertEqual(first['verified_positions'],first['proposed_positions'])
            self.assertEqual(first['effective_positions'],first['verified_positions'])
            self.assertEqual(sum(p['count'] for p in first['per_position']),first['effective_positions'])
            self.assertEqual(len(first['numerical_probes']),4)
            payload=torch.load(next(Path(a).glob('private-probe-*.pt')),weights_only=True)
            self.assertEqual(payload['actual_q'].dtype,torch.float64)
            self.assertEqual(len(payload['sequential_probs']),4)
            rows=[json.loads(r) for r in (Path(a)/'private-rounds.jsonl').read_text().splitlines()]
            self.assertTrue(all(len(r['prefix_labels'])==r['verified_positions'] for r in rows))
            self.assertLessEqual(rows[-1]['proposed_positions'],3)

    def test_budget_one_has_no_fabricated_proposal_tail(self):
        self.binding['manifest']['protocol']['max_new_tokens']=1
        with tempfile.TemporaryDirectory() as directory:
            report=self.run_fixture(Path(directory))
            self.assertEqual(report['blocks'],0)
            self.assertFalse(report['runs'][0]['ended_eos'])
            self.assertEqual(report['runs'][0]['output_tokens'],1)

    def test_first_token_eos_has_no_fake_blocks_or_replacement_probe(self):
        self.model.generation_config.eos_token_id=list(range(32))
        with tempfile.TemporaryDirectory() as directory:
            report=self.run_fixture(Path(directory))
            self.assertEqual(report['blocks'],0)
            self.assertEqual(report['effective_positions'],0)
            self.assertEqual(report['runs'][0]['output_tokens'],1)
            self.assertEqual(report['runs'][0]['preselected_probes'],[dict(round=0,status='unreached')])
            self.assertEqual(report['numerical_probes'],[])

    def test_calibration_group_does_not_replace_quality_probe_ids(self):
        self.binding['collection_group']='fit'
        self.binding['manifest']['tv_probes']=[dict(id='quality-only',round=0)]
        with tempfile.TemporaryDirectory() as directory:
            report=self.run_fixture(Path(directory))
            self.assertEqual(report['runs'][0]['output_tokens'],6)
            self.assertEqual(report['runs'][0]['preselected_probes'],[])
            self.assertEqual(report['numerical_probes'],[])
            self.assertFalse(list(Path(directory).glob('private-probe-*')))

    def test_probe_failure_preserves_rounds_outputs_and_partial_tensor(self):
        original=collector.compare_probe
        def fail(model,ids,output,emitted,payload,temperature,on_row):
            def stop(metric):
                on_row(metric)
                raise RuntimeError('injected probe failure')
            return original(model,ids,output,emitted,payload,temperature,stop)
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            with patch.object(collector,'compare_probe',side_effect=fail):
                with self.assertRaisesRegex(RuntimeError,'injected probe failure'): self.run_fixture(out)
            report=json.loads((out/'result.json').read_text())
            self.assertEqual(len(report['runs']),1)
            self.assertEqual(len(report['numerical_probes']),1)
            self.assertTrue((out/'private-blocks.jsonl').exists())
            import torch
            payload=torch.load(next(out.glob('private-probe-*.pt')),weights_only=True)
            self.assertEqual(len(payload['sequential_probs']),1)


if __name__=='__main__': unittest.main()
