"""Completed-group CPU fixtures; no model load or device access."""
import builtins
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_rollout_collection import fixture
from dspark_qwen import collect_rollout as collector, calibrate_rollout as cli
from dspark_qwen.rollout_protocol import digest, prefix_evidence


def write(path, value):
    path.write_text(json.dumps(value)+'\n')


def selection_file(root, checkpoint, panel):
    manifest=cli.read(panel);metadata=cli.read(checkpoint/'metadata.json')
    selection=dict(version=1,kind='frozen_sts_checkpoint_selection',checkpoint_step=metadata['step'],
        checkpoint_sha256=cli.sha256(checkpoint/'draft.safetensors'),
        checkpoint_metadata_sha256=cli.sha256(checkpoint/'metadata.json'),
        manifest_sha256=manifest['manifest_sha256'],manifest_file_sha256=cli.sha256(panel),
        rollout_protocol_sha256=digest(manifest['protocol']),
        development_records_sha256=manifest['identity']['development_records_sha256'],groups={'fit':44,'eval':43})
    selection['selection_sha256']=digest(selection)
    path=root/f'selection-{checkpoint.name}.json';write(path,selection)
    return path


def make_collection(root, checkpoint, panel, group, *, name=None, pattern='mixed'):
    directory = root/(name or group)
    collector.main(['--checkpoint',str(checkpoint),'--manifest',str(panel),
                    '--output',str(directory),'--group',group,
                    '--selection',str(selection_file(root,checkpoint,panel)),'--dry-run'])
    binding = cli.read(directory/'run.json')
    report = collector.new_report(binding)
    report.update(status='completed',execution_checks_passed=True)
    rounds, blocks, outputs = [], [], []
    for index, case in enumerate(binding['cases']):
        # Include a genuine no-proposal prompt with an initial EOS draw.
        if index == 0 or pattern == 'zero':
            report['runs'].append(dict(case=index,seed=case['seed'],output_tokens=1,ended_eos=True,
                rounds=0,execution_invariants_passed=True,**{k:0 for k in cli.COUNTERS}))
            outputs.append(dict(id=case['id'],tokens=[9]))
            continue
        accepted = 1 if pattern == 'short' else index % 4
        accepted_eos = accepted-1 if accepted else None
        rejected = 0 if accepted == 0 else None
        # EOS before proposal end truncates accepted tails. Residual EOS after
        # rejection leaves all three verified prefix labels (zeros) observed.
        evidence = prefix_evidence(3,3,accepted,rejected,accepted_eos)
        committed = [9]*max(1,accepted)
        logits = [1.3,-.4,.8]
        block = dict(block_id=f"{case['id']}:0",prompt_id=case['id'],split='validation',
            confidence_logits=logits,proposal_length=3,verified_length=3,
            accepted_prefix_length=accepted,accepted_eos_position=accepted_eos,
            sampling_mode='stochastic',collection_policy='full_proposal')
        row = dict(case=index,round=0,accepted=accepted,rejected_index=rejected,
            accepted_eos_position=accepted_eos,confidence_logits=logits,
            finite_and_shape_checked=True,q_dtype='torch.float64',
            cache_before=len(case['prompt_token_ids']),cache_after=len(case['prompt_token_ids'])+len(committed),
            committed_tokens=committed,extra_token_kind='residual' if accepted==0 else None,
            stop_reason='eos',**evidence)
        rounds.append(row);blocks.append(block);outputs.append(dict(id=case['id'],tokens=[2]+committed))
        report['blocks']+=1
        for key in cli.COUNTERS:
            report[key]+=accepted if key=='accepted_draft_tokens' else evidence[key]
        for j,label in enumerate(evidence['prefix_labels']):
            while len(report['per_position'])<=j:report['per_position'].append(dict(position=len(report['per_position']),count=0,accepted_prefix_events=0))
            report['per_position'][j]['count']+=1;report['per_position'][j]['accepted_prefix_events']+=label
        report['runs'].append(dict(case=index,seed=case['seed'],output_tokens=len(committed)+1,
            ended_eos=True,rounds=1,execution_invariants_passed=True,
            **{key:accepted if key=='accepted_draft_tokens' else evidence[key] for key in cli.COUNTERS}))
    collector.save(directory,report)
    write(directory/'worker-exit.json',{'returncode':0})
    write(directory/'runtime.json',dict(torch='synthetic',transformers='synthetic',hip=None,device='CPU fixture'))
    for name, rows in [('private-blocks.jsonl',blocks),('private-rounds.jsonl',rounds),('private-outputs.jsonl',outputs)]:
        (directory/name).write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return directory


class CalibrationWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        _,_,_,self.panel,self.checkpoint=fixture(self.root)
        metadata=cli.read(self.checkpoint/'metadata.json');metadata['step']=1280
        write(self.checkpoint/'metadata.json',metadata)

    def pair(self, **kwargs):
        fit=make_collection(self.root,self.checkpoint,self.panel,'fit',**kwargs)
        evaluation=make_collection(self.root,self.checkpoint,self.panel,'eval')
        return fit,evaluation

    def test_group_binding_quality_compatibility_and_stdlib_dryrun(self):
        original=builtins.__import__
        def guard(name,*args,**kwargs):
            if name.split('.')[0] in ('torch','transformers','numpy'):raise AssertionError('Device/runtime import')
            return original(name,*args,**kwargs)
        with patch('builtins.__import__',side_effect=guard), patch.object(collector,'launch_worker') as launch:
            for group,expected in [('quality',32),('fit',44),('eval',43)]:
                directory=self.root/group
                selected=[] if group=='quality' else ['--selection',str(selection_file(self.root,self.checkpoint,self.panel))]
                self.assertEqual(collector.main(['--checkpoint',str(self.checkpoint),'--manifest',str(self.panel),'--output',str(directory),'--group',group,'--dry-run',*selected]),0)
                binding=cli.read(directory/'run.json');self.assertEqual(len(binding['cases']),expected)
                self.assertEqual(binding.get('collection_group','quality'),group)
                collector.verify_binding(binding,directory/'source/dspark_qwen')
            launch.assert_not_called()
        quality=collector.bind_inputs(self.checkpoint,self.panel)
        self.assertNotIn('collection_group',quality)
        self.assertEqual(quality,collector.bind_inputs(self.checkpoint,self.panel,group='quality'))
        with self.assertRaises(ValueError):collector.bind_inputs(self.checkpoint,self.panel,group='test')

    def test_explicit_selection_required_and_checkpoint_panel_tamper_rejected(self):
        with self.assertRaisesRegex(ValueError,'require --selection'):
            collector.bind_inputs(self.checkpoint,self.panel,group='fit')
        path=selection_file(self.root,self.checkpoint,self.panel)
        original=cli.read(path)
        for key,value in [('checkpoint_sha256','f'*64),('manifest_sha256','e'*64),('checkpoint_step',512)]:
            modified=copy.deepcopy(original);modified[key]=value
            modified['selection_sha256']=digest({k:v for k,v in modified.items() if k!='selection_sha256'})
            write(path,modified)
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'selection differs'):
                collector.bind_inputs(self.checkpoint,self.panel,group='fit',selection_path=path)

    def test_fit_eval_cli_complete_and_frozen_fit_constants(self):
        fit,evaluation=self.pair()
        original=builtins.__import__
        def guard(name,*args,**kwargs):
            if name.split('.')[0] in ('torch','transformers','numpy'):raise AssertionError('Runtime import')
            return original(name,*args,**kwargs)
        with patch('builtins.__import__',side_effect=guard):
            artifact=cli.fit_collection(fit,temperature_grid=[1,2])
            report=cli.evaluate_collection(artifact,fit,evaluation)
        self.assertEqual(artifact['fit_evidence']['completed_prompts'],44)
        self.assertEqual(report['eval_evidence']['completed_prompts'],43)
        self.assertEqual(report['eval_evidence']['zero_block_prompts'],1)
        self.assertEqual(report['num_bins'],20)
        self.assertNotEqual(artifact['fit_prefix_prevalence'][0],report['per_position'][0]['unscaled']['target_mean'])
        for j,row in enumerate(report['per_position']):
            self.assertEqual(row['fit_prevalence_constant']['probability'],artifact['fit_prefix_prevalence'][j])
            self.assertEqual(row['unscaled']['count'],row['sts']['count'])
            self.assertEqual(row['unscaled']['count'],row['fit_prevalence_constant']['count'])
            self.assertLess(row['prompt_coverage'],1.)
        output=self.root/'artifact.json'
        self.assertEqual(cli.main(['fit','--collection',str(fit),'--temperature-grid','1','2','--output',str(output)]),0)
        self.assertEqual(cli.main(['eval','--artifact',str(output),'--fit-collection',str(fit),'--collection',str(evaluation),'--output',str(self.root/'eval.json')]),0)
        with self.assertRaises(FileExistsError):cli.main(['fit','--collection',str(fit),'--output',str(output)])

    def test_missing_fit_tail_never_uses_eval_prevalence(self):
        fit,evaluation=self.pair(pattern='short')
        artifact=cli.fit_collection(fit,temperature_grid=[1])
        self.assertEqual(artifact['fit_prefix_prevalence'],[1.,None,None])
        report=cli.evaluate_collection(artifact,fit,evaluation)
        for row in report['per_position'][1:]:
            self.assertGreater(row['observed_label_count'],0)
            self.assertFalse(row['fit_prevalence_constant']['available'])
            self.assertIsNone(row['fit_prevalence_constant']['brier'])
            self.assertEqual(row['fit_prevalence_constant']['count'],0)
            self.assertFalse(row['sts']['fitted_on_fit'])

    def test_incomplete_exit_status_and_group_rejected(self):
        fit,evaluation=self.pair()
        with self.assertRaises(ValueError):cli.load_collection(fit,'eval')
        for filename, mutate in [('worker-exit.json',lambda x:x.update(returncode=1)),
                                 ('result.json',lambda x:x.update(status='running')),
                                 ('result.json',lambda x:x['runs'].pop()),
                                 ('aggregate.json',lambda x:x.update(completed_prompts=43))]:
            original=(fit/filename).read_text();value=json.loads(original);mutate(value);write(fit/filename,value)
            with self.subTest(filename=filename),self.assertRaises(ValueError):cli.load_collection(fit,'fit')
            (fit/filename).write_text(original)
        (fit/'worker-exit.json').unlink()
        with self.assertRaises(FileNotFoundError):cli.load_collection(fit,'fit')

    def test_raw_confidence_labels_counters_and_prompt_leakage_rejected(self):
        fit,_=self.pair()
        for filename,key,value in [('private-blocks.jsonl','split','test'),('private-blocks.jsonl','prompt_id','dev0'),
                ('private-blocks.jsonl','confidence_logits',[0.,0.,0.]),('private-rounds.jsonl','prefix_labels',[0,0,0]),
                ('private-rounds.jsonl','round',1),('private-rounds.jsonl','cache_after',999)]:
            original=(fit/filename).read_text();rows=[json.loads(x) for x in original.splitlines()];rows[0][key]=value
            (fit/filename).write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.subTest(key=key),self.assertRaises(ValueError):cli.load_collection(fit,'fit')
            (fit/filename).write_text(original)

    def test_artifact_changed_fit_changed_checkpoint_changed_and_source_changed(self):
        fit,evaluation=self.pair();artifact=cli.fit_collection(fit,temperature_grid=[1])
        bad=copy.deepcopy(artifact);bad['fit']['temperatures'][0]=2
        with self.assertRaisesRegex(ValueError,'artifact changed'):cli.evaluate_collection(bad,fit,evaluation)
        bad=copy.deepcopy(artifact);bad['implementation_sha256']['calibration.py']='f'*64
        bad['artifact_sha256']=digest({k:v for k,v in bad.items() if k!='artifact_sha256'})
        with self.assertRaisesRegex(ValueError,'implementation changed'):cli.evaluate_collection(bad,fit,evaluation)
        # Whitespace is semantically harmless but must invalidate frozen evidence.
        original=(fit/'result.json').read_text();(fit/'result.json').write_text(original+'\n')
        with self.assertRaisesRegex(ValueError,'Fit collection changed'):cli.evaluate_collection(artifact,fit,evaluation)
        (fit/'result.json').write_text(original)
        (evaluation/'source/dspark_qwen/collect_rollout.py').write_text('tampered')
        with self.assertRaisesRegex(ValueError,'Source content changed'):cli.evaluate_collection(artifact,fit,evaluation)
        (self.checkpoint/'draft.safetensors').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'weight hash mismatch'):cli.load_collection(fit,'fit')

    def test_same_group_relabel_cannot_create_disjoint_evaluation(self):
        fit,evaluation=self.pair()
        binding=cli.read(evaluation/'run.json');old=cli.read(fit/'run.json')
        binding['cases']=old['cases'];binding['binding_sha256']=digest({k:v for k,v in binding.items() if k!='binding_sha256'})
        write(evaluation/'run.json',binding)
        with self.assertRaisesRegex(ValueError,'Bound inputs changed'):cli.load_collection(evaluation,'eval')

    def test_valid_other_checkpoint_or_runtime_cannot_use_frozen_fit(self):
        fit,evaluation=self.pair();artifact=cli.fit_collection(fit,temperature_grid=[1])
        runtime=cli.read(evaluation/'runtime.json');runtime['torch']='different runtime'
        write(evaluation/'runtime.json',runtime)
        with self.assertRaisesRegex(ValueError,'identity differs'):cli.evaluate_collection(artifact,fit,evaluation)
        other=self.root/'other-checkpoint';other.mkdir()
        (other/'draft.safetensors').write_bytes(b'another valid checkpoint')
        metadata=cli.read(self.checkpoint/'metadata.json')
        metadata['draft_weights_sha256']=cli.sha256(other/'draft.safetensors')
        write(other/'metadata.json',metadata)
        changed=make_collection(self.root,other,self.panel,'eval',name='other-eval')
        with self.assertRaisesRegex(ValueError,'identity differs'):cli.evaluate_collection(artifact,fit,changed)

    def test_zero_block_prompts_all_complete_but_no_fit_fabrication(self):
        fit=make_collection(self.root,self.checkpoint,self.panel,'fit',pattern='zero')
        checked=cli.load_collection(fit,'fit');self.assertEqual(checked['evidence']['zero_block_prompts'],44)
        with self.assertRaisesRegex(ValueError,'No observable'):cli.fit_collection(fit)

    def test_strict_json_rejects_nan_and_duplicate_keys(self):
        p=self.root/'bad.json'
        for text in ['{"x":NaN}','{"x":1,"x":2}']:
            p.write_text(text)
            with self.assertRaises(ValueError):cli.read(p)


if __name__=='__main__':unittest.main()
