"""Bounded stdlib fixtures for the historical quality32 analysis contract."""
import builtins
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('first_risk',ROOT/'scripts/analyze_natural_first_risk.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)


def write(path,value):path.write_text(json.dumps(value)+'\n')
def write_lines(path,rows):path.write_text(''.join(json.dumps(row)+'\n' for row in rows))


def fixture(root, step=128):
    directory=root/'raw'/str(step)/'collection';directory.mkdir(parents=True)
    source=root/'source';source.mkdir(exist_ok=True);(source/'collector.py').write_text('# frozen fixture\n')
    cases=[dict(index=i,id=f'private-id-{i}',seed=i,prompt_token_ids=[i+100],split='validation',record_sha256=f'record-{i}',prompt_sha256=a.digest([i+100])) for i in range(32)]
    protocol=dict(max_new_tokens=128,block_size=7,temperature=1.,probability_policy='float64_softmax_normalize_cdf_v1',collection_policy='full_proposal')
    manifest=dict(protocol=protocol,groups={'quality':[{k:v for k,v in c.items() if k in ('id','seed','record_sha256','prompt_sha256')} for c in cases]})
    manifest['manifest_sha256']=a.digest(manifest);write(directory.parent/'panel.private.json',manifest)
    binding=dict(cases=cases,manifest=manifest,source_sha256={'collector.py':a.sha(source/'collector.py')},checkpoint=f'step-{step:06d}',manifest_file_sha256=a.sha(directory.parent/'panel.private.json'))
    binding['binding_sha256']=a.digest(binding);write(directory/'run.json',binding)
    rows=[];blocks=[];outputs=[];runs=[];totals={k:0 for k in a.COUNTERS};counts=[0]*7;events=[0]*7
    for c in cases:
        i=c['index'];tokens=[1];own={k:0 for k in a.COUNTERS};number=0
        while i and len(tokens)<128:
            progress=len(tokens);n=min(7,128-progress);accepted=n if i%2 else 0
            # Frequent rejection on even prompts; full acceptance on odd prompts.
            rejected=0 if accepted==0 else None;extra='residual' if not accepted else 'bonus' if progress+n<128 else None
            committed=[2]*accepted+([3] if extra else []);stop='budget' if progress+len(committed)==128 else 'round_complete'
            e=a.prefix_evidence(n,n,accepted,rejected,None)
            row=dict(case=i,round=number,proposal_tokens=[2]*n,committed_tokens=committed,selected_q=[.5]*n,selected_p=[.1 if i%2 else .4]*n,
                accepted=accepted,rejected_index=rejected,accepted_eos_position=None,verified_proposal_length=n,confidence_logits=[0.]*n,
                finite_and_shape_checked=True,q_dtype='torch.float64',cache_before=progress,cache_after=progress+len(committed),extra_token_kind=extra,stop_reason=stop,**e)
            rows.append(row);blocks.append(dict(block_id=f'{c["id"]}:{number}',prompt_id=c['id'],split='validation',sampling_mode='stochastic',collection_policy='full_proposal',
                proposal_length=n,verified_length=n,accepted_prefix_length=accepted,accepted_eos_position=None,confidence_logits=[0.]*n))
            for k in a.COUNTERS:own[k]+=accepted if k=='accepted_draft_tokens' else e[k]
            for j,label in enumerate(e['prefix_labels']):counts[j]+=1;events[j]+=label
            tokens.extend(committed);number+=1
        runs.append(dict(case=i,seed=c['seed'],output_tokens=len(tokens),ended_eos=i==0,rounds=number,execution_invariants_passed=True,**own))
        outputs.append(dict(id=c['id'],tokens=tokens))
        for k in own:totals[k]+=own[k]
    positions=[dict(position=j,count=counts[j],accepted_prefix_events=events[j]) for j in range(7)]
    result=dict(status='completed',binding_sha256=binding['binding_sha256'],execution_checks_passed=True,runs=runs,blocks=len(rows),per_position=positions,**totals)
    aggregate={k:v for k,v in result.items() if k!='runs'};aggregate.update(completed_prompts=32,output_tokens=sum(r['output_tokens'] for r in runs),eos_prompts=1)
    for name,value in [('result.json',result),('aggregate.json',aggregate),('runtime.json',dict(torch='fixture',transformers='fixture',hip=None,device='cpu',free_bytes_before_load=step)),('worker-exit.json',{'returncode':0})]:write(directory/name,value)
    for name,value in [('private-rounds.jsonl',rows),('private-blocks.jsonl',blocks),('private-outputs.jsonl',outputs)]:write_lines(directory/name,value)
    return directory,source


class FirstRiskTests(unittest.TestCase):
    def test_progress_boundaries_pooled_equal_prompt_initial_eos_and_privacy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);directory,source=fixture(root)
            summary,rows,_=a.check_collection(directory,128,source)
            self.assertEqual(summary['completed_prompts'],32);self.assertEqual(summary['zero_round_prompts'],1)
            self.assertEqual(summary['prompts'][0]['mean_alpha'],None)
            self.assertEqual(summary['equal_prompt']['mean_alpha']['available_prompts'],31)
            self.assertNotAlmostEqual(summary['pooled']['mean_alpha'],summary['equal_prompt']['mean_alpha']['value'])
            self.assertEqual([(r['emitted_before'],r['progress_bin']) for r in rows if r['ordinal']==2 and r['emitted_before'] in (1,31,32,63,64,95,96,127)],
                [(1,'0-31'),(31,'0-31'),(32,'32-63'),(63,'32-63'),(64,'64-95'),(95,'64-95'),(96,'96-127'),(127,'96-127')])
            self.assertEqual(summary['pooled']['first_prefix_events'],sum(p['first_prefix_events'] for p in summary['prompts']))
            encoded=json.dumps(summary)
            for forbidden in ('private-id-', 'prompt_token_ids', 'selected_p', 'selected_q', 'seed', 'committed_tokens'):
                self.assertNotIn(forbidden,encoded)

    def test_missing_or_invalid_risk_retains_labels_and_finite_overflow_alpha(self):
        for row,reason in [({},'missing_selected_p'),({'selected_p':[.1],'selected_q':[0.]},'nonpositive_selected_q'),
                           ({'selected_p':[-.1],'selected_q':[.2]},'negative_selected_p'),
                           ({'selected_p':[float('inf')],'selected_q':[.2]},'nonfinite_or_nonnumeric_selected_p')]:
            value=a.first_risk(row);self.assertFalse(value['valid']);self.assertIn(reason,value['missing_or_invalid_reasons']);json.dumps(value,allow_nan=False)
        overflow=a.first_risk({'selected_p':[1.],'selected_q':[5e-324]})
        self.assertTrue(overflow['valid']);self.assertEqual(overflow['alpha'],1.);self.assertTrue(overflow['ratio_float_overflow']);json.dumps(overflow,allow_nan=False)
        with tempfile.TemporaryDirectory() as tmp:
            directory,source=fixture(Path(tmp));rows=a.lines(directory/'private-rounds.jsonl');rows[0].pop('selected_p');write_lines(directory/'private-rounds.jsonl',rows)
            summary,private,_=a.check_collection(directory,128,source)
            self.assertEqual(summary['pooled']['missing_or_invalid_risk_count'],1)
            self.assertEqual(summary['pooled']['label_count'],len(rows));self.assertEqual(summary['pooled']['valid_risk_count'],len(rows)-1)
            self.assertIsNone(private[0]['alpha'])

    def test_cache_output_label_and_aggregate_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory,source=fixture(Path(tmp));original=a.lines(directory/'private-rounds.jsonl')
            for field,value in [('cache_before',99),('prefix_labels',[0]*7),('committed_tokens',[99])]:
                rows=copy.deepcopy(original);rows[0][field]=value;write_lines(directory/'private-rounds.jsonl',rows)
                with self.assertRaises(ValueError):a.check_collection(directory,128,source)
            write_lines(directory/'private-rounds.jsonl',original)
            result=a.read(directory/'aggregate.json');result['accepted_draft_tokens']+=1;write(directory/'aggregate.json',result)
            with self.assertRaisesRegex(ValueError,'Overall aggregate'):a.check_collection(directory,128,source)

    def test_stdlib_three_steps_identity_gap_and_all32_missing_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for step in a.STEPS:_,source=fixture(root,step)
            original=builtins.__import__
            def guarded(name,*args,**kwargs):
                if name.split('.')[0] in ('torch','numpy','transformers','dspark_qwen'):raise AssertionError('Non-stdlib import')
                return original(name,*args,**kwargs)
            with patch('builtins.__import__',side_effect=guarded):result=a.analyze(root/'raw',source,root/'analysis')
            self.assertEqual(result['status'],'completed');self.assertEqual(len(result['comparisons']),2)
            self.assertTrue(all(len(c['prompts'])==32 for c in result['checkpoints']))
            (root/'raw/512/collection/private-rounds.jsonl').unlink()
            result=a.analyze(root/'raw',source,root/'gap')
            self.assertEqual(result['status'],'evidence_gap');self.assertEqual(result['checkpoints'][1]['status'],'evidence_gap')
            self.assertEqual([r['ordinal'] for r in result['checkpoints'][1]['prompts']],list(range(32)))
            self.assertIn('private-rounds.jsonl',result['checkpoints'][1]['reason'])

    def test_source_binding_and_duplicate_json_key_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory,source=fixture(Path(tmp));(source/'collector.py').write_text('changed')
            with self.assertRaisesRegex(ValueError,'source inventory'):a.check_collection(directory,128,source)
        with self.assertRaisesRegex(ValueError,'Duplicate JSON'):a.loads('{"x":1,"x":2}')
        with self.assertRaisesRegex(ValueError,'Nonfinite JSON'):a.loads('{"x":NaN}')


if __name__=='__main__':unittest.main()
