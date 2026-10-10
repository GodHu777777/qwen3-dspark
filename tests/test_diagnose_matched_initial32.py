"""CPU-only synthetic protocol tests: no real model, tokenizer, checkpoint or GPU."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/diagnose_matched_initial32.py'
spec=importlib.util.spec_from_file_location('initial32',SCRIPT)
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)


def raw_fixture():
    p=torch.tensor([.125,.375,.5],dtype=torch.float64)
    q=torch.tensor([.25,.25,.5],dtype=torch.float64)
    seq=torch.tensor([.25,.5,.25],dtype=torch.float64)
    return dict(q=q,block=p,seq=seq,q_logits=torch.zeros(3,dtype=torch.bfloat16),
        block_logits=torch.tensor([0.,1.,2.],dtype=torch.bfloat16),seq_logits=torch.tensor([.25,1.,1.5],dtype=torch.bfloat16),
        confidence0=.25,block_laws=dict(passed=True),cache_lengths=dict(prefill=32,draft_context=32,block_target=40,sequential_prefill=32,sequential_target=33),shapes=dict(prompt=[1,32],backbone=[7,4],base=[7,3],q_adapter=[1,3],block_adapter=[8,3],seq_adapter=[3]))


def panel_fixture(out):
    raw=raw_fixture();states=[]
    for step,ordinal in a.PLAN:
        state=dict(step=step,ordinal=ordinal,prompt=[0]*32,initial_token=2,proposal_tokens=[0,1,2,0,1,2,0],
            selected_q0=.25,selected_p0=.125,confidence0=.25,anchor=None)
        if ordinal<2:
            payload=dict(actual_q=raw['q'].repeat(7,1),block_probs=raw['block'].repeat(8,1),sequential_probs=[raw['seq'].clone() for _ in range(8)],
                block_logits=raw['block_logits'].repeat(8,1),sequential_logits=[raw['seq_logits'].clone() for _ in range(8)],
                proposal_tokens=torch.tensor(state['proposal_tokens']),semantic_committed_output_prefix=[2])
            path=out/f'anchor-{step}-{ordinal}.pt';torch.save(payload,path)
            state['anchor']=dict(step=step,ordinal=ordinal,path=str(path),sha256=a.sha(path),proposal_tokens=state['proposal_tokens'],initial_output=[2],
                selected_q0=.25,selected_p0=.125,historical_alpha=.5,first_prefix_label=0,numerical_control=dict(total_variation=.25,max_probability_difference=.25,
                max_absolute_logit_difference=.5,mean_absolute_logit_difference=.25,argmax_equal=False))
        states.append(state)
    return dict(states=states,plan=a.PLAN,input_sha256={},source_sha256={})


class CaptureTests(unittest.TestCase):
    def test_original_shapes_amp_no_grad_order_and_fresh_caches(self):
        events=[];targets=[]
        def inference(label,shape):
            self.assertFalse(torch.is_grad_enabled())
            events.append((label,tuple(shape)))
            if label in ('prefill','append','predict'):self.assertFalse(torch.is_autocast_enabled('cpu'))
        class Target:
            def __init__(self,model,layers):self.length=0;self.layers=layers;targets.append(self)
            def prefill(self,ids):
                inference('prefill',ids.shape);self.length=ids.shape[1]
                return SimpleNamespace(context=torch.ones(1,self.length,4))
            def append(self,ids):inference('append',ids.shape);self.length+=ids.shape[1];return SimpleNamespace(n=ids.shape[1])
            def predict(self,features,last_only=False):
                inference('predict',(features.n,last_only))
                return torch.zeros((1,3) if last_only else (1,features.n,3),dtype=torch.bfloat16)
            def reset(self):self.length=0
        class Cache:
            def __init__(self,draft):self.length=0;self.layers=[]
            def append(self,features):
                inference('context',features.shape);self_outer.assertTrue(torch.is_autocast_enabled('cpu'));self.length=features.shape[1]
            def reset(self):self.length=0;self.layers=[]
        self_outer=self
        model=Mock();model.parameters.return_value=iter([torch.tensor(0.)]);model.config.vocab_size=3
        draft=Mock();draft.spec.layer_ids=(1,2)
        def operation(label,shape,result):
            def call(value,*rest):
                inference(label,value.shape);self.assertEqual(tuple(value.shape),shape);self.assertTrue(torch.is_autocast_enabled('cpu'));return result
            return call
        draft.backbone_cached.side_effect=operation('backbone',(1,1),torch.ones(7,4))
        draft.lm_head.side_effect=operation('base',(7,4),torch.zeros(7,3,dtype=torch.bfloat16))
        draft.markov_embedding.side_effect=operation('embedding',(1,),torch.ones(1,2))
        draft.confidence.side_effect=operation('confidence',(1,6),torch.zeros(1,1,dtype=torch.bfloat16))
        draft.markov_projection.side_effect=operation('markov',(1,2),torch.zeros(1,3,dtype=torch.bfloat16))
        def adapter(logits,temperature):
            inference('adapter',logits.shape);self.assertEqual(torch.is_autocast_enabled('cpu'),tuple(logits.shape)==(1,3));self.assertEqual(temperature,1.);return torch.softmax(logits.double(),dim=-1)
        state=dict(prompt=[1]*32,initial_token=2,proposal_tokens=[0]*7)
        raw=a.capture_state(model,draft,state,target_class=Target,cache_class=Cache,adapter=adapter)
        a.validate_new(raw,vocab=3)
        self.assertEqual([shape for name,shape in events if name=='adapter'],[(1,3),(8,3),(3,)])
        self.assertEqual([shape for name,shape in events if name=='prefill'],[(1,32),(1,32)])
        self.assertEqual([shape for name,shape in events if name=='append'],[(1,8),(1,1)])
        self.assertLess([x[0] for x in events].index('confidence'),[x[0] for x in events].index('markov'))
        self.assertEqual(len(targets),2);self.assertEqual([t.layers for t in targets],[(1,2),()]);self.assertTrue(all(t.length==0 for t in targets))
        model.eval.assert_called_once();draft.eval.assert_called_once()
        for value in raw.values():
            if isinstance(value,torch.Tensor):self.assertFalse(value.requires_grad)
        draft.propose_stochastic_cached.assert_not_called();draft._sample_stochastic.assert_not_called()

    def test_invalid_transient_tail_and_retained_law(self):
        for change in (lambda r:r['q'].fill_(float('nan')),lambda r:r.update(block_laws=dict(passed=False)),
                       lambda r:r.update(seq=r['seq'].float()),lambda r:r['shapes'].update(block_adapter=[3])):
            raw=raw_fixture();change(raw)
            with self.assertRaises(ValueError):a.validate_new(raw,vocab=3)
        block=torch.full((8,3),1/3,dtype=torch.float64);block[7,0]+=1e-5
        self.assertFalse(a.law_record(block,(8,3))['passed'])


class PanelTests(unittest.TestCase):
    def test_all64_order_four_loads_and_no_private_public_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);binding=panel_fixture(out);calls=[];loads=[]
            def capture(s,observe):
                calls.append((s['step'],s['ordinal']))
                for event in ('target_prefill','draft_block','target_block_append','target_prefill','target_anchor_append'):observe(event)
                return raw_fixture()
            with patch.object(a,'verify_hashes'):
                result=a.panel(binding,out,loads.append,capture,vocab=3)
            self.assertEqual(calls,a.PLAN);self.assertEqual(loads,[512,1280,512,1280])
            self.assertEqual(result['aggregate']['count'],32);self.assertEqual(result['aggregate']['sign_counts']['zero'],32)
            text=json.dumps(result)
            for secret in ('prompt','initial_token','proposal_tokens','selected_q0','confidence0','seed'):self.assertNotIn('"'+secret+'"',text)
            self.assertEqual(len(list(out.glob('private-step*.pt'))),64)

    def test_mismatch_persists_raw_and_comparison_before_stop(self):
        for kind in ('anchor','gather','confidence'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
                out=Path(tmp);binding=panel_fixture(out);calls=[]
                def capture(s,observe):
                    calls.append((s['step'],s['ordinal']));raw=raw_fixture()
                    if kind=='anchor':raw['seq']=torch.tensor([.2,.5,.3],dtype=torch.float64)
                    elif kind=='gather':raw['q']=torch.tensor([.2,.3,.5],dtype=torch.float64)
                    else:raw['confidence0']+=1e-6
                    return raw
                with self.assertRaises(a.ReplayMismatch):a.panel(binding,out,lambda s:None,capture,vocab=3)
                self.assertEqual(calls,[(512,0)])
                self.assertTrue((out/'private-step512-ordinal0.pt').exists())
                self.assertFalse(a.read(out/'private-check-step512-ordinal0.json')['passed'])
                summary=a.read(out/'summary.json');self.assertEqual(summary['status'],'historical_numerical_mismatch');self.assertFalse(summary['aggregate_available'])

    def test_fourth_anchor_failure_never_starts_remaining30(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);binding=panel_fixture(out);calls=[]
            def capture(s,observe):
                calls.append((s['step'],s['ordinal']));r=raw_fixture()
                if len(calls)==4:r['confidence0']=.5
                return r
            with self.assertRaises(a.ReplayMismatch):a.panel(binding,out,lambda s:None,capture,vocab=3)
            self.assertEqual(calls,a.PLAN[:4]);self.assertEqual(len(a.read(out/'progress.private.json')['completed']),3)

    def test_illegal_law_evidence_gap_after_raw_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);binding=panel_fixture(out);raw=raw_fixture();raw['q'][1]=float('nan')
            with self.assertRaises(ValueError):a.panel(binding,out,lambda s:None,lambda s,observe:raw,vocab=3)
            self.assertTrue((out/'private-step512-ordinal0.pt').exists())
            self.assertEqual(a.read(out/'summary.json')['status'],'evidence_gap')

    def test_missing_duplicate_plan_and_partial32_cannot_aggregate(self):
        for mutation in ('missing','duplicate','plan'):
            with tempfile.TemporaryDirectory() as tmp:
                out=Path(tmp);binding=panel_fixture(out)
                if mutation=='missing':binding['states'].pop()
                elif mutation=='duplicate':binding['states'][-1]=binding['states'][0]
                else:binding['plan']=list(reversed(a.PLAN))
                capture=Mock()
                with self.assertRaises(ValueError):a.panel(binding,out,Mock(),capture,vocab=3)
                capture.assert_not_called()
        states=[a.compare_pair(i,{512:raw_fixture(),1280:raw_fixture()}) for i in range(32)]
        for invalid in (states[:-1],states[:-1]+[states[0]],list(reversed(states))):
            with self.assertRaises(ValueError):a.aggregate(invalid)

    def test_deadline_no_forward_and_partial_saved(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);binding=panel_fixture(out);capture=Mock()
            with self.assertRaises(TimeoutError):a.panel(binding,out,Mock(),capture,vocab=3,deadline=time.monotonic()-1)
            capture.assert_not_called();self.assertEqual(a.read(out/'summary.json')['status'],'partial_deadline')

    def test_final_input_mutation_does_not_publish_complete_aggregate(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);binding=panel_fixture(out)
            def capture(state,observe):
                for event in ('target_prefill','draft_block','target_block_append','target_prefill','target_anchor_append'):observe(event)
                return raw_fixture()
            with patch.object(a,'verify_hashes',side_effect=ValueError('Input changed')):
                with self.assertRaises(ValueError):a.panel(binding,out,lambda s:None,capture,vocab=3)
            self.assertEqual(len(a.read(out/'progress.private.json')['completed']),64)
            self.assertFalse(a.read(out/'summary.json')['aggregate_available'])

    def test_pair_sensitivity_and_equal_state_disagreement(self):
        rows={512:raw_fixture(),1280:raw_fixture()}
        rows[512].update(q=torch.tensor([.8,.1,.1],dtype=torch.float64),seq=torch.tensor([.9,.05,.05],dtype=torch.float64))
        rows[1280].update(q=torch.tensor([.1,.8,.1],dtype=torch.float64),seq=torch.tensor([.05,.9,.05],dtype=torch.float64))
        result=a.compare_pair(0,rows)
        self.assertEqual(result['numerical_sensitivity'],'numerically_sensitive')
        self.assertEqual([t['sign'] for t in result['teachers']],[-1,1])
        self.assertLess(result['delta'],0)
        for item in result['paired_controls'].values():self.assertLessEqual(item['shift'],item['bound'])


class BindingTests(unittest.TestCase):
    def evidence(self,out):
        source=out/'source';source.mkdir()
        for name in ('model.py','cached_target.py','cached_decode.py','tensor_sampling.py'):shutil.copy(a.ROOT/'dspark_qwen'/name,source/name)
        base=dict(input_sha256={},probes=[dict(step=s,ordinal=i) for s in a.STEPS for i in range(2)])
        for step in a.STEPS:
            d=out/str(step)/'collection';d.mkdir(parents=True)
            cases=[dict(index=i,id=f'case{i}',seed=i,split='validation',prompt_token_ids=[i]+[0]*31,prompt_sha256=a.digest([i]+[0]*31)) for i in range(32)]
            group=[{k:c[k] for k in ('id','seed','prompt_sha256')} for c in cases]
            run=dict(cases=cases,checkpoint='unused',checkpoint_sha256=a.CHECKPOINTS[step][0],checkpoint_metadata_sha256=a.CHECKPOINTS[step][1],
                manifest_file_sha256='b5119c7f83c1f6d0ea08b58eb795bad2d1475fb6e4b71ece46ed9ae1fe00ee1e',manifest=dict(groups=dict(quality=group),identity=dict(target_fingerprint={},development_records_sha256='unused')),
                model='/FORBIDDEN/model',draft_config={})
            a.write(d/'run.json',run)
            a.write(d/'result.json',dict(runs=[dict(case=i,seed=i,execution_invariants_passed=True,output_tokens=2) for i in range(32)]))
            records={'private-outputs.jsonl':[dict(id=f'case{i}',tokens=[2,1]) for i in range(32)],
                'private-rounds.jsonl':[dict(case=i,round=0,proposal_tokens=[0]*7,verified_proposal_length=7,cache_before=32,cache_after=33,committed_tokens=[1],finite_and_shape_checked=True,q_dtype='torch.float64',selected_q=[.25]*7,selected_p=[.125]*7,confidence_logits=[.25]*7,accepted=0) for i in range(32)],
                'private-blocks.jsonl':[dict(block_id=f'case{i}:0',prompt_id=f'case{i}',split='validation',sampling_mode='stochastic',collection_policy='full_proposal',proposal_length=7,verified_length=7,confidence_logits=[.25]*7,accepted_prefix_length=0) for i in range(32)]}
            for name,rows in records.items():(d/name).write_text(''.join(json.dumps(r)+'\n' for r in rows))
        return source,base

    def test_identity_extension_rejects_missing_duplicate_order_anchor(self):
        for mutation in ('none','missing','duplicate','reorder','anchor'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as tmp:
                out=Path(tmp);source,base=self.evidence(out);d=out/'1280'/'collection'
                if mutation in ('missing','duplicate'):
                    p=d/'private-rounds.jsonl';rows=a.old.lines(p)
                    if mutation=='missing':rows.pop()
                    else:rows[-1]=rows[0]
                    p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                elif mutation=='reorder':
                    p=d/'run.json';run=a.read(p);run['cases'].reverse();a.write(p,run)
                elif mutation=='anchor':
                    p=d/'private-outputs.jsonl';rows=a.old.lines(p);rows[31]['tokens'][0]=1;p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                with patch.object(a.old,'bind',return_value=base):
                    if mutation=='none':self.assertEqual(len(a.bind_evidence(out,out,source)['states']),64)
                    else:
                        with self.assertRaises(ValueError):a.bind_evidence(out,out,source)

    def test_dependency_hashes_without_dereferencing_training_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);model=out/'model';model.mkdir()
            a.write(model/'config.json',dict(model_type='qwen3',vocab_size=a.old.VOCAB))
            (model/'model.safetensors').write_bytes(b'synthetic-weight-bytes')
            (model/'tokenizer.json').write_text('{}')
            fp={p.name:a.sha(p) for p in model.iterdir()}
            binding=dict(input_sha256={},model=str(model),target_fingerprint=fp,draft_config={},development_records_sha256='bound-records',checkpoints={})
            for step in a.STEPS:
                cp=out/str(step);cp.mkdir();(cp/'draft.safetensors').write_bytes(b'synthetic-draft-bytes')
                m=dict(step=step,draft_weights_sha256=a.sha(cp/'draft.safetensors'),draft_config={},identity=dict(target_fingerprint=fp,records_sha256='bound-records',config=dict(model=str(model),records='/FORBIDDEN/shared-generation',generation_manifest='/FORBIDDEN/final-test')))
                a.write(cp/'metadata.json',m)
                binding['checkpoints'][str(step)]=dict(path=str(cp),weights_sha256=a.sha(cp/'draft.safetensors'),metadata_sha256=a.sha(cp/'metadata.json'))
            real_open=Path.open
            def guarded(path,*args,**kwargs):
                self.assertNotIn('FORBIDDEN',str(path));self.assertNotIn('resume.pt',str(path));return real_open(path,*args,**kwargs)
            with patch.object(Path,'open',guarded):self.assertTrue(a.verify_dependencies(binding)['dependencies_verified'])
            (out/'512'/'draft.safetensors').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Checkpoint bytes'):a.verify_dependencies(binding)
            (out/'512'/'draft.safetensors').write_bytes(b'synthetic-draft-bytes');(model/'tokenizer.json').write_text('changed')
            with self.assertRaisesRegex(ValueError,'Target/tokenizer bytes'):a.verify_dependencies(binding)

    def test_default_cli_import_and_data_read_traps(self):
        code='''import builtins, importlib.util, pathlib, sys
original=builtins.__import__
def guard(name,*args,**kw):
 if name=='torch' or name.startswith('transformers') or name.startswith('safetensors'):raise AssertionError('model/runtime import')
 return original(name,*args,**kw)
builtins.__import__=guard
s=importlib.util.spec_from_file_location('a',sys.argv[1]);a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
a.bind_evidence=lambda *args:dict(dependencies_verified=False,source_sha256={})
a.verify_dependencies=lambda *args:(_ for _ in ()).throw(AssertionError('dependency read'))
a.supervise=lambda *args:(_ for _ in ()).throw(AssertionError('execution'))
assert a.main(['--probe-root','x','--quality-root','x','--historical-source','x','--evidence-only'])==0
'''
        p=subprocess.run([sys.executable,'-c',code,str(SCRIPT)],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr);self.assertFalse(json.loads(p.stdout)['dependencies_verified'])


class SupervisorTests(unittest.TestCase):
    def test_sysfs_unique_physical_device_and_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);device=root/'pci-device';device.mkdir();(device/'vendor').write_text('0x1002');(device/'driver').symlink_to(root/'amdgpu');(root/'amdgpu'/'module').mkdir(parents=True);(root/'amdgpu'/'module'/'srcversion').write_text('driver-source')
            (device/'mem_info_vram_total').write_text('100');(device/'mem_info_vram_used').write_text('20')
            for n in range(2):card=root/f'card{n}';card.mkdir();(card/'device').symlink_to(device)
            r=a.telemetry(root);self.assertEqual(r['used'],20);self.assertEqual(r['physical_device'],str(device.resolve()))
            other=root/'pci-other';shutil.copytree(device,other,symlinks=True);(root/'card1'/'device').unlink();(root/'card1'/'device').symlink_to(other)
            with self.assertRaisesRegex(ValueError,'Unique'):a.telemetry(root)

    def test_device_mapping_must_be_exact_and_driver_stable(self):
        baseline=dict(physical_device='/sys/devices/0000:03:00.0',total=100,driver_srcversion='v',kernel_release='k')
        a.device_gate(SimpleNamespace(pci_bus_id='0000:03:00.0',total_memory=100),baseline,baseline)
        for properties in (SimpleNamespace(total_memory=100),SimpleNamespace(pci_bus_id='0000:04:00.0',total_memory=100)):
            with self.assertRaisesRegex(ValueError,'PCI'):a.device_gate(properties,baseline,baseline)
        with self.assertRaisesRegex(ValueError,'baseline'):a.device_gate(SimpleNamespace(pci_bus_id='0000:03:00.0',total_memory=100),baseline,dict(baseline,driver_srcversion='changed'))

    def test_nonzero_exit_or_final_telemetry_revokes_completed_summary(self):
        baseline=dict(total=32*1024**3,used=0,physical_device='pci0',driver_srcversion='v',kernel_release='k')
        for final_failure in (False,True):
            with self.subTest(final_failure=final_failure),tempfile.TemporaryDirectory() as tmp:
                out=Path(tmp)/'run'
                def start(*args,**kwargs):
                    a.write(out/'summary.json',dict(status='completed',aggregate={'count':32}))
                    child=Mock();child.poll.return_value=0;child.wait.return_value=0 if final_failure else 1;return child
                with patch.object(a,'telemetry',side_effect=[baseline,OSError('lost telemetry') if final_failure else baseline]),patch.object(a.subprocess,'Popen',side_effect=start):
                    self.assertEqual(a.supervise({},out),1)
                self.assertEqual(a.read(out/'summary.json')['status'],'evidence_gap')
                self.assertTrue((out/'unconfirmed-summary.private.json').exists())
                self.assertIn('summary.json',a.read(out/'artifact-sha256.json'))

    def test_missing_telemetry_no_child_and_saved_exit(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(a,'telemetry',side_effect=OSError),patch.object(a.subprocess,'Popen') as popen:
            out=Path(tmp)/'run';self.assertEqual(a.supervise({},out),1);popen.assert_not_called()
            result=a.read(out/'supervision.json');self.assertIsNone(result['worker_os_exit']);self.assertFalse(result['worker_started'])

    def test_wall_rss_output_caps_reap_child(self):
        baseline=dict(total=32*1024**3,used=0,physical_device='pci0',driver_srcversion='v',kernel_release='k')
        for cap,reason in (('wall_seconds','wall_limit'),('rss_bytes','rss_limit'),('output_bytes','output_limit')):
            with self.subTest(cap=cap),tempfile.TemporaryDirectory() as tmp,patch.dict(a.CAPS,{cap:0}),patch.object(a,'telemetry',return_value=baseline),patch.object(a.subprocess,'Popen') as popen,patch.object(a.subprocess,'check_output',return_value='1'):
                child=popen.return_value;child.poll.return_value=None;child.pid=111;child.wait.return_value=-9
                out=Path(tmp)/'run';self.assertEqual(a.supervise({},out),1)
                child.kill.assert_called_once();child.wait.assert_called_once()
                self.assertEqual(a.read(out/'supervision.json')['limit_reason'],reason)

    def test_pressure_kills_only_owned_child_and_reaps_real_code(self):
        baseline=dict(total=32*1024**3,used=0,physical_device='pci0',driver_srcversion='version',kernel_release='kernel')
        high=dict(baseline,used=a.CAPS['device_used_delta_bytes']+1)
        with tempfile.TemporaryDirectory() as tmp,patch.object(a,'telemetry',side_effect=[baseline,high]),patch.object(a.subprocess,'Popen') as popen,patch.object(a.subprocess,'check_output',return_value='1'):
            child=popen.return_value;child.poll.return_value=None;child.pid=111;child.wait.return_value=-9
            out=Path(tmp)/'run';self.assertEqual(a.supervise({},out),1)
            child.kill.assert_called_once();child.wait.assert_called_once()
            result=a.read(out/'supervision.json');self.assertEqual(result['worker_os_exit'],-9);self.assertEqual(result['limit_reason'],'device_pressure_limit')
            self.assertTrue(result['child_reaped']);self.assertFalse(a.read(out/'summary.json')['quality_verdict'])


if __name__=='__main__':unittest.main()
