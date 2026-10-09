import builtins
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
probe=load('capacity_probe_test',ROOT/'scripts/probe_native_capacity_graph.py')
fixture=load('capacity_probe_binding_fixture',ROOT/'tests/test_probe_packed_decoder.py')

class CapacityProbeStdlibTests(unittest.TestCase):
    def test_real_architecture_binding_is_stdlib_and_defaults_to_no_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            m,g,_,_=fixture.PackedGateTests().binding_fixture(Path(tmp))
            workload=json.loads((ROOT/'configs/performance-workloads.example.json').read_text())
            workload['model']['config_sha256']=probe.GUARD.sha(m/'config.json');w=Path(tmp)/'workload.json';w.write_text(json.dumps(workload))
            original=builtins.__import__
            def guarded(name,*args,**kwargs):
                if name.split('.')[0] in ('torch','transformers','dspark_qwen'):raise AssertionError('Heavyweight import in binding')
                return original(name,*args,**kwargs)
            with patch('builtins.__import__',side_effect=guarded),patch('sys.stdout',new_callable=io.StringIO) as output:
                self.assertEqual(probe.main(['--model',str(m),'--generation-manifest',str(g),'--workloads',str(w)]),0)
            row=json.loads(output.getvalue());self.assertEqual(row['status'],'dry_run_no_torch_no_gpu')
            self.assertEqual(row['binding']['protocol']['physical_key_capacity'],293)
            self.assertEqual(len(probe.expected_observations()),18)
            self.assertFalse(row['binding']['protocol']['whole_model_graph'])
            self.assertEqual(row['binding']['protocol']['timeout_seconds'],300)

class CapacityProbeTensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import Qwen3Config,Qwen3ForCausalLM
        torch.manual_seed(800);torch.set_num_threads(2)
        cfg=Qwen3Config(vocab_size=1200,hidden_size=32,intermediate_size=48,num_hidden_layers=2,
            num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=256,attention_dropout=0.)
        cfg._attn_implementation='sdpa';cls.model=Qwen3ForCausalLM(cfg).eval().requires_grad_(False)
        manifest=json.loads((ROOT/'configs/performance-workloads.example.json').read_text())
        req=next(c['requests'] for c in manifest['cases'] if c['case_id']=='r2-c256')
        cls.cases=probe.extract_cases(cls.model,req)

    def oracle(self,q,k,v,cq,ck,maxq,maxk):
        import torch
        outputs=[]
        for i in range(len(cq)-1):
            a,b=int(cq[i]),int(cq[i+1]);c,d=int(ck[i]),int(ck[i+1]);n=b-a;length=d-c
            mask=torch.arange(length)[None,:]<=torch.arange(length-n,length)[:,None]
            outputs.append(torch.nn.functional.scaled_dot_product_attention(q[a:b].transpose(0,1)[None],k[c:d].transpose(0,1)[None],v[c:d].transpose(0,1)[None],attn_mask=mask[None,None],enable_gqa=True).squeeze(0).transpose(0,1))
        return torch.cat(outputs)

    def capture(self,fn):
        # CPU contract emulator only: it reruns Python tensor work and maintains a
        # fixed output allocation. It cannot establish native capture or replay.
        output=fn().clone()
        class Replay:
            def replay(self):output.copy_(fn());return output
        return Replay()

    def report(self):
        r=probe.new_report();r['stages']['real_qkv']='passed';r['runtime_scope']='cpu_contract_fixture_only';return r

    def test_real_tiny_qkv_extraction_and_complete_capacity_contract(self):
        self.assertEqual(self.model.config._attn_implementation,'sdpa')
        self.assertEqual([c['contexts'] for c in self.cases],[(128,128),(129,131),(130,135)])
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(probe,'gpu_capture',side_effect=AssertionError('CPU must not claim native graph')):
                r=probe.exercise(self.cases,self.oracle,self.capture,tmp,self.report())
            probe.verify_report(r);self.assertEqual(len(r['observations']),18)
            self.assertEqual(r['runtime_scope'],'cpu_contract_fixture_only')
            replay=[x for x in r['observations'] if x['stage']=='capture_replay']
            self.assertEqual(replay[0]['positions'],[128,128,129,130,131])
            self.assertEqual(replay[-1]['positions'],[130,135,136,137,138])
            self.assertNotEqual(replay[0]['cu_key'],replay[-1]['cu_key'])
            self.assertEqual(replay[0]['input_pointers'],replay[-1]['input_pointers'])
            self.assertEqual(len(list(Path(tmp).glob('eager-*-inputs.pt'))),15)
            exact=r['observations'][0]['operator_work'];tail=r['observations'][2]['operator_work']
            self.assertEqual(exact['physical_k_rows'],261);self.assertEqual(tail['physical_k_rows'],293)
            self.assertEqual(tail['padding_k_rows'],32)
            broken=copy.deepcopy(r);broken['observations'].pop()
            with self.assertRaises(ValueError):probe.verify_report(broken)

    def test_finite_tail_failure_keeps_raw_and_never_captures(self):
        def corrupt(q,k,v,cq,ck,maxq,maxk):
            value=self.oracle(q,k,v,cq,ck,maxq,maxk)
            if len(k)>int(ck[-1]) and float(k[-1,0,0])==100.:return value+1.
            return value
        with tempfile.TemporaryDirectory() as tmp,patch.object(self,'capture',side_effect=AssertionError('gate must stop')):
            with self.assertRaisesRegex(ValueError,'capacity_finite'):probe.exercise(self.cases,corrupt,self.capture,tmp,self.report())
            r=json.loads((Path(tmp)/'result.json').read_text())
            self.assertEqual(r['stages']['eager_tail'],'failed');self.assertEqual(r['stages']['capture_replay'],'pending')
            self.assertTrue((Path(tmp)/'eager-0-capacity_finite-inputs.pt').exists());self.assertTrue((Path(tmp)/'eager-0-capacity_finite-output.pt').exists())

    def test_nan_tail_rejection_is_required_gate_not_posthoc_exception(self):
        import torch
        def reject(q,k,v,cq,ck,maxq,maxk):
            if bool(torch.isnan(k).any()):raise RuntimeError('backend rejects NaN tail')
            return self.oracle(q,k,v,cq,ck,maxq,maxk)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError,'NaN tail'):probe.exercise(self.cases,reject,self.capture,tmp,self.report())
            r=json.loads((Path(tmp)/'result.json').read_text());self.assertEqual(r['stages']['capture_replay'],'pending')
            self.assertTrue((Path(tmp)/'eager-0-capacity_nan-inputs.pt').exists())
            self.assertFalse((Path(tmp)/'eager-0-capacity_nan-output.pt').exists())

    def test_values_only_contamination_fails_eager_and_replay_isolation(self):
        original=probe.make_pool
        for phase in ('eager','replay'):
            with self.subTest(phase=phase),tempfile.TemporaryDirectory() as tmp:
                pools=[]
                def make(cases):
                    result=original(cases);pools.append(result[0]);return result
                def native(*args):
                    output=self.oracle(*args)
                    if phase=='eager':pools[0].values.add_(1.)
                    return output
                def capture(fn):
                    graph=self.capture(fn)
                    class Replay:
                        def replay(self):
                            value=graph.replay();pools[-1].values.add_(1.);return value
                    return Replay()
                with patch.object(probe,'make_pool',side_effect=make),self.assertRaises((AssertionError,ValueError)):
                    probe.exercise(self.cases,native,capture,tmp,self.report())
                r=json.loads((Path(tmp)/'result.json').read_text())
                self.assertFalse(r['observations'][-1]['committed_unchanged_before_fixture_commit'])
                self.assertTrue(r['observations'][-1]['comparison']['passed'])
                self.assertEqual(r['status'],'failed')
                self.assertEqual(r['stages']['eager_tail'],'failed' if phase=='eager' else 'passed')

    def test_capture_error_preserves_passed_eager_and_deadline_keeps_expected_domain(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError,'capture failed'):
                probe.exercise(self.cases,self.oracle,lambda fn:(_ for _ in ()).throw(RuntimeError('capture failed')),tmp,self.report())
            r=json.loads((Path(tmp)/'result.json').read_text())
            self.assertEqual(r['stages']['eager_tail'],'passed');self.assertEqual(r['stages']['capture_replay'],'failed')
            self.assertEqual(len(r['observations']),15)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(TimeoutError):probe.exercise(self.cases,self.oracle,self.capture,tmp,self.report(),deadline=0.)
            r=json.loads((Path(tmp)/'result.json').read_text());self.assertEqual(r['status'],'partial_deadline');self.assertEqual(len(r['expected_observations']),18)

if __name__=='__main__':unittest.main()
