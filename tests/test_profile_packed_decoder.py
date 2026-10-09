import builtins
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock,patch

ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
profile=load('packed_profile',ROOT/'scripts/profile_packed_decoder.py')
fixture=load('packed_profile_fixture',ROOT/'tests/test_probe_packed_decoder.py')

class PackedProfileTests(unittest.TestCase):
    def test_complete_64_domain_and_rate_units(self):
        cells=profile.PROTOCOL['cells']
        self.assertEqual(len(cells),64);self.assertEqual({tuple(x['ell']) for x in cells},{(a,b) for a in range(8) for b in range(8)})
        self.assertEqual({x['logical_b'] for x in cells},set(range(2,17)))
        self.assertEqual(profile.statistics_for([{'wall_seconds':2.},{'wall_seconds':4.}])['rounds_per_second'],1/3)
        schedule=list(profile.schedule(profile.PROTOCOL))
        self.assertEqual(len(schedule),64*8)
        for phase,repeats in [('warmups',2),('primary',5),('diagnostic',1)]:
            for repeat in range(repeats):
                self.assertEqual({i for p,n,i in schedule if p==phase and n==repeat},set(range(64)))
        self.assertNotEqual([i for p,n,i in schedule if p=='primary' and n==0],[i for p,n,i in schedule if p=='primary' and n==1])

    def test_real_binding_is_stdlib_only_and_includes_supervisor(self):
        with tempfile.TemporaryDirectory() as tmp:
            m,g,c,h=fixture.PackedGateTests().binding_fixture(Path(tmp));original=builtins.__import__
            def guarded(name,*args,**kwargs):
                if name.split('.')[0] in ('torch','transformers','dspark_qwen'):raise AssertionError('heavyweight import')
                return original(name,*args,**kwargs)
            workloads=json.loads((ROOT/'configs/performance-workloads.example.json').read_text());workloads['model']['config_sha256']=profile.GUARD.sha(m/'config.json')
            path=Path(tmp)/'workloads.json';path.write_text(json.dumps(workloads))
            with patch.dict(profile.GATE.PROTOCOL,h),patch('builtins.__import__',side_effect=guarded),patch('sys.stdout',new_callable=io.StringIO) as stdout:
                self.assertEqual(profile.main(['--model',str(m),'--generation-manifest',str(g),'--checkpoint',str(c),'--workloads',str(path)]),0)
            result=json.loads(stdout.getvalue());self.assertEqual(len(result['binding']['protocol']['cells']),64)
            self.assertIn('scripts/guard_vllm_smoke.py',result['binding']['source_sha256'])
            self.assertIn('scripts/profile_packed_decoder.py',result['binding']['source_sha256'])

    def session(self):
        import torch
        from dspark_qwen.packed_sampling import RequestSpec
        from dspark_qwen.tensor_sampling import TensorRandom
        session,_=fixture.PackedGateTests().fixture()
        session.admit({r:RequestSpec(torch.arange(base,base+16)[None],128,TensorRandom(torch.Generator().manual_seed(seed)))
            for r,base,seed in [('A',100,3),('B',200,4)]})
        return session

    def test_snapshot_restores_real_KV_rng_and_repeated_round(self):
        session=self.session();frozen=profile.snapshot(session);identity=profile.snapshot_identity(frozen)
        original_models=(id(session.target.model),id(session.draft.draft))
        a=profile.round_once(session,[1,3]);self.assertNotEqual(session.target.lengths,frozen['target']['_lengths'])
        elapsed=profile.restore(session,frozen);self.assertGreater(elapsed,0)
        self.assertEqual(profile.snapshot_identity(profile.snapshot(session)),identity)
        b=profile.round_once(session,[1,3])
        self.assertEqual(a['requests'],b['requests']);self.assertEqual(a['work'],b['work'])
        self.assertEqual(original_models,(id(session.target.model),id(session.draft.draft)))
        self.assertEqual(a['work']['policy_confidence_host_values'],14)
        self.assertEqual(a['work']['proposal_batches'][0]['backbone']['query_rows'],14)

    def test_restore_never_revives_old_capability_even_when_fields_repeat(self):
        session=self.session();frozen=profile.snapshot(session)
        old=session.propose(['A','B'],mode='shadow')
        with self.assertRaisesRegex(ValueError,'outstanding'):profile.restore(session,frozen)
        session.verify_commit(old.proposals,{'A':0,'B':0},allocation_policy='external_nonanticipating')
        profile.restore(session,frozen)
        new=session.propose(['A','B'],mode='shadow')
        self.assertEqual(old.proposals['A'].nonce,new.proposals['A'].nonce)
        self.assertEqual(old.proposals['A'].epoch,new.proposals['A'].epoch)
        with self.assertRaisesRegex(ValueError,'Stale'):
            session.verify_commit(old.proposals,{'A':0,'B':0},allocation_policy='external_nonanticipating')
        session.verify_commit(new.proposals,{'A':0,'B':0},allocation_policy='external_nonanticipating')

    def test_launcher_reuses_identity_supervisor_and_exact_deadline(self):
        binding={'fixture':'stdlib'};guard=Mock()
        with tempfile.TemporaryDirectory() as tmp,patch.object(profile,'bind',return_value=binding),patch.object(profile.GUARD,'ASRGuard',return_value=guard) as make_guard,patch.object(profile.GUARD,'supervise',return_value={'status':'completed','released':True}) as supervise:
            self.assertEqual(profile.main(['--execute','--model','m','--generation-manifest','g','--checkpoint','c','--output',str(Path(tmp)/'run'),'--asr-pid','123','--asr-url','http://example.invalid/health']),0)
            make_guard.assert_called_once_with(123,'http://example.invalid/health',pre_free_bytes=8*1024**3)
            args,kwargs=supervise.call_args
            self.assertIs(args[2],guard);self.assertEqual(kwargs['timeout'],300)
            self.assertIn('--worker-binding',args[0]);self.assertIn('supervision',str(args[1]))

    def test_separate_passes_complete_cell_and_lookup_refuses_other_domain(self):
        session=self.session();spec=copy.deepcopy(profile.PROTOCOL)
        spec.update(cells=[dict(ell=[0,0],logical_b=2),dict(ell=[2,1],logical_b=5)],warmups_per_cell=1,primary_samples_per_cell=2,diagnostic_samples_per_cell=1)
        with tempfile.TemporaryDirectory() as tmp:
            r=profile.exercise(session,Path(tmp)/'run',{'test':'tiny_cpu'},spec=spec)
            self.assertEqual(r['status'],'completed')
            for row in r['cells']:
                self.assertEqual(len(row['warmups']),1);self.assertEqual(len(row['primary']),2);self.assertEqual(len(row['diagnostic']),1)
                self.assertNotIn('diagnostic_regions',row['primary'][0])
                self.assertTrue(row['diagnostic'][0]['diagnostic_regions'])
                self.assertEqual(row['primary'][0]['physical_b'],row['logical_b'])
                self.assertEqual(profile.lookup(r,r['domain'],row['ell'])['samples'],2)
                self.assertFalse(profile.lookup(r,r['domain'],row['ell'])['capacity_round_driver_profile_eligible'])
            with self.assertRaisesRegex(ValueError,'planner/history'):profile.lookup(r,r['domain'],[0,0],purpose='capacity_round_driver')
            with self.assertRaisesRegex(ValueError,'domain'):profile.lookup(r,{'R':3},[0,0])
            with self.assertRaisesRegex(ValueError,'Unmeasured'):profile.lookup(r,r['domain'],[7,7])

    def test_deadline_preserves_all_unmeasured_cells_and_rejects_lookup(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=profile.exercise(self.session(),Path(tmp)/'run',{'test':'tiny_cpu'},deadline=time.monotonic()-1)
            self.assertEqual(r['status'],'partial_deadline');self.assertEqual(len(r['cells']),64)
            self.assertTrue(all(x['status']=='unmeasured' for x in r['cells']))
            with self.assertRaisesRegex(ValueError,'Unmeasured'):profile.lookup(r,r['domain'],[0,0])

    def test_partial_failure_keeps_prior_samples_and_all_cells(self):
        session=self.session();spec=copy.deepcopy(profile.PROTOCOL);spec.update(warmups_per_cell=0,primary_samples_per_cell=1)
        original=profile.round_once;calls=[]
        def fail_second(*args,**kwargs):
            calls.append(None)
            if len(calls)==2:raise RuntimeError('injected round failure')
            return original(*args,**kwargs)
        with tempfile.TemporaryDirectory() as tmp,patch.object(profile,'round_once',side_effect=fail_second):
            out=Path(tmp)/'run'
            with self.assertRaisesRegex(RuntimeError,'injected'):profile.exercise(session,out,{'test':'tiny_cpu'},spec=spec)
            r=json.loads((out/'result.json').read_text());self.assertEqual(r['status'],'failed');self.assertEqual(len(r['cells']),64)
            self.assertEqual(len(r['cells'][0]['primary']),1)
            with self.assertRaisesRegex(ValueError,'Unmeasured'):profile.lookup(r,r['domain'],[0,0])

if __name__=='__main__':unittest.main()
