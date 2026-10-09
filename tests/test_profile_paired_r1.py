import contextlib
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import sys
import subprocess
import shutil
import os
import signal
import time
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
profile=load('tested_profile_runner',ROOT/'scripts/profile_paired_r1.py')
fixture=load('profile_paired_fixture',ROOT/'tests/test_benchmark_paired_r1.py')
from dspark_qwen.paired_observer import Observer,ResourceMonitor,LIMITS,analyze_trace,union_us
from dspark_qwen.profile_resource_monitor import ResourceProbe


class FakeEvent:
    def __init__(self,enable_timing=True):self.when=None
    def record(self):self.when=time.perf_counter()
    def elapsed_time(self,other):return (other.when-self.when)*1000
class FakeEvents:
    Event=FakeEvent
    @staticmethod
    def synchronize():pass


class ProfileObserverTests(fixture.PairedR1RealFactoryTests):
    # Inherit only the useful tiny factory fixture, not its test methods below.
    def test_observer_preserves_full128_outputs_rng_kv_and_work_both_arms_cases(self):
        import torch
        from dspark_qwen.packed_target_sampling import PackedTargetOnlySession
        from dspark_qwen.tensor_sampling import TensorRandom
        make,device=self.factory()
        original_step=PackedTargetOnlySession.step;original_rng=TensorRandom.uniform
        class CaptureFactory:
            def __init__(self,factory):self.factory=factory;self.target=factory.target;self.draft=factory.draft;self.witness=factory.witness;self.session=None
            def __call__(self,arm):self.session=self.factory(arm);return self.session
        factory=CaptureFactory(make)
        for case in profile.BASE.selected_manifest(fixture.MANIFEST)['cases']:
            for arm in profile.BASE.ARMS:
                plain=profile.BASE.batch(factory,device,case,fixture.MANIFEST,arm)
                rng=factory.session.requests['r0']['rng'].generator.get_state().clone()
                kv=tuple(x.clone() for pair in factory.target.request_kv('r0') for x in pair)
                with Observer(factory,profile.BASE.NATIVE,events=True,event_backend=FakeEvents) as obs:
                    observed=profile.BASE.batch(factory,device,case,fixture.MANIFEST,arm,diagnostic=True)
                self.assertIs(PackedTargetOnlySession.step,original_step);self.assertIs(TensorRandom.uniform,original_rng)
                self.assertTrue(obs.closed);self.assertFalse(obs.patches)
                self.assertEqual(profile.invariant(plain),profile.invariant(observed))
                self.assertTrue(torch.equal(rng,factory.session.requests['r0']['rng'].generator.get_state()))
                for expected,actual in zip(kv,(x for pair in factory.target.request_kv('r0') for x in pair)):self.assertTrue(torch.equal(expected,actual))
                stats=obs.summarize();self.assertFalse(obs.event_pairs)
                self.assertEqual(len(stats['rounds']),len(observed['rounds']))
                self.assertEqual([x['query_tokens'] for x in stats['rounds']],[x['query_tokens'] for x in observed['rounds']])
                self.assertTrue(all(x['exclusive_host_us']>=0 for x in stats['spans']))
                self.assertTrue(all(x['gpu_stream_end_us']>=x['gpu_stream_start_us'] for x in stats['spans']))
                self.assertIn('fp64_logits_to_law',stats['stages']);self.assertIn('lm_head',stats['stages'])
                self.assertEqual('draft_backbone' in stats['stages'],arm!=profile.BASE.ARMS[0])

    def test_span_failure_restores_originals_and_saves_partial_artifact(self):
        import torch
        from dspark_qwen.packed_target_sampling import PackedTargetOnlySession
        make,device=self.factory();original=PackedTargetOnlySession.step
        class HostObserver(Observer):
            def __init__(self,*args,**kwargs):super().__init__(*args,**dict(kwargs,events=False))
        class Monitor:
            max_rss=1
            def __init__(self,*args,**kwargs):pass
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def check(self):pass
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'diag'
            with patch.object(profile.BASE,'batch',side_effect=RuntimeError('injected batch failure')):
                with self.assertRaisesRegex(RuntimeError,'injected batch'):
                    profile.diagnostic(make,device,profile.BASE.selected_manifest(fixture.MANIFEST)['cases'][0],fixture.MANIFEST,
                        'target_only',0,out,deadline=None,result_path=Path(tmp)/'result.json',observer_class=HostObserver,monitor_class=Monitor)
            self.assertIs(PackedTargetOnlySession.step,original)
            self.assertEqual(json.loads((out/'status.json').read_text())['status'],'failed')
            self.assertTrue(json.loads((out/'partial-observation.json').read_text())['observer_restored'])
            self.assertIn('injected batch failure',(out/'error.txt').read_text())

# Remove inherited tests: running this module exercises only the new observation
# interface on the shared real fixture, not a redundant full original suite.
for _name in dir(fixture.PairedR1RealFactoryTests):
    if _name.startswith('test_') and _name not in ProfileObserverTests.__dict__:setattr(ProfileObserverTests,_name,None)


class ProfileBoundaryTests(unittest.TestCase):
    def trace(self):
        def event(cat,name,ts,dur,args=None):return dict(ph='X',cat=cat,name=name,ts=ts,dur=dur,args=args or {},pid=1,tid=1)
        return dict(traceEvents=[event('user_annotation','paired::complete_session',0,100),
            event('cpu_op','aten::mm',10,70,{'External id':1}),event('hip_runtime','hipLaunchKernel',20,5,{'External id':1,'correlation':9}),
            event('kernel','actual_gemm_name',30,30,{'correlation':9}),event('kernel','unknown_graph_kernel',40,30,{'correlation':10})])

    def test_gpu_trace_requires_session_gpu_and_usable_correlation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'trace.json';trace=self.trace();p.write_text(json.dumps(trace));result=analyze_trace(p)
            self.assertEqual(result['gpu_active_union_us'],40);self.assertEqual(result['correlated_gpu_activities'],1)
            self.assertEqual(result['unknown_gpu_associations'],1)
            for change in ('gpu','correlation','outside','session'):
                t=copy.deepcopy(trace)
                if change=='gpu':t['traceEvents']=[x for x in t['traceEvents'] if x['cat']!='kernel']
                if change=='correlation':t['traceEvents'][2]['args']['External id']=999
                if change=='outside':t['traceEvents'][3]['ts']=200
                if change=='session':t['traceEvents'][0]['name']='other'
                p.write_text(json.dumps(t))
                with self.assertRaises(RuntimeError):analyze_trace(p)
        self.assertEqual(union_us([(0,10),(5,15),(20,25)]),20)

    def test_invalid_ids_missing_threads_and_duplicate_runtime_are_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'trace.json'
            for value in (None,False,True,-1,0,'1',1.0):
                for field in ('external','correlation'):
                    trace=self.trace()
                    if field=='external':
                        trace['traceEvents'][1]['args']['External id']=value
                        trace['traceEvents'][2]['args']['External id']=value
                    else:
                        trace['traceEvents'][2]['args']['correlation']=value
                        trace['traceEvents'][3]['args']['correlation']=value
                    path.write_text(json.dumps(trace))
                    with self.subTest(field=field,value=value),self.assertRaises(RuntimeError):analyze_trace(path)
            for field in ('pid','tid'):
                for value in (None,False,0,-1,'1'):
                    trace=self.trace()
                    for event in trace['traceEvents']:event[field]=value
                    path.write_text(json.dumps(trace))
                    with self.subTest(thread=field,value=value),self.assertRaises(RuntimeError):analyze_trace(path)
            trace=self.trace();extra=copy.deepcopy(trace['traceEvents'][2]);extra['args']['External id']=999
            trace['traceEvents'].append(extra);path.write_text(json.dumps(trace))
            with self.assertRaises(RuntimeError):analyze_trace(path)

    def test_trace_assigns_round_and_stage_via_cpu_chain_not_gpu_overlap(self):
        trace=self.trace()
        trace['traceEvents'] += [dict(ph='X',cat='user_annotation',name='paired::complete_round#1',ts=5,dur=90,pid=1,tid=1),
            dict(ph='X',cat='user_annotation',name='paired::lm_head#2',ts=8,dur=75,pid=1,tid=1)]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'trace.json';path.write_text(json.dumps(trace))
            result=analyze_trace(path,dict(rounds=[dict(index=0,span_id=1,query_tokens=7,context_lengths={'r0':184},execution_kind='native_eager_tail',eager_tail=True)]))
            self.assertEqual(result['rounds'][0]['query_tokens'],7)
            self.assertEqual(result['rounds'][0]['correlated_gpu']['active_union_us'],30)
            self.assertEqual(result['stages']['paired::lm_head#2']['active_union_us'],30)
            self.assertEqual(result['activities'][1]['association'],'unknown')
            self.assertIsNone(result['activities'][1]['round'])

    def test_resource_limits_abort_preserving_stage_and_no_silent_truncation(self):
        for kind in ('rss','file','total','count','deadline'):
            with tempfile.TemporaryDirectory() as tmp:
                b=Path(tmp);result=b/'result.json';result.write_text(json.dumps(dict(status='running',sample_count=28)))
                limits=dict(LIMITS,rss_bytes=100,trace_bytes=10,total_trace_bytes=15,max_traces=2)
                for i,size in enumerate({'file':[11],'total':[8,8],'count':[1,1,1]}.get(kind,[])):
                    d=b/'diagnostics'/str(i);d.mkdir(parents=True);(d/'trace.json.partial').write_bytes(b'x'*size)
                stopped=[];monitor=ResourceProbe(b/'diagnostics',result,limits=limits,rss=lambda:101 if kind=='rss' else 10,
                    fatal=stopped.append,deadline=time.monotonic()-1 if kind=='deadline' else None)
                with self.assertRaisesRegex(RuntimeError,'abort'):monitor.check()
                self.assertEqual(stopped,[86]);saved=json.loads(result.read_text());self.assertEqual(saved['sample_count'],28)
                self.assertEqual(saved['status'],'failed');self.assertTrue((b/'diagnostics/resource-failure.json').exists())
                self.assertEqual(sum(p.stat().st_size for p in (b/'diagnostics').glob('*/trace.json.partial')),sum({'file':[11],'total':[8,8],'count':[1,1,1]}.get(kind,[])))

    def test_primary_direct_call_and_diagnostic_failure_keep28_samples(self):
        from dspark_qwen.packed_target_sampling import PackedTargetOnlySession
        original=PackedTargetOnlySession.step;calls=[]
        def plain(factory,device,case,manifest,arm,**kwargs):
            self.assertIs(PackedTargetOnlySession.step,original);self.assertFalse(kwargs['diagnostic']);calls.append('plain')
            return fixture.PairedR1ProtocolTests.fake_batch(factory,device,case,manifest,arm)
        def failure(*args,**kwargs):calls.append('diagnostic');raise RuntimeError('trace export fails')
        with tempfile.TemporaryDirectory() as tmp,patch.object(profile.BASE,'batch',side_effect=plain),patch.object(profile,'diagnostic',side_effect=failure),patch.object(profile,'invariant',return_value='same'):
            with self.assertRaisesRegex(RuntimeError,'trace export'):
                profile.run(None,None,fixture.MANIFEST,Path(tmp))
            result=json.loads((Path(tmp)/'result.json').read_text())
            self.assertEqual(result['sample_count'],28);self.assertFalse(result['profiling_complete'])
            self.assertEqual(calls,['plain']*28+['diagnostic'])
            self.assertTrue(all(x['complete_primary_pair'] for x in result['aggregates']['paired_comparisons']))

    def test_resource_rename_race_then_final_size_enforcement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'diagnostics';directory=root/'one';directory.mkdir(parents=True)
            partial=directory/'trace.json.partial';final=directory/'trace.json';partial.write_bytes(b'x'*11)
            original=Path.stat;changed=[]
            def racing(path,*args,**kwargs):
                if path==partial and not changed:
                    changed.append(True);partial.rename(final);raise FileNotFoundError('atomic rename')
                return original(path,*args,**kwargs)
            monitor=ResourceProbe(root,Path(tmp)/'result.json',limits=dict(LIMITS,trace_bytes=10),rss=lambda:1,fatal=lambda code:None)
            with patch.object(Path,'stat',racing):self.assertNotIn('failure',monitor.inspect())
            self.assertEqual(monitor.inspect()['failure'],'trace_size_limit')

    def test_exact_four_final_traces_and_eight_complete_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'diagnostics'
            for case in profile.BASE.CASE_IDS:
                for arm in profile.BASE.ARMS:
                    for repeat in (0,1):
                        p=root/f'{case}-{arm}-{repeat}';p.mkdir(parents=True)
                        (p/'status.json').write_text(json.dumps(dict(status='completed',observer_restored=True,complete_session=True)))
                        if repeat==1:(p/'trace.json').write_text('{}')
            self.assertEqual(len(profile.validate_diagnostics(Path(tmp))),4)
            trace=next(root.glob('*/trace.json'));trace.rename(trace.with_suffix('.json.partial'))
            with self.assertRaisesRegex(ValueError,'Exactly four'):profile.validate_diagnostics(Path(tmp))

    def test_resource_write_failure_still_invokes_hard_abort(self):
        with tempfile.TemporaryDirectory() as tmp:
            stopped=[]
            monitor=ResourceProbe(Path(tmp)/'diagnostics',Path(tmp)/'result.json',rss=lambda:LIMITS['rss_bytes']+1,fatal=stopped.append)
            with patch('dspark_qwen.profile_resource_monitor.write',side_effect=OSError('disk full')):
                with self.assertRaisesRegex(RuntimeError,'abort'):monitor.check()
            self.assertEqual(stopped,[86])

    def test_startup_budget_failure_and_existing_deadline_evidence_preserved(self):
        exc=RuntimeError('graph reservation exceeded');exc.memory_accounting={'actual':999};exc.pool_snapshot=[{'segment':999}]
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'run'
            with patch.object(profile,'worker',side_effect=exc):
                self.assertEqual(profile.main(['--worker-binding','unused','--output',str(out)]),1)
            self.assertEqual(json.loads((out/'failed-graph-pool-snapshot.json').read_text()),exc.pool_snapshot)
            self.assertEqual(json.loads((out/'result.json').read_text())['memory_accounting'],exc.memory_accounting)
            prior=dict(status='partial_deadline',sample_count=35,error='original deadline',aggregates={'retained':True})
            (out/'result.json').write_text(json.dumps(prior))
            with patch.object(profile,'worker',side_effect=TimeoutError('outer')):
                self.assertEqual(profile.main(['--worker-binding','unused','--output',str(out)]),1)
            self.assertEqual(json.loads((out/'result.json').read_text()),prior)

    def test_stdlib_binding_contains_profile_sources_and_child_enters_new_script(self):
        import builtins
        import io
        with tempfile.TemporaryDirectory() as tmp:
            model,generation,checkpoint,hashes=fixture.fixture.PackedGateTests().binding_fixture(Path(tmp))
            manifest=copy.deepcopy(fixture.MANIFEST);manifest['model']['config_sha256']=profile.GUARD.sha(model/'config.json')
            path=Path(tmp)/'workload.json';path.write_text(json.dumps(manifest))
            original=builtins.__import__
            def guarded(name,*args,**kwargs):
                if name.split('.')[0] in ('torch','transformers','dspark_qwen'):raise AssertionError('heavy import during binding')
                return original(name,*args,**kwargs)
            arguments=['--model',str(model),'--generation-manifest',str(generation),'--checkpoint',str(checkpoint),'--workloads',str(path)]
            with patch.dict(profile.BASE.PROFILE.GATE.PROTOCOL,hashes),patch.object(profile.BASE,'baseline_reference',return_value={'fixture':True}),patch('builtins.__import__',side_effect=guarded),patch('sys.stdout',new_callable=io.StringIO) as stdout:
                self.assertEqual(profile.main(['--dry-run']+arguments),0)
                binding=json.loads(stdout.getvalue())['binding']
            self.assertEqual(binding['worker_entry'],'scripts/profile_paired_r1.py')
            self.assertEqual(binding['profile_protocol'],profile.PROFILE_PROTOCOL)
            for name in ['scripts/benchmark_paired_r1.py','scripts/profile_paired_r1.py','dspark_qwen/paired_observer.py','dspark_qwen/profile_resource_monitor.py','tests/test_profile_paired_r1.py','docs/paired-r1-profile.md']:
                self.assertEqual(binding['source_sha256'][name],profile.GUARD.sha(ROOT/name))
            def supervise(command,*args,**kwargs):
                self.assertEqual(Path(command[2]).resolve(),(ROOT/'scripts/profile_paired_r1.py').resolve())
                self.assertIn('--worker-binding',command)
                return dict(status='failed',released=True)
            with patch.object(profile,'bind',return_value=binding),patch.object(profile.GUARD,'ASRGuard'),patch.object(profile.GUARD,'supervise',side_effect=supervise):
                self.assertEqual(profile.main(arguments+['--execute','--output',str(Path(tmp)/'execute'),'--asr-pid','1','--asr-url','http://unused']),1)

    def test_worker_dispatch_binding_and_run_restored(self):
        old_bind,old_run=profile.BASE.bind,profile.BASE.run
        def fake_worker(binding,out):
            self.assertIs(profile.BASE.bind,profile.bind);self.assertIs(profile.BASE.run,profile.run)
            raise RuntimeError('worker failure')
        with patch.object(profile.BASE,'worker',side_effect=fake_worker):
            with self.assertRaisesRegex(RuntimeError,'worker failure'):profile.worker('binding','out')
        self.assertIs(profile.BASE.bind,old_bind);self.assertIs(profile.BASE.run,old_run)


@unittest.skipUnless(sys.platform.startswith('linux'),'Real /proc+pidfd monitor regression runs only on Linux')
class IndependentLinuxMonitorTests(unittest.TestCase):
    def setUp(self):
        import ctypes
        if ctypes.CDLL(None).prctl(36,1,0,0,0)!=0:raise RuntimeError('test subreaper setup failed')

    def exercise(self,abort):
        import time
        from dspark_qwen.profile_resource_monitor import process_record,same_process
        code = """
import ctypes,json,sys,time
from pathlib import Path
from dspark_qwen.profile_resource_monitor import ResourceMonitor,LIMITS,write
root=Path(sys.argv[1]);diag=root/'diagnostics'/'session';diag.mkdir(parents=True)
(root/'result.json').write_text(json.dumps(dict(status='running',sample_count=28)))
with ResourceMonitor(root/'diagnostics',root/'result.json',state_directory=diag,limits=dict(LIMITS,monitor_seconds=.05,trace_bytes=10)) as monitor:
    write(root/'gil-start.json',dict(start=time.monotonic()))
    ctypes.PyDLL(None).sleep(2)
    write(root/'gil-end.json',dict(end=time.monotonic()))
"""
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);worker=subprocess.Popen([sys.executable,'-c',code,tmp],cwd=ROOT,start_new_session=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            identity=None;monitor_pid=None
            try:
                deadline=time.monotonic()+10
                while not (root/'gil-start.json').exists() and worker.poll() is None and time.monotonic()<deadline:time.sleep(.01)
                self.assertTrue((root/'gil-start.json').exists(),'worker/monitor startup failed')
                state=root/'diagnostics/session';ready=json.loads((state/'monitor-state.json').read_text())
                identity=ready['worker'];monitor_pid=ready['monitor']['pid'];self.assertTrue(same_process(identity))
                if abort:(state/'trace.json.partial').write_bytes(b'x'*11)
                stdout,stderr=worker.communicate(timeout=8)
                self.assertEqual(worker.returncode,-signal.SIGKILL if abort else 0,stderr.decode())
                if abort:
                    # Monitor was reparented to this explicit test subreaper.
                    until=time.monotonic()+3;reaped=None
                    while time.monotonic()<until:
                        pid,status=os.waitpid(monitor_pid,os.WNOHANG)
                        if pid:reaped=status;break
                        time.sleep(.01)
                    self.assertIsNotNone(reaped);self.assertEqual(os.waitstatus_to_exitcode(reaped),86)
                    self.assertFalse(same_process(identity));self.assertFalse((root/'gil-end.json').exists())
                    self.assertEqual(json.loads((root/'result.json').read_text())['status'],'failed')
                    self.assertEqual((state/'trace.json.partial').stat().st_size,11)
                else:
                    self.assertEqual(json.loads((state/'monitor-exit.json').read_text())['os_exit'],0)
                    start=json.loads((root/'gil-start.json').read_text())['start'];end=json.loads((root/'gil-end.json').read_text())['end']
                    samples=[json.loads(line) for line in (state/'monitor-samples.jsonl').read_text().splitlines()]
                    during=[x for x in samples if start<x['monotonic']<end]
                    self.assertGreaterEqual(len(during),10)
                    self.assertLessEqual(max(x['gap_seconds'] for x in samples),1)
                    self.assertTrue(all(x['worker']['pid']==worker.pid for x in samples))
            finally:
                if worker.poll() is None:os.killpg(worker.pid,signal.SIGKILL);worker.wait(timeout=3)
                if monitor_pid:
                    try:os.waitpid(monitor_pid,os.WNOHANG)
                    except ChildProcessError:pass
                if os.environ.get('PAIRED_PROFILE_TEST_EVIDENCE'):
                    destination=Path(os.environ['PAIRED_PROFILE_TEST_EVIDENCE'])/('gil-abort' if abort else 'gil-hold')
                    shutil.copytree(root,destination)

    def test_independent_sampling_continues_during_native_gil_hold(self):self.exercise(False)
    def test_limit_aborts_only_bound_worker_during_native_gil_hold(self):self.exercise(True)


if __name__=='__main__':unittest.main()
