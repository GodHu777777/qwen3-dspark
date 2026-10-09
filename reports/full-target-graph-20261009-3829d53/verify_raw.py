"""Independent CPU audit of immutable full-target graph private artifacts."""
from pathlib import Path
import argparse,json,hashlib,time
import torch

torch.set_num_threads(2)
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--evidence-root',type=Path,required=True,help='Directory containing complete/, source.tar, expected-source.json and ssh.os-exit')
parser.add_argument('--output',type=Path,help='Optional scalar audit JSON; defaults to evidence-root/raw-cpu-audit.json')
args=parser.parse_args();base=args.evidence_root.resolve();remote=base/'complete';worker=remote/'run/worker'
report=json.loads((worker/'result.json').read_text());binding=json.loads((remote/'run/binding.json').read_text())
protocol=json.loads((worker/'protocol.json').read_text());expected_source=json.loads((base/'expected-source.json').read_text())
assert not torch.cuda.is_available()
start=time.monotonic();checks=[];numerical=[];files={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check(ok,kind,**detail):
    row=dict(kind=kind,passed=bool(ok),**detail);checks.append(row)
    if not ok:raise AssertionError(row)
def load(name):
    p=worker/name;files[name]=sha(p);return torch.load(p,map_location='cpu',weights_only=True)
def equivalent(actual,expected,label,lengths):
    check(actual.shape==expected.shape and actual.dtype==expected.dtype,'shape_dtype',label=label)
    for request,lo,hi in [('pooled',0,len(actual))]+[(str(i),sum(lengths[:i]),sum(lengths[:i+1])) for i in range(len(lengths))]:
        a,b=actual[lo:hi],expected[lo:hi];finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        check(finite,'finite',label=label,request=request)
        delta=(a.double()-b.double()).abs();rms=float(delta.square().mean().sqrt())
        mismatches=int((delta>.02+.02*b.double().abs()).sum())
        row=dict(label=label,request=request,elements=a.numel(),max_abs=float(delta.max()),rms=rms,elementwise_mismatches=mismatches,bit_equal=torch.equal(a,b),storage_byte_equal=torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8)),passed=mismatches==0 and rms<=.005)
        numerical.append(row);check(row['passed'],'frozen_numerical_limits',label=label,request=request)

def comparisons(a,b,prefix):
    for i,layer in enumerate((1,7,14,21,26)):
        equivalent(a['context'][0,:,i*1024:(i+1)*1024],b['context'][0,:,i*1024:(i+1)*1024],prefix+f'/raw{layer}',(1,4))
    for key in ('final_norm','logits'):equivalent(a[key][0],b[key][0],prefix+'/'+key,(1,4))
    for layer in range(28):
        for key in ('scratch_keys','scratch_values'):equivalent(a[key][layer],b[key][layer],prefix+f'/{key}/{layer}',(1,4))

check(report['status']=='completed' and all(v=='passed' for v in report['stages'].values()),'completed_stages')
check(len(report['observations'])==6 and len(report['structural_checks'])>0 and all(r['passed'] for r in report['structural_checks']),'reported_complete_domain')
check(protocol['selected_layers']==[1,7,14,21,26] and protocol['comparisons']['atol']==.02 and protocol['comparisons']['rtol']==.02 and protocol['comparisons']['max_rms']==.005,'unchanged_protocol')
check(protocol==binding['protocol'],'protocol_binding')
check(sha(base/'source.tar')==expected_source['archive_sha256'],'immutable_archive')
for name,h in expected_source['files'].items():check(sha(remote/'source'/name)==h,'source_file',file=name)
initial=load('initial-resident.pt');before_eager={k:v.clone() for k,v in initial.items()};before_graph={k:v.clone() for k,v in initial.items()}
previous_tokens=None;previous_positions=None
for index,(contexts,counts) in enumerate(zip(((128,128),(129,131),(130,135)),((1,3),(1,4),(0,0)))):
    eager=load(f'eager-{index}-outputs.pt');isolated=load(f'eager-{index}-inactive-poison-outputs.pt');graph=load(f'replay-{index}-outputs.pt')
    witnesses=load(f'eager-{index}-independent-layer-witnesses.pt');tokens=load(f'eager-{index}-request-tokens.pt');metadata=load(f'eager-{index}-prepared-metadata.pt');replay_input=load(f'replay-{index}-inputs.pt')
    positions=[contexts[0],*range(contexts[1],contexts[1]+4)];cu_key=[0,contexts[0]+1,contexts[0]+contexts[1]+5]
    check(metadata['positions'].tolist()==positions and replay_input['positions'].tolist()==positions,'actual_positions',case=index)
    for m in (metadata,replay_input):check(m['cu_query'].tolist()==[0,1,5] and m['cu_key'].tolist()==cu_key,'actual_cumulative_lengths',case=index)
    wanted=torch.tensor([[t for i,(c,q) in enumerate(zip(contexts,(1,4))) for t in binding['requests'][i]['prompt_token_ids'][c:c+q]]])
    check(torch.equal(replay_input['input_ids'],wanted) and torch.equal(torch.cat([tokens['r0'],tokens['r1']],dim=1),wanted),'bound_exact_token_slices',case=index)
    if previous_tokens is not None:check(not torch.equal(wanted,previous_tokens) and positions!=previous_positions,'replay_inputs_changed',case=index)
    previous_tokens=wanted;previous_positions=positions
    for slice_index,layer in enumerate((1,7,14,21,26)):
        check(torch.equal(eager['context'][:,:,slice_index*1024:(slice_index+1)*1024],witnesses[f'raw_layer_{layer}']),'selected_raw_identity',case=index,layer=layer)
    check(torch.equal(eager['final_norm'],witnesses['final_norm']),'original_final_norm_identity',case=index)
    check(all(torch.equal(eager[k],isolated[k]) for k in eager),'inactive_poison_exact_outputs_and_scratch',case=index)
    comparisons(graph,eager,f'graph_vs_native_eager/{index}')
    eager_after=load(f'eager-{index}-committed.pt');graph_after=load(f'replay-{index}-committed.pt')
    for phase,before,outputs,after in (('eager',before_eager,eager,eager_after),('graph',before_graph,graph,graph_after)):
        reconstructed={k:v.clone() for k,v in before.items()};offset=0
        for slot,(c,n,q) in enumerate(zip(contexts,counts,(1,4))):
            for resident_name,scratch_name in (('keys','scratch_keys'),('values','scratch_values')):
                reconstructed[resident_name][:,slot,c:c+n]=outputs[scratch_name][:,offset:offset+n]
            offset+=q
        check(all(torch.equal(reconstructed[k],after[k]) for k in after),'exact_reconstructed_committed_or_rollback_all_bytes',phase=phase,case=index)
        check(all(torch.equal(after[k][:,2],initial[k][:,2]) for k in after),'inactive_resident_exact',phase=phase,case=index)
    for slot,(c,n) in enumerate(zip(contexts,counts)):
        for layer in range(28):
            for key in ('keys','values'):
                equivalent(graph_after[key][layer,slot,:c+n],eager_after[key][layer,slot,:c+n],f'committed/{index}/r{slot}/{key}/{layer}',(c+n,))
    before_eager=eager_after;before_graph=graph_after
    observation=report['observations'][index+3]
    check(observation['stage']=='graph_replay' and observation['case']==index and observation['python_before']==observation['python_after'] and observation['python_unchanged'],'python_free_replay_counters',case=index)
    check(observation['positions']==positions,'reported_positions_match_tensor',case=index)
capture_tokens=load('capture-request-tokens.pt');capture=load('capture-prepared-inputs.pt')
check(torch.equal(capture['input_ids'],torch.cat([capture_tokens['r0'],capture_tokens['r1']],dim=1)),'capture_bound_input_ids')
check(capture['positions'].tolist()==[128,128,129,130,131] and capture['cu_key'].tolist()==[0,129,261],'actual_capture_metadata')
counts=report['capture'];check(counts['python_after']['model_forward']-counts['python_before']['model_forward']==3 and all(a-b==3 for a,b in zip(counts['python_after']['decoder_hooks'],counts['python_before']['decoder_hooks'])),'capture_three_python_model_calls')
memory=counts['graph_memory'];check(memory['private_pool_reserved_bytes']==sum(s['total_size'] for s in memory['segments']) and memory['private_pool_reserved_bytes']<=memory['reservation_bytes']==512*1024**2,'private_pool_segment_accounting')
check(memory['global_reserved_delta']==memory['global_reserved_after']-memory['global_reserved_before'],'global_delta_separate')
for row in report['observations']:
    for key in ('inputs','outputs','inactive_poison_outputs'):
        if key in row:check(sha(worker/row[key]['file'])==row[key]['sha256'],'reported_artifact_hash',file=row[key]['file'])
for file in ('post-input-integrity.json','run/post-input-integrity.json','run/worker/post-input-integrity.json'):
    p=remote/file
    if p.exists():check(json.loads(p.read_text())['passed'],'post_input_integrity',file=file)
controller=json.loads((remote/'run/supervision/controller-result.json').read_text());workerexit=json.loads((remote/'run/supervision/worker-exit.json').read_text())
check(controller['status']=='completed' and controller['released'] and workerexit['returncode']==0,'supervisor_worker_exit_release')
check((remote/'controller.os-exit').read_text().strip()=='0' and (base/'ssh.os-exit').read_text().strip()=='0','actual_controller_ssh_exit')
for name in ('independent-release.json','independent-owned-release.json'):check(json.loads((remote/name).read_text())['passed'],'independent_release',file=name)
result=dict(status='passed',scope='CPU raw artifact audit; no GPU execution',torch=str(torch.__version__),seconds=time.monotonic()-start,checks=len(checks),numerical_comparisons=len(numerical),all_numerical_bit_equal=all(r['bit_equal'] for r in numerical),all_torch_equal=all(r['bit_equal'] for r in numerical),all_storage_byte_equal=all(r['storage_byte_equal'] for r in numerical),byte_equality_is_diagnostic_not_a_new_pass_threshold=True,max_abs=max(r['max_abs'] for r in numerical),max_rms=max(r['rms'] for r in numerical),graph_memory=memory,controller_elapsed_seconds=controller['elapsed_seconds'],raw_tensor_artifacts=len(files),artifact_sha256=files,structural_checks=checks,numerical=numerical,limitations=['Same-backend graph fidelity, not a new independent attention oracle','Prior whole-Qwen RMS failure unchanged','Wall time is not a benchmark or performance claim'])
(args.output or base/'raw-cpu-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ('status','seconds','checks','numerical_comparisons','all_numerical_bit_equal','all_storage_byte_equal','max_abs','max_rms','raw_tensor_artifacts')}))
