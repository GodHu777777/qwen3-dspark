"""Recheck a completed private STS run: python verify_sts.py PRIVATE_RUN_DIRECTORY.

No GPU imports. Expects the immutable source/ archive plus fit/ and eval/ evidence.
Prints only scalar checks/hashes, never prompt IDs, tokens, logits or private paths.
"""
import hashlib,json,math,sys,tarfile
from pathlib import Path
BASE=Path(sys.argv[1])
def read(path):return json.loads(path.read_text())
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def close(x,y):
 if x is None or y is None:assert x is y
 else:assert abs(x-y)<=1e-12,(x,y)
archive_sha='3d4ebda50a24c58175e6ff96428663155533f1ea3b419ee5f17eae02b59e0dda'
assert sha(BASE/'source.tar')==archive_sha
expected=set()
with tarfile.open(BASE/'source.tar') as archive:
 for member in archive.getmembers():
  if member.isfile():
   expected.add(member.name)
   assert hashlib.sha256((BASE/'source'/member.name).read_bytes()).digest()==hashlib.sha256(archive.extractfile(member).read()).digest()
assert {str(p.relative_to(BASE/'source')) for p in (BASE/'source').rglob('*') if p.is_file()}==expected
sys.path.insert(0,str(BASE/'source'))
from dspark_qwen.calibrate_rollout import load_collection, implementation_identity
from dspark_qwen.calibration import DEFAULT_TEMPERATURE_GRID
from dspark_qwen.rollout_protocol import digest
artifact=read(BASE/'fit/sts-fit.private.json');evaluation=read(BASE/'eval/sts-eval.private.json')
assert artifact['artifact_sha256']==digest({k:v for k,v in artifact.items() if k!='artifact_sha256'})
assert evaluation['report_sha256']==digest({k:v for k,v in evaluation.items() if k!='report_sha256'})
assert artifact['implementation_sha256']==implementation_identity()
assert artifact['fit']['num_bins']==evaluation['num_bins']==20
assert artifact['fit']['temperature_grid']==list(DEFAULT_TEMPERATURE_GRID)
assert all(t in DEFAULT_TEMPERATURE_GRID for t in artifact['fit']['temperatures'])
assert evaluation['artifact_sha256']==artifact['artifact_sha256']
fit=load_collection(BASE/'fit/collection','fit');ev=load_collection(BASE/'eval/collection','eval')
assert artifact['identity']==fit['identity']==ev['identity']==evaluation['identity']
assert artifact['fit_evidence']==fit['evidence']==evaluation['fit_evidence']
assert evaluation['eval_evidence']==ev['evidence']
for key in ('prompt_ids','prompt_sha256'):assert not set(fit['evidence'][key]) & set(ev['evidence'][key])
for group,collection,n in [('fit',fit,44),('eval',ev,43)]:
 root=BASE/group
 assert (root/'controller.os-exit-code').read_text().strip()=='0'
 assert read(root/'controller-exit.json')['returncode']==0 and not read(root/'controller-exit.json')['timed_out']
 assert read(root/'launcher-exit.json')['returncode']==0 and not read(root/'launcher-exit.json')['timed_out']
 assert read(root/'collection/worker-exit.json')['returncode']==0
 assert read(root/'deployment.json')==read(root/'post-input-integrity.json')
 assert read(root/'binding-dryrun/run.json')==read(root/'collection/run.json')
 assert collection['evidence']['completed_prompts']==n
 release=read(root/'postrelease.json')
 assert not any(release['owned_pid_exists'].values())
 assert {int(x) for x in release['kfd']['stdout'].split()}=={1208354}
 health=json.loads(release['asr']['stdout']);assert health['ready'] and not health['busy']
freeze=read(BASE/'fit/fit-freeze.json')
assert freeze['artifact_file_sha256']==sha(BASE/'fit/sts-fit.private.json')
assert freeze['artifact_sha256']==artifact['artifact_sha256']
assert freeze['cpu_fit_exit_code']==0
assert freeze['frozen_epoch']<read(BASE/'eval/launch.json')['utc']
assert read(BASE/'eval/deployment.json')['frozen_fit_artifact_file_sha256']==freeze['artifact_file_sha256']
assert (BASE/'fit/cpu-fit.exit-code').read_text().strip()=='0'
assert (BASE/'eval/cpu-eval.exit-code').read_text().strip()=='0'

def sigmoid(x):
 if x>=0:return 1/(1+math.exp(-x))
 z=math.exp(x);return z/(1+z)
def independent_metrics(pred,labels):
 if not labels:return dict(count=0,ece=None,brier=None,pred_mean=None,target_mean=None)
 pred=[max(1e-8,min(1-1e-8,p)) for p in pred]
 bins=[[] for _ in range(20)]
 for p,y in zip(pred,labels):bins[min(19,int(p*20))].append((p,y))
 ece=math.fsum(abs(math.fsum(p for p,_ in b)-math.fsum(y for _,y in b)) for b in bins if b)/len(labels)
 return dict(count=len(labels),ece=ece,brier=math.fsum((p-y)**2 for p,y in zip(pred,labels))/len(labels),pred_mean=math.fsum(pred)/len(labels),target_mean=math.fsum(labels)/len(labels))
positions=[]
for j in range(fit['block_size']):
 f=[b for b in fit['blocks'] if b.effective_length>j];e=[b for b in ev['blocks'] if b.effective_length>j]
 events=sum(j<b.accepted_prefix_length for b in f)
 constant=events/len(f) if f else None
 close(constant,artifact['fit_prefix_prevalence'][j])
 for population,rows,stored in [('fit',f,artifact['fit_metrics'][j]),('eval',e,evaluation['per_position'][j])]:
  labels=[int(j<b.accepted_prefix_length) for b in rows]
  for method,temperatures in [('unscaled',[1]*fit['block_size']),('sts',artifact['fit']['temperatures'])]:
   predictions=[math.prod(sigmoid(z/t) for z,t in zip(b.confidence_logits[:j+1],temperatures)) for b in rows]
   metrics=independent_metrics(predictions,labels)
   for key,value in metrics.items():close(value,stored[method][key])
  if constant is not None:
   metrics=independent_metrics([constant]*len(labels),labels)
   for key,value in metrics.items():close(value,stored['fit_prevalence_constant'][key])
  else:assert stored['fit_prevalence_constant']['available'] is False and stored['fit_prevalence_constant']['ece'] is None
  assert stored['observed_label_count']==len(rows)
  close(stored['prompt_coverage'],len({b.prompt_id for b in rows})/(44 if population=='fit' else 43))
 positions.append(dict(position=j,fit_count=len(f),fit_prefix_events=events,eval_count=len(e),eval_prefix_events=sum(j<b.accepted_prefix_length for b in e)))
summary=dict(passed=True,source_commit='034064bf8fea7f67c9039c2dd103b65ca12e803a',archive_sha256=archive_sha,source_file_count=len(expected),checkpoint_sha256=artifact['identity']['checkpoint_sha256'],selection_sha256=artifact['identity']['selection']['selection_sha256'],fit_artifact_file_sha256=sha(BASE/'fit/sts-fit.private.json'),fit_artifact_sha256=artifact['artifact_sha256'],eval_report_file_sha256=sha(BASE/'eval/sts-eval.private.json'),eval_report_sha256=evaluation['report_sha256'],all_input_hashes_unchanged=True,all_os_exits_zero=True,all_owned_pids_gone=True,prompt_disjoint=True,fit_frozen_before_eval_launch=True,default_61_temperature_grid=True,num_bins=20,independent_ece_brier_prevalence_recheck=True,position_evidence=positions,verification_script_sha256=sha(Path(__file__)))
print(json.dumps(summary,indent=2,allow_nan=False))
