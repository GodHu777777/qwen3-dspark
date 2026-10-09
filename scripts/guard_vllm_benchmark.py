#!/usr/bin/env python3
"""Reuse the identity-aware supervisor for one bounded benchmark engine lifetime."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dspark_qwen.performance_workloads import load_workloads, schedule, verify_complete_samples
from guard_vllm_smoke import ASRGuard, sha, supervise, verify_binding, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binding', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    binding = json.loads(Path(args.binding).read_text())
    verify_binding(binding)
    if str(Path(__file__).resolve()) not in binding['files']:
        raise ValueError('Benchmark controller must be in frozen source binding')
    manifest, workload_sha = load_workloads(binding['request'])
    if not args.execute:
        print(json.dumps(dict(status='cpu_prepared_only',workload_sha256=workload_sha,
            samples=sum(1 for _ in schedule(manifest)),worker_deadline_seconds=600)))
        return 0
    output = Path(args.output)
    command = [binding['python'],'-u',binding['worker'],'--model',binding['model'],
        '--workload',binding['request'],'--output',str(output/'worker'),'--execute']
    env = dict(os.environ,OMP_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONPATH',None)
    for name in ('CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'):
        env.pop(name,None)
    result = supervise(command,output,ASRGuard(binding['asr_pid'],binding['asr_url']),env=env)
    try:
        verify_binding(binding)
        write(output/'post-input-integrity.json',dict(passed=True,binding_sha256=sha(args.binding),workload_sha256=workload_sha))
    except BaseException:
        (output/'post-integrity-error.txt').write_text(traceback.format_exc())
        return 1
    if result['status'] != 'completed' or not result['released']:
        return 1
    worker = json.loads((output/'worker/result.json').read_text())
    samples = [json.loads(line) for line in (output/'worker/samples.jsonl').read_text().splitlines()]
    recomputed = verify_complete_samples(samples,manifest)
    if recomputed != worker['aggregates']:
        raise ValueError('Worker aggregate differs from complete raw sample recomputation')
    return 0 if (worker['status']=='completed' and worker['workload_sha256']==workload_sha and
        worker['sample_count']==sum(1 for _ in schedule(manifest)) and all(a['complete'] for a in worker['aggregates'])) else 1


if __name__ == '__main__':
    raise SystemExit(main())
