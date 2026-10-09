#!/usr/bin/env python3
"""One-engine fixed-batch vLLM measurement, default CPU preflight only."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dspark_qwen.performance_workloads import load_workloads, schedule, sha_json, summarize, expected_sample_identities, verify_complete_samples


def sampling_kwargs(manifest, request):
    return dict(manifest['sampling'], n=1, max_tokens=manifest['output_tokens'],
        seed=request['seed'], detokenize=False, logprobs=None, prompt_logprobs=None)


def primary_batch(llm, case, manifest, sampling_class, *, clock=time.perf_counter):
    prompts = [{'prompt_token_ids': r['prompt_token_ids']} for r in case['requests']]
    params = [sampling_class(**sampling_kwargs(manifest, r)) for r in case['requests']]
    if llm.llm_engine.has_unfinished_requests():
        raise ValueError('Fresh empty request state required')
    started = clock()
    outputs = llm.generate(prompts, params, use_tqdm=False)
    elapsed = clock() - started
    # Completion is the synchronous offline API boundary. No per-token observer,
    # profiler, logging, hashing or independent model replay inside this region.
    if len(outputs) != len(prompts):
        raise ValueError('Incomplete batch')
    records = []
    for request, output in zip(case['requests'], outputs):
        if output.prompt_token_ids != request['prompt_token_ids'] or not output.finished or len(output.outputs) != 1:
            raise ValueError('Output/request identity or completion mismatch')
        sequence = output.outputs[0]
        tokens = list(sequence.token_ids)
        if len(tokens) != manifest['output_tokens'] or sequence.finish_reason != 'length':
            raise ValueError('Fixed output workload incomplete')
        records.append(dict(request=request['request'], output_tokens=len(tokens), output_sha256=sha_json(tokens),
            completion_latency_seconds=None, ttft_seconds=None,
            latency_status='unsupported_in_primary_statsoff_generate_api'))
    return dict(batch_wall_seconds=elapsed, output_tokens=sum(r['output_tokens'] for r in records), requests=records,
        time_scope='synchronous LLM.generate entry to return, including submission/prefill/decode/completion',
        observed_scheduler_B=None, scheduler_shape_status='not_instrumented; ordinary decode B is generally active R, not speculative R+sum(ell)')


def diagnostic_batch(llm, case, manifest, sampling_class, delta_kind, *, tag, clock=time.perf_counter):
    """Separate pass: host-observed output events include observer/queue costs."""
    engine = llm.llm_engine
    if engine.has_unfinished_requests():
        raise ValueError('Fresh empty request state required')
    params = [sampling_class(**sampling_kwargs(manifest,r), output_kind=delta_kind) for r in case['requests']]
    states = {}
    started = clock()
    for request, parameter in zip(case['requests'], params):
        identity = f'{tag}-{request["request"]}'
        submitted = clock()
        engine.add_request(identity, {'prompt_token_ids': request['prompt_token_ids']}, parameter)
        states[identity] = dict(request=request['request'], submitted=submitted, first=None, finished=None,
            tokens=[], prompt=request['prompt_token_ids'])
    while engine.has_unfinished_requests():
        outputs = engine.step()
        if not outputs:
            continue
        observed = clock()
        for output in outputs:
            if output.request_id not in states or len(output.outputs) != 1:
                raise ValueError('Unexpected diagnostic request')
            state = states[output.request_id]
            if output.prompt_token_ids != state['prompt'] or state['finished'] is not None:
                raise ValueError('Diagnostic request identity/completion mismatch')
            sequence = output.outputs[0]
            tokens = list(sequence.token_ids)
            if tokens and state['first'] is None:
                state['first'] = observed
            state['tokens'].extend(tokens)
            if output.finished:
                if sequence.finish_reason != 'length':
                    raise ValueError('Diagnostic workload stopped early')
                state['finished'] = observed
    elapsed = clock() - started
    records = []
    for state in states.values():
        if state['first'] is None or state['finished'] is None or len(state['tokens']) != manifest['output_tokens']:
            raise ValueError('Incomplete diagnostic request')
        records.append(dict(request=state['request'], output_tokens=len(state['tokens']),
            output_sha256=sha_json(state['tokens']), submitted_offset_seconds=state['submitted']-started,
            ttft_seconds=state['first']-state['submitted'],
            completion_latency_seconds=state['finished']-state['submitted'],
            post_first_to_finish_seconds=state['finished']-state['first'],
            first_output_note='Host-observed first nonempty output chunk, not isolated prefill kernel time',
            latency_scope='immediately before add_request to host observation; includes submission, queuing and observer overhead'))
    return dict(batch_wall_seconds=elapsed, output_tokens=sum(r['output_tokens'] for r in records), requests=records,
        time_scope='separate diagnostic add_request/step event loop; never substituted for primary timing',
        pure_prefill_seconds=None, pure_prefill_status='unsupported_without_additional_engine_instrumentation')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--workload', '--request', dest='workload', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    manifest, manifest_sha = load_workloads(args.workload)
    model = Path(args.model).resolve()
    if hashlib.sha256((model/'config.json').read_bytes()).hexdigest() != manifest['model']['config_sha256']:
        raise ValueError('Model config differs from frozen workload identity')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    engine_kwargs = dict(model=str(model), dtype='bfloat16', tensor_parallel_size=1,
        generation_config='vllm', max_model_len=4096, max_num_seqs=4,
        max_num_batched_tokens=4096, gpu_memory_utilization=.18,
        enable_prefix_caching=False, enforce_eager=False, disable_log_stats=True, seed=20261009)
    specification = dict(schema='vllm-offline-batch-benchmark-v1', workload_sha256=manifest_sha,
        engine=engine_kwargs, protocol=manifest['measurement'], expected_sample_count=sum(1 for _ in schedule(manifest)), expected_samples=expected_sample_identities(manifest),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), execute=args.execute,
        end_to_end_baseline=True, speculative_verification_sps=False)
    (output/'specification.json').write_text(json.dumps(specification,indent=2)+'\n')
    if not args.execute:
        print(json.dumps(dict(status='cpu_preflight_only', **specification),sort_keys=True))
        return
    start = time.perf_counter(); samples = []
    def event(stage, **fields):
        with (output/'stages.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(stage=stage, elapsed_seconds=time.perf_counter()-start, **fields),sort_keys=True)+'\n')
    try:
        event('import_started')
        import vllm
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import RequestOutputKind
        versions = dict(distribution_actual=importlib.metadata.version('vllm'), distribution_expected='0.30.0+rocm723',
            module_actual=vllm.__version__, module_expected='0.30.0')
        event('version_identity_checked', **versions)
        if versions['distribution_actual'] != versions['distribution_expected'] or versions['module_actual'] != versions['module_expected']:
            raise ValueError('Separately pinned vLLM versions mismatch')
        event('engine_initialization_started')
        llm = LLM(**engine_kwargs)  # Exactly one engine for every warmup/sample.
        actual = llm.llm_engine.vllm_config
        if actual.model_config.enforce_eager:
            raise ValueError('Unexpected eager mode')
        event('engine_initialization_completed', configured_cudagraph_mode=str(actual.compilation_config.cudagraph_mode))
        for phase, repeat, case in schedule(manifest):
            event('sample_started', phase=phase, repeat=repeat, case_id=case['case_id'])
            if phase == 'diagnostic':
                sample = diagnostic_batch(llm,case,manifest,SamplingParams,RequestOutputKind.DELTA,
                    tag=f'{phase}-{repeat}-{case["case_id"]}')
            else:
                sample = primary_batch(llm,case,manifest,SamplingParams)
            sample.update(phase=phase,repeat=repeat,case_id=case['case_id'],
                request_count=case['request_count'],prompt_length=case['prompt_length'])
            samples.append(sample)
            with (output/'samples.jsonl').open('a') as stream:
                stream.write(json.dumps(sample,sort_keys=True,allow_nan=False)+'\n')
            event('sample_completed',phase=phase,repeat=repeat,case_id=case['case_id'])
        aggregates = verify_complete_samples(samples,manifest)
        if not all(a['complete'] for a in aggregates):
            raise ValueError('Incomplete primary measurement domain')
        (output/'result.json').write_text(json.dumps(dict(status='completed',schema=specification['schema'],
            workload_sha256=manifest_sha,sample_count=len(samples),aggregates=aggregates,
            setup_timing_location='stages.jsonl; outside batch timings',
            per_request_latency_pass='diagnostic_only',arrival_load_frontier=False,
            speculative_verification_sps=False),indent=2)+'\n')
        event('worker_completed')
    except BaseException as error:
        event('worker_failed',error_type=type(error).__name__,error=str(error),completed_samples=len(samples))
        (output/'failure.txt').write_text(traceback.format_exc())
        (output/'partial-aggregates.json').write_text(json.dumps(summarize(samples,manifest),indent=2)+'\n')
        raise


if __name__ == '__main__':
    main()
