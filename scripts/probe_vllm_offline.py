#!/usr/bin/env python3
"""One bounded externally supervised vLLM init/generate smoke, not a benchmark.

Default mode validates inputs and emits the exact intended configuration without
importing vLLM/torch. --execute requires an independently guarded GPU window.
The outer controller owns deadline, process group/tree cleanup and ASR checks.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

EXPECTED_VLLM_VERSION = '0.30.0+rocm723'


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def prepare(model, request_path):
    model = Path(model).resolve()
    if not model.is_dir() or not (model / 'config.json').is_file():
        raise ValueError('Existing local model with config.json required')
    raw = Path(request_path).read_bytes()
    request = json.loads(raw)
    if set(request) != {'prompt_token_ids', 'seed'}:
        raise ValueError('Request must contain only frozen prompt_token_ids and seed')
    tokens, seed = request['prompt_token_ids'], request['seed']
    config = json.loads((model / 'config.json').read_text())
    vocab_size = config['vocab_size']
    if type(vocab_size) is not int or vocab_size < 1:
        raise ValueError('Invalid model vocabulary')
    if not isinstance(tokens, list) or not 1 <= len(tokens) <= 3968 or any(type(t) is not int or not 0 <= t < vocab_size for t in tokens):
        raise ValueError('Frozen nonempty in-vocabulary prompt must fit 4096 with 128 output tokens')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Explicit uint32 seed required')
    engine = dict(model=str(model), dtype='bfloat16', tensor_parallel_size=1,
        generation_config='vllm', max_model_len=4096, max_num_seqs=1,
        max_num_batched_tokens=4096, gpu_memory_utilization=.18,
        enable_prefix_caching=False, enforce_eager=False, disable_log_stats=True, seed=seed)
    sampling = dict(n=1, temperature=1., top_p=1., top_k=0, min_p=0., max_tokens=128,
        ignore_eos=True, repetition_penalty=1., presence_penalty=0., frequency_penalty=0.,
        detokenize=False, logprobs=None, prompt_logprobs=None, seed=seed)
    return tokens, engine, sampling, dict(request_sha256=digest_bytes(raw),
        model_config_sha256=digest_bytes((model / 'config.json').read_bytes()),
        prompt_tokens=len(tokens), output_tokens_required=128)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--request', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    tokens, engine, sampling, identity = prepare(args.model, args.request)
    output = Path(args.output)
    if output.exists():
        raise ValueError('Output must be a new directory; no overwrite/retry')
    output.mkdir(parents=True)
    specification = dict(schema='vllm-offline-smoke-v1', engine=engine, sampling=sampling,
        identity=identity, performance_claim=False, execute=args.execute, expected_vllm_version=EXPECTED_VLLM_VERSION,
        runner_sha256=digest_bytes(Path(__file__).read_bytes()))
    (output / 'specification.json').write_text(json.dumps(specification, indent=2) + '\n')
    if not args.execute:
        print(json.dumps(dict(status='cpu_preflight_only', **specification), sort_keys=True))
        return
    started = time.monotonic()

    def stage(name, **fields):
        record = dict(stage=name, elapsed_seconds=time.monotonic()-started, **fields)
        with (output / 'stages.jsonl').open('a') as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
        print(json.dumps(record, sort_keys=True), flush=True)

    try:
        stage('import_started')
        import vllm
        from vllm import LLM, SamplingParams
        if vllm.__version__ != EXPECTED_VLLM_VERSION:
            raise RuntimeError('Installed vLLM version differs from frozen inventory')
        stage('import_completed', vllm_version=vllm.__version__)
        stage('engine_initialization_started')
        llm = LLM(**engine)
        actual = llm.llm_engine.vllm_config
        if actual.model_config.enforce_eager:
            raise RuntimeError('Unexpected eager engine; do not reinterpret as default-graph success')
        stage('engine_initialization_completed', enforce_eager=actual.model_config.enforce_eager,
            configured_cudagraph_mode=str(actual.compilation_config.cudagraph_mode))
        parameters = SamplingParams(**sampling)
        stage('single_request_generation_started')
        results = llm.generate([{'prompt_token_ids': tokens}], parameters, use_tqdm=False)
        if len(results) != 1 or len(results[0].outputs) != 1:
            raise RuntimeError('Expected exactly one request and one sequence')
        item = results[0].outputs[0]
        generated = list(item.token_ids)
        if len(generated) != 128:
            raise RuntimeError(f'Expected fixed 128-token work; got {len(generated)}')
        if results[0].prompt_token_ids != tokens:
            raise RuntimeError('Engine prompt token IDs differ from frozen input')
        stage('single_request_generation_completed', output_tokens=len(generated),
            output_token_sha256=digest_bytes(json.dumps(generated, separators=(',', ':')).encode()),
            finish_reason=item.finish_reason, stop_reason=item.stop_reason)
        (output / 'result.json').write_text(json.dumps(dict(status='completed', identity=identity,
            output_tokens=len(generated), configured_cudagraph_mode=str(actual.compilation_config.cudagraph_mode),
            graph_replay_independently_verified=False, performance_claim=False), indent=2) + '\n')
        stage('worker_completed')
    except BaseException as error:
        stage('worker_failed', error_type=type(error).__name__, error=str(error))
        (output / 'failure.txt').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
