"""Bounded native BF16 stochastic gate; dry-run is stdlib-only and GPU-free.

Original pilot train + first two dev cases are bound through the prior private
gate. Every execution launches a worker from an immutable local source copy.
Execution/reproducibility and same-prefix numerical fidelity are separate results.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import shutil
import subprocess
import sys
import time
import traceback


POLICY = "float64_softmax_normalize_cdf_v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def append_json(path, value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value, allow_nan=False) + '\n')
        stream.flush()


def emit(out, value):
    line = json.dumps(value, allow_nan=False)
    if os.environ.get('DSPARK_GATE_CAPTURED_STDOUT') != '1':
        with (out / 'stdout.log').open('a') as stream:
            stream.write(line + '\n')
            stream.flush()
    print(line, flush=True)


def probability_protocol(args):
    if not 1 <= args.max_new_tokens <= 32:
        raise ValueError('Token budget must lie in [1,32] for this bounded gate')
    if (not math.isfinite(args.temperature) or args.temperature <= 0 or
            not 0 <= args.probe_rounds <= 2 or not 1 <= len(args.seeds) <= 2 or
            len(set(args.seeds)) != len(args.seeds) or
            any(not 0 <= seed < 2**63 for seed in args.seeds)):
        raise ValueError('Require positive finite temperature, 0..2 probes, and 1..2 distinct 63-bit seeds')
    if not 1 <= args.timeout_seconds <= 1800:
        raise ValueError('Worker timeout must lie in [1,1800] seconds')
    return dict(probability_policy=POLICY, temperature=args.temperature, seeds=args.seeds,
        max_new_tokens=args.max_new_tokens, probe_rounds=args.probe_rounds,
        target_forward_dtype='bfloat16', draft_parameters='float32', draft_autocast='bfloat16',
        backend='sdpa', filtering='none', admission='fixed block_size capped by remaining budget',
        phases=['reference', 'immediate_repeat', 'interleaved_reverse'],
        numerical_probe_selection='first configured rounds of reference phase; all n+1 target rows',
        max_process_memory_gib=6, minimum_free_memory_gib=8,
        cross_algorithm_token_equality='not asserted', timing='not a performance benchmark')


def bind_inputs(args):
    """Hash inputs without importing torch, querying CUDA or loading model weights."""
    protocol = probability_protocol(args)
    prior_path = Path(args.prior_gate).expanduser().resolve()
    prior = json.loads(prior_path.read_text())
    checkpoint = Path(args.checkpoint or prior['identity']['args']['checkpoint']).expanduser().resolve()
    metadata_path = checkpoint / 'metadata.json'
    metadata = json.loads(metadata_path.read_text())
    cfg = metadata['identity']['config']
    model = Path(cfg['model']).expanduser().resolve()
    records_path = Path(cfg['records']).expanduser().resolve()
    weights_hash = sha256(checkpoint / 'draft.safetensors')
    metadata_hash = sha256(metadata_path)
    if weights_hash != metadata['draft_weights_sha256']:
        raise ValueError('Checkpoint weights differ from metadata')
    if (weights_hash != prior['identity']['checkpoint_weights_sha256'] or
            metadata_hash != prior['identity']['checkpoint_metadata_sha256']):
        raise ValueError('Gate must use the original prior checkpoint identity')
    records_hash = sha256(records_path)
    if records_hash != metadata['identity']['records_sha256']:
        raise ValueError('Checkpoint development-record identity mismatch')
    fingerprint = metadata['identity']['target_fingerprint']
    if not fingerprint or 'config.json' not in fingerprint or not any(
            name.endswith('.safetensors') for name in fingerprint):
        raise ValueError('Target fingerprint must include model config and weights')
    for name, digest in fingerprint.items():
        if Path(name).is_absolute() or '..' in Path(name).parts or sha256(model / name) != digest:
            raise ValueError(f'Target fingerprint mismatch or invalid path: {name}')
    config = json.loads((model / 'config.json').read_text())
    if config.get('model_type') != 'qwen3' or config.get('use_sliding_window', False):
        raise ValueError('Only the dense Qwen3 target is supported')
    records = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
    if (any(r['split'] not in ('train', 'validation') for r in records) or
            len({r['id'] for r in records}) != len(records)):
        raise ValueError('Final-test/unknown split or duplicate development identity')
    train = [r for r in records if r['id'] == metadata.get('record_id') and r['split'] == 'train' and r['accepted']]
    dev = [r for r in records if r['split'] == 'validation' and r['accepted']][:2]
    rows = train + dev
    if len(train) != 1 or len(dev) != 2 or [r['id'] for r in rows] != [r['id'] for r in prior['examples']]:
        raise ValueError('Require all original three cases: training record and first two accepted dev records')
    if [r['split'] for r in prior['examples']] != ['train', 'validation', 'validation']:
        raise ValueError('Prior case splits changed')
    for row, original in zip(rows, prior['examples']):
        tokens = row['prompt_token_ids']
        if (not tokens or len(tokens) > 4096 or len(tokens) != original['prompt_tokens'] or
                any(type(token) is not int or not 0 <= token < config['vocab_size'] for token in tokens)):
            raise ValueError('Invalid or changed original prompt-token sequence')
    block_size = metadata['draft_config']['block_size']
    if type(block_size) is not int or not 1 <= block_size <= 16:
        raise ValueError('Bounded gate requires block_size in [1,16]')
    return dict(checkpoint=str(checkpoint), checkpoint_weights_sha256=weights_hash,
        checkpoint_metadata_sha256=metadata_hash, model=str(model), records=str(records_path),
        records_sha256=records_hash, target_fingerprint=fingerprint,
        prior_gate=str(prior_path), prior_gate_sha256=sha256(prior_path),
        draft_config=metadata['draft_config'], protocol=protocol,
        cases=[dict(index=i, id=r['id'], split=r['split'], prompt_token_ids=r['prompt_token_ids'],
                    prompt_sha256=hashlib.sha256(json.dumps(r['prompt_token_ids']).encode()).hexdigest())
               for i, r in enumerate(rows)])


def snapshot_source(out):
    source = Path(__file__).resolve().parent
    destination = out / 'source' / 'dspark_qwen'
    destination.mkdir(parents=True)
    hashes = {}
    for path in sorted(source.glob('*.py')):
        before = sha256(path)
        shutil.copyfile(path, destination / path.name)
        if before != sha256(path) or before != sha256(destination / path.name):
            raise ValueError('Source changed while snapshotting')
        hashes[path.name] = before
    return hashes


def verify_binding(binding, source):
    """Worker checks before touching CUDA; source copy is the executable identity."""
    unsigned = {key: value for key, value in binding.items() if key != 'binding_sha256'}
    if hashlib.sha256(json.dumps(unsigned, sort_keys=True).encode()).hexdigest() != binding['binding_sha256']:
        raise ValueError('Run binding manifest changed')
    if {p.name for p in source.glob('*.py')} != set(binding['source_sha256']):
        raise ValueError('Source snapshot file inventory changed')
    paths = [(Path(binding['checkpoint']) / 'draft.safetensors', binding['checkpoint_weights_sha256']),
             (Path(binding['checkpoint']) / 'metadata.json', binding['checkpoint_metadata_sha256']),
             (Path(binding['records']), binding['records_sha256']),
             (Path(binding['prior_gate']), binding['prior_gate_sha256'])]
    paths += [(Path(binding['model']) / name, digest) for name, digest in binding['target_fingerprint'].items()]
    paths += [(source / name, digest) for name, digest in binding['source_sha256'].items()]
    for path, digest in paths:
        if sha256(path) != digest:
            raise ValueError(f'Bound input changed: {path.name}')


def aggregate(report):
    probes = report.get('numerical_probes', [])
    checks = report.get('checks', [])
    examples = []
    for run in report.get('runs', []):
        if run['phase'] != 'reference':
            continue
        matched = [p for p in probes if p['case'] == run['case'] and p['seed'] == run['seed']]
        repeated = [c for c in checks if c['case'] == run['case'] and c['seed'] == run['seed']]
        examples.append(dict(case=run['case'], split=run['split'], seed=run['seed'],
            baseline_output_count=len(run['baseline_tokens']), speculative_output_count=len(run['speculative_tokens']),
            rounds=len(run['rounds']), committed_draft_tokens=sum(r['accepted'] for r in run['rounds']),
            completed_repeat_checks=len(repeated), completed_repeat_checks_passed=all(c['passed'] for c in repeated) if repeated else None,
            probed_rows=len(matched), max_total_variation=max((p['total_variation'] for p in matched), default=None)))
    return dict(status=report['status'], binding_sha256=report.get('binding_sha256'),
        execution_checks_passed=report.get('execution_checks_passed'),
        examples=examples,
        completed_runs=len(report.get('runs', [])), completed_reproducibility_checks=len(checks),
        all_completed_checks_passed=all(c['passed'] for c in checks) if checks else None,
        error_type=report.get('error_type'),
        native_numerical_fidelity=dict(status='measured' if probes else 'not_measured',
            compared_rows=len(probes), max_total_variation=max((p['total_variation'] for p in probes), default=None),
            max_absolute_logit_difference=max((p['max_absolute_logit_difference'] for p in probes), default=None),
            nonzero_tv_rows=sum(p['total_variation'] > 0 for p in probes),
            argmax_changed_rows=sum(not p['argmax_equal'] for p in probes),
            interpretation='Measured same-prefix path differences only; no distribution-losslessness verdict'),
        scope='Bounded execution/reproduction gate, not a speed benchmark or proof of lossless model sampling')


def save_report(out, report):
    write_json(out / 'result.json', report)
    write_json(out / 'aggregate.json', aggregate(report))


def audited_adapters(model, draft):
    """Check actual cache/projected chunks; no tensors retained across requests."""
    import torch
    from .cached_target import CachedTarget

    class AuditedTarget(CachedTarget):
        def reset(self):
            self.last_prediction = None
            super().reset()
        def audit_cache(self):
            for layer in self.cache.layers:
                if (layer.keys.shape[0] != 1 or layer.keys.shape[2] != self.length or
                        layer.values.shape != layer.keys.shape or
                        not bool(torch.isfinite(layer.keys).all()) or
                        not bool(torch.isfinite(layer.values).all())):
                    raise RuntimeError('Nonfinite or misshaped target cache')
        def append(self, ids):
            result = super().append(ids)
            self.audit_cache()
            return result
        def crop(self, length):
            super().crop(length)
            self.audit_cache()
        def predict(self, features, last_only=False):
            logits = super().predict(features, last_only)
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError('Nonfinite target logits')
            self.last_prediction = logits.detach()
            return logits

    class AuditedDraft:
        spec = draft.spec
        def __init__(self): self.projected_end = 0
        def eval(self):
            draft.eval()
            return self
        def project_context_kv(self, features, start):
            if start == 0:
                self.projected_end = 0
            if start != self.projected_end or not bool(torch.isfinite(features).all()):
                raise RuntimeError('Invalid draft feature commit boundary')
            pairs = draft.project_context_kv(features, start)
            if len(pairs) != len(draft.layers):
                raise RuntimeError('Wrong number of projected draft KV layers')
            for layer, (key, value) in zip(draft.layers, pairs):
                expected = (1, layer.attention.kv_heads, features.shape[1], layer.attention.dim)
                if (key.shape != expected or value.shape != expected or
                        not bool(torch.isfinite(key).all()) or not bool(torch.isfinite(value).all())):
                    raise RuntimeError('Nonfinite or misshaped projected draft KV chunk')
            self.projected_end += features.shape[1]
            return pairs
        def propose_stochastic_cached(self, *args, **kwargs):
            proposal = draft.propose_stochastic_cached(*args, **kwargs)
            if proposal.tokens.numel() != kwargs['max_draft_tokens']:
                raise RuntimeError('Fixed admission returned an unexpected proposal length')
            return proposal
    return AuditedTarget(model, ()), AuditedTarget(model, draft.spec.layer_ids), AuditedDraft()


def check_output(tokens, budget, eos, prompt_length, cache_length):
    if not 1 <= len(tokens) <= budget or any(token in eos for token in tokens[:-1]):
        raise RuntimeError('EOS/output budget violation')
    if len(tokens) < budget and tokens[-1] not in eos:
        raise RuntimeError('Output stopped before budget without EOS')
    if cache_length != prompt_length + len(tokens) - 1:
        raise RuntimeError('Final cache does not exclude latest emitted anchor')


def audit_observation(observation, expected_before, projected_end, vocabulary):
    import torch
    from .tensor_sampling import check_probabilities
    proposal, p, result = observation.proposal, observation.target_probs, observation.result
    n = proposal.tokens.numel()
    check_probabilities(proposal.draft_probs)
    check_probabilities(p)
    if (p.shape != (n + 1, vocabulary) or proposal.draft_probs.shape != (n, vocabulary) or
            proposal.confidence_logits.shape != (n,) or not bool(torch.isfinite(proposal.confidence_logits).all()) or
            observation.cache_before != expected_before or observation.cache_after != projected_end or
            observation.cache_after != expected_before + result.tokens.numel()):
        raise RuntimeError('Round shape/finite/commit invariant failed')
    accepted = result.accepted_draft_tokens
    if not torch.equal(result.tokens[:accepted], proposal.tokens[:accepted]):
        raise RuntimeError('Committed accepted prefix differs from proposal')
    if result.rejected_index is not None:
        j, replacement = result.rejected_index, result.tokens[-1]
        if j != accepted or not bool(p[j, replacement] > proposal.draft_probs[j, replacement]):
            raise RuntimeError('Replacement lies outside positive residual support')
    return dict(proposal_tokens=proposal.tokens.tolist(), committed_tokens=result.tokens.tolist(),
        selected_q=proposal.draft_probs.gather(1, proposal.tokens[:, None]).flatten().tolist(),
        selected_p=p[:-1].gather(1, proposal.tokens[:, None]).flatten().tolist(),
        confidence_logits=proposal.confidence_logits.tolist(), accepted=accepted,
        rejected_index=result.rejected_index, extra_token_kind=result.extra_token_kind,
        stop_reason=result.stop_reason, cache_before=observation.cache_before, cache_after=observation.cache_after,
        verified_proposal_length=observation.verified_proposal_length,
        accepted_eos_position=observation.accepted_eos_position, finite_and_shape_checked=True)


def compare_probe(model, ids, output, emitted, payload, temperature, on_row):
    """Fresh sequential history on exactly the sampled proposal semantic prefix."""
    import torch
    from .cached_target import CachedTarget
    from .tensor_sampling import logits_to_probabilities
    sequential = CachedTarget(model, ())
    features = sequential.prefill(ids)
    for token in output[:emitted]:
        features = sequential.append(ids.new_tensor([[token]]))
    proposal = payload['proposal_tokens'].tolist()
    seq_logits, seq_probs = [], []
    try:
        for position in range(len(proposal) + 1):
            if position:
                features = sequential.append(ids.new_tensor([[proposal[position - 1]]]))
            logits = sequential.predict(features, last_only=True)[0]
            p = logits_to_probabilities(logits, temperature)
            block_logits = payload['block_logits'][position].to(logits.device).double()
            block_p = payload['block_probs'][position].to(p.device)
            delta = (block_logits - logits.double()).abs()
            metric = dict(position=position, semantic_prefix_length=ids.shape[1] + emitted + position,
                total_variation=float((block_p - p).abs().sum().item() / 2),
                max_absolute_logit_difference=float(delta.max().item()),
                mean_absolute_logit_difference=float(delta.mean().item()),
                max_probability_difference=float((block_p - p).abs().max().item()),
                argmax_equal=int(block_p.argmax()) == int(p.argmax()))
            seq_logits.append(logits.detach().cpu().clone())
            seq_probs.append(p.detach().cpu().clone())
            on_row(metric)
    finally:
        payload['sequential_logits'] = seq_logits
        payload['sequential_probs'] = seq_probs
        payload['semantic_committed_output_prefix'] = output[:emitted]


def exercise(model, draft, binding, out, report, *, device, amp):
    """Shared real/toy-Qwen CPU-tested driver; public CLI only runs BF16 CUDA."""
    import torch
    from .cached_sampling import cached_speculative_sample, cached_target_sample
    from .tensor_sampling import TensorRandom
    protocol = binding['protocol']
    eos = model.generation_config.eos_token_id
    eos = frozenset([] if eos is None else eos if isinstance(eos, list) else [eos])
    baseline, target, audited_draft = audited_adapters(model, draft)
    cases = binding['cases']
    budget, temperature = protocol['max_new_tokens'], protocol['temperature']
    references = {}

    def run(case, seed, phase):
        number = case['index']
        emit(out, dict(event='case_started', case=number, seed=seed, phase=phase))
        ids = torch.tensor([case['prompt_token_ids']], dtype=torch.long, device=device)
        rng = lambda: TensorRandom(torch.Generator(device=device).manual_seed(seed))
        baseline_tokens = cached_target_sample(baseline, ids, budget, temperature=temperature, rng=rng(), eos_ids=eos)
        check_output(baseline_tokens, budget, eos, ids.shape[1], baseline.length)
        audit_rows, probes = [], []
        expected_before = ids.shape[1]

        def observe(observation):
            nonlocal expected_before
            row = audit_observation(observation, expected_before, audited_draft.projected_end, model.config.vocab_size)
            expected_before = observation.cache_after
            index = len(audit_rows)
            audit_rows.append(row)
            append_json(out / 'private-rounds.jsonl', dict(case=number, seed=seed, phase=phase, round=index, **row))
            if phase == 'reference' and index < protocol['probe_rounds']:
                payload = dict(block_logits=target.last_prediction[0].detach().cpu().clone(),
                    block_probs=observation.target_probs.detach().cpu().clone(),
                    actual_q=observation.proposal.draft_probs.detach().cpu().clone(),
                    proposal_tokens=observation.proposal.tokens.detach().cpu().clone(),
                    confidence_logits=observation.proposal.confidence_logits.detach().cpu().clone())
                path = out / f'private-probe-case{number}-seed{seed}-round{index}.pt'
                torch.save(payload, path)
                probes.append((index, path, payload))

        tokens, rounds = cached_speculative_sample(target, audited_draft, ids, budget,
            temperature=temperature, rng=rng(), eos_ids=eos, amp=amp, observer=observe, trace_tokens=True)
        check_output(tokens, budget, eos, ids.shape[1], target.length)
        if len(rounds) != len(audit_rows) or any(r['cache_after'] != a['cache_after'] for r, a in zip(rounds, audit_rows)):
            raise RuntimeError('Recorded rounds differ from observer commits')
        result = dict(case=number, split=case['split'], seed=seed, phase=phase,
            baseline_tokens=baseline_tokens, speculative_tokens=tokens, rounds=rounds,
            audit_rows=audit_rows, final_cache_length=target.length, execution_invariants_passed=True)
        report['runs'].append(result)
        save_report(out, report)
        for index, path, payload in probes:
            emitted = 1 + sum(r['committed'] for r in rounds[:index])
            def on_row(metric):
                entry = dict(case=number, seed=seed, round=index, **metric)
                report['numerical_probes'].append(entry)
                append_json(out / 'probe-metrics.jsonl', entry)
                save_report(out, report)
            try:
                compare_probe(model, ids, tokens, emitted, payload, temperature, on_row)
            finally:
                torch.save(payload, path.with_suffix('.partial.pt'))
                path.with_suffix('.partial.pt').replace(path)
        emit(out, dict(event='case_completed', case=number, seed=seed, phase=phase,
                       baseline_count=len(baseline_tokens), speculative_count=len(tokens), rounds=len(rounds)))
        return result

    def compare(reference, actual):
        baseline_equal = reference['baseline_tokens'] == actual['baseline_tokens']
        spec_equal = (reference['speculative_tokens'] == actual['speculative_tokens'] and
                      reference['rounds'] == actual['rounds'] and reference['audit_rows'] == actual['audit_rows'])
        report['checks'].append(dict(case=actual['case'], seed=actual['seed'], phase=actual['phase'],
            target_only_same_path_equal=baseline_equal, speculative_same_path_equal=spec_equal,
            passed=baseline_equal and spec_equal))
        save_report(out, report)
        if not baseline_equal or not spec_equal:
            raise RuntimeError('Same-path seeded reproduction/state-isolation check failed; traces preserved')

    for seed in protocol['seeds']:
        for case in cases:
            reference = run(case, seed, 'reference')
            references[(seed, case['index'])] = reference
            compare(reference, run(case, seed, 'immediate_repeat'))
        # Rotate reverse order so even the last reference case is retested only
        # after a different request (for three cases: 1,0,2 after 0,1,2).
        interleaved = list(reversed(cases))
        for case in interleaved[1:] + interleaved[:1]:
            compare(references[(seed, case['index'])], run(case, seed, 'interleaved_reverse'))


def worker(out):
    binding = json.loads((out / 'run.json').read_text())
    report = dict(status='running', binding_sha256=binding['binding_sha256'],
                  runs=[], checks=[], numerical_probes=[], execution_checks_passed=None)
    save_report(out, report)
    try:
        source = out / 'source' / 'dspark_qwen'
        if Path(__file__).resolve().parent != source.resolve():
            raise ValueError('Worker must execute from the bound source snapshot')
        verify_binding(binding, source)
        import torch
        import transformers
        from transformers import AutoModelForCausalLM
        from .checkpoint import load_checkpoint
        from .config import DraftConfig
        from .model import DSparkDraft
        from .tensor_sampling import PROBABILITY_POLICY
        if PROBABILITY_POLICY != binding['protocol']['probability_policy']:
            raise ValueError('Probability policy identity changed')
        if not torch.cuda.is_available():
            raise RuntimeError('Real gate requires an explicitly coordinated CUDA/ROCm GPU window')
        free, total = torch.cuda.mem_get_info()
        if free < binding['protocol']['minimum_free_memory_gib'] * 1024**3:
            raise RuntimeError('Bounded gate requires 8 GiB free before model load')
        torch.cuda.set_per_process_memory_fraction(binding['protocol']['max_process_memory_gib'] * 1024**3 / total)
        torch.manual_seed(0)
        runtime = dict(torch=torch.__version__, transformers=transformers.__version__, hip=torch.version.hip,
                       device=torch.cuda.get_device_name(), free_bytes_before_load=free,
                       deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
        write_json(out / 'runtime.json', runtime)
        model = AutoModelForCausalLM.from_pretrained(binding['model'], local_files_only=True,
            dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval()
        draft = DSparkDraft(model, DraftConfig.from_dict(binding['draft_config'])).to('cuda').eval()
        load_checkpoint(binding['checkpoint'], draft)
        with torch.no_grad():
            exercise(model, draft, binding, out, report, device='cuda', amp=True)
        report.update(status='completed', execution_checks_passed=True,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as exc:
        report.update(status='failed', execution_checks_passed=False, error_type=type(exc).__name__, error=str(exc))
        (out / 'error.txt').write_text(traceback.format_exc())
        emit(out, dict(event='failed', error_type=type(exc).__name__, message=str(exc)))
        return 1
    finally:
        save_report(out, report)
    emit(out, dict(event='completed', **aggregate(report)))
    return 0


def launch_worker(out, timeout):
    """Stream and preserve Python/native stdout+stderr; kill only this worker on timeout."""
    command = [sys.executable, '-m', 'dspark_qwen.eval_stochastic_gate', '--_worker', str(out)]
    environment = dict(os.environ, DSPARK_GATE_CAPTURED_STDOUT='1')
    deadline = time.monotonic() + timeout
    with subprocess.Popen(command, cwd=out / 'source', env=environment,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as process:
        try:
            with selectors.DefaultSelector() as selector, (out / 'stdout.log').open('ab') as log:
                selector.register(process.stdout, selectors.EVENT_READ)
                while selector.get_map():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(command, timeout)
                    for key, _ in selector.select(min(1.0, remaining)):
                        data = os.read(key.fd, 65536)
                        if not data:
                            selector.unregister(key.fileobj)
                            continue
                        log.write(data); log.flush()
                        sys.stdout.write(data.decode(errors='replace')); sys.stdout.flush()
            return process.wait(timeout=max(0.001, deadline - time.monotonic()))
        except BaseException:
            process.kill()
            process.wait()
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior-gate')
    parser.add_argument('--checkpoint', help='Optional relocated copy; hashes must equal original prior checkpoint')
    parser.add_argument('--output')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--seeds', type=int, nargs='+', default=[20261009])
    parser.add_argument('--max-new-tokens', type=int, default=32)
    parser.add_argument('--probe-rounds', type=int, default=2)
    parser.add_argument('--timeout-seconds', type=int, default=600)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--_worker', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._worker:
        return worker(Path(args._worker).resolve())
    if not args.prior_gate or not args.output:
        parser.error('--prior-gate and --output are required')
    out = Path(args.output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False, mode=0o700)
    report = dict(status='preparing', runs=[], checks=[], numerical_probes=[], execution_checks_passed=None)
    save_report(out, report)
    try:
        write_json(out / 'request.json', vars(args))
        binding = bind_inputs(args)
        binding['source_sha256'] = snapshot_source(out)
        binding['binding_sha256'] = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
        write_json(out / 'run.json', binding)
        report['binding_sha256'] = binding['binding_sha256']
        save_report(out, report)
        if args.dry_run:
            report.update(status='dry_run_complete', execution_checks_passed=None)
            save_report(out, report)
            emit(out, dict(event='dry_run_complete', cases=len(binding['cases']), binding_sha256=binding['binding_sha256'],
                           gpu_touched=False, source_snapshotted=True))
            return 0
        emit(out, dict(event='launching_snapshot_worker', timeout_seconds=args.timeout_seconds))
        # cwd makes the snapshot package authoritative; inherited PYTHONPATH is
        # unnecessary because Python adds this working directory for -m.
        returncode = launch_worker(out, args.timeout_seconds)
        worker_status = json.loads((out / 'result.json').read_text())['status']
        if returncode and worker_status != 'failed':
            raise RuntimeError(f'Worker terminated without complete failure record (exit {returncode})')
        if not returncode and worker_status != 'completed':
            raise RuntimeError('Worker returned success without a completed result')
        return returncode
    except BaseException as exc:
        if (out / 'result.json').exists():
            report = json.loads((out / 'result.json').read_text())
        report.update(status='failed', execution_checks_passed=False, error_type=type(exc).__name__, error=str(exc))
        save_report(out, report)
        (out / 'launcher-error.txt').write_text(traceback.format_exc())
        emit(out, dict(event='failed', error_type=type(exc).__name__, message=str(exc)))
        return 1


if __name__ == '__main__':
    sys.exit(main())
