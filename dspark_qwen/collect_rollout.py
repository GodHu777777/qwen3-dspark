"""Generic quality32 rollout collector. Dry-run is stdlib-only; GPU runs require coordination."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

from .eval_stochastic_gate import (sha256, write_json, append_json, audited_adapters,
                                  audit_observation, check_output, compare_probe)
from .rollout_protocol import digest, verify_manifest, prefix_evidence, POLICY


def bind_inputs(checkpoint, manifest_path):
    checkpoint = Path(checkpoint).resolve()
    metadata_path = checkpoint / 'metadata.json'
    metadata = json.loads(metadata_path.read_text())
    cfg = metadata['identity']['config']
    manifest = json.loads(Path(manifest_path).read_text())
    rows = verify_manifest(manifest, cfg)
    weights = sha256(checkpoint / 'draft.safetensors')
    if weights != metadata['draft_weights_sha256']:
        raise ValueError('Checkpoint weight hash mismatch')
    identity = manifest['identity']
    if (identity['development_records_sha256'] != metadata['identity']['records_sha256'] or
            identity['target_fingerprint'] != metadata['identity']['target_fingerprint'] or
            manifest['protocol']['block_size'] != metadata['draft_config']['block_size']):
        raise ValueError('Checkpoint differs from frozen data/target/block identity')
    model_config = json.loads((Path(cfg['model']) / 'config.json').read_text())
    if model_config.get('model_type') != 'qwen3' or model_config.get('use_sliding_window', False):
        raise ValueError('Dense Qwen3 target required')
    cases = []
    for index, case in enumerate(manifest['groups']['quality']):
        tokens = rows[case['id']]['prompt_token_ids']
        if len(tokens) > 4096 or any(t >= model_config['vocab_size'] for t in tokens):
            raise ValueError('Prompt exceeds bound or target vocabulary; no silent filtering')
        cases.append(dict(index=index, **case, split='validation', prompt_token_ids=tokens))
    return dict(checkpoint=str(checkpoint), checkpoint_sha256=weights,
        checkpoint_metadata_sha256=sha256(metadata_path), manifest_path=str(Path(manifest_path).resolve()),
        manifest_file_sha256=sha256(manifest_path), manifest=manifest, model=cfg['model'],
        draft_config=metadata['draft_config'], cases=cases)


def snapshot(out):
    source = Path(__file__).resolve().parent
    dest = out / 'source' / 'dspark_qwen'
    dest.mkdir(parents=True)
    hashes = {}
    for path in sorted(source.glob('*.py')):
        if path.name.startswith('._'):
            raise ValueError('AppleDouble source sidecar: deploy with git archive')
        before = sha256(path)
        shutil.copyfile(path, dest / path.name)
        if before != sha256(path) or before != sha256(dest / path.name):
            raise ValueError('Source changed during snapshot')
        hashes[path.name] = before
    return hashes


def verify_binding(binding, source):
    if digest({k: v for k, v in binding.items() if k != 'binding_sha256'}) != binding['binding_sha256']:
        raise ValueError('Run binding changed')
    if {p.name for p in source.glob('*.py')} != set(binding['source_sha256']):
        raise ValueError('Source inventory changed')
    for name, expected in binding['source_sha256'].items():
        if sha256(source / name) != expected:
            raise ValueError('Source content changed')
    current = bind_inputs(binding['checkpoint'], binding['manifest_path'])
    if current != {k: v for k, v in binding.items() if k not in ('source_sha256', 'binding_sha256')}:
        raise ValueError('Bound inputs changed')


def new_report(binding=None):
    return dict(status='preparing', binding_sha256=None if binding is None else binding['binding_sha256'],
                runs=[], numerical_probes=[], blocks=0, proposed_positions=0, verified_positions=0,
                attempted_positions=0, effective_positions=0, accepted_draft_tokens=0,
                per_position=[], execution_checks_passed=None)


def aggregate(report):
    fields = ('status', 'binding_sha256', 'blocks', 'proposed_positions', 'verified_positions',
              'attempted_positions', 'effective_positions', 'accepted_draft_tokens', 'per_position',
              'execution_checks_passed', 'error_type')
    result = {k: report.get(k) for k in fields}
    result['completed_prompts'] = len(report['runs'])
    result['output_tokens'] = sum(r['output_tokens'] for r in report['runs'])
    result['eos_prompts'] = sum(r['ended_eos'] for r in report['runs'])
    probes = report['numerical_probes']
    result['numerical_fidelity'] = dict(rows=len(probes),
        max_total_variation=max((r['total_variation'] for r in probes), default=None),
        nonzero_tv_rows=sum(r['total_variation'] > 0 for r in probes),
        argmax_changed_rows=sum(not r['argmax_equal'] for r in probes),
        scope='Native block versus fresh sequential same-prefix differences; no losslessness verdict')
    result['scope'] = 'Native-protocol quality only. Not a speed benchmark or untouched model-selection holdout.'
    return result


def save(out, report):
    write_json(out / 'result.json', report)
    write_json(out / 'aggregate.json', aggregate(report))


def exercise(model, draft, binding, out, report, *, device, amp):
    import torch
    from .cached_sampling import cached_speculative_sample
    from .tensor_sampling import TensorRandom, PROBABILITY_POLICY
    protocol = binding['manifest']['protocol']
    if protocol['probability_policy'] != PROBABILITY_POLICY:
        raise ValueError('Probability policy mismatch')
    eos = model.generation_config.eos_token_id
    eos = frozenset([] if eos is None else eos if isinstance(eos, list) else [eos])
    _, target, audited_draft = audited_adapters(model, draft)
    selected = {(p['id'], p['round']) for p in binding['manifest']['tv_probes']}
    for case in binding['cases']:
        ids = torch.tensor([case['prompt_token_ids']], dtype=torch.long, device=device)
        expected_before, index, committed_count = ids.shape[1], 0, 1
        probes = []
        counts_before = {k: report[k] for k in ('accepted_draft_tokens', 'proposed_positions',
            'verified_positions', 'attempted_positions', 'effective_positions')}
        def observe(observation):
            nonlocal expected_before, index, committed_count
            row = audit_observation(observation, expected_before, audited_draft.projected_end, model.config.vocab_size)
            evidence = prefix_evidence(len(row['proposal_tokens']), row['verified_proposal_length'],
                                       row['accepted'], row['rejected_index'], row['accepted_eos_position'])
            remaining = protocol['max_new_tokens'] - committed_count
            if len(row['proposal_tokens']) != min(protocol['block_size'], remaining):
                raise RuntimeError('Proposal was censored before remaining-budget truncation')
            q, p = observation.proposal.draft_probs, observation.target_probs
            row.update(q_dtype=str(q.dtype), q_max_sum_error=float((q.sum(-1) - 1).abs().max()) if q.shape[0] else 0.,
                       p_max_sum_error=float((p.sum(-1) - 1).abs().max()), **evidence)
            block = dict(block_id=f"{case['id']}:{index}", prompt_id=case['id'], split='validation',
                         confidence_logits=row['confidence_logits'], proposal_length=len(row['proposal_tokens']),
                         accepted_prefix_length=row['accepted'], verified_length=row['verified_proposal_length'],
                         accepted_eos_position=row['accepted_eos_position'], sampling_mode='stochastic',
                         collection_policy='full_proposal')
            append_json(out / 'private-blocks.jsonl', block)
            append_json(out / 'private-rounds.jsonl', dict(case=case['index'], round=index, **row))
            report['blocks'] += 1
            for field in ('proposed_positions', 'verified_positions', 'attempted_positions', 'effective_positions'):
                report[field] += evidence[field]
            report['accepted_draft_tokens'] += row['accepted']
            for position, label in enumerate(evidence['prefix_labels']):
                if len(report['per_position']) <= position:
                    report['per_position'].append(dict(position=position, count=0, accepted_prefix_events=0))
                report['per_position'][position]['count'] += 1
                report['per_position'][position]['accepted_prefix_events'] += label
            save(out, report)
            if (case['id'], index) in selected:
                payload = dict(block_logits=target.last_prediction[0].detach().cpu().clone(),
                    block_probs=p.detach().cpu().clone(), actual_q=q.detach().cpu().clone(),
                    proposal_tokens=observation.proposal.tokens.detach().cpu().clone())
                path = out / f"private-probe-case{case['index']}-round{index}.pt"
                torch.save(payload, path)
                probes.append((index, committed_count, path, payload))
            expected_before = observation.cache_after
            committed_count += len(row['committed_tokens'])
            index += 1
        tokens, rounds = cached_speculative_sample(target, audited_draft, ids, protocol['max_new_tokens'],
            temperature=protocol['temperature'], rng=TensorRandom(torch.Generator(device=device).manual_seed(case['seed'])),
            eos_ids=eos, amp=amp, observer=observe)
        check_output(tokens, protocol['max_new_tokens'], eos, ids.shape[1], target.length)
        if len(rounds) != index:
            raise RuntimeError('Observer and returned rounds differ')
        report['runs'].append(dict(case=case['index'], seed=case['seed'], output_tokens=len(tokens),
            ended_eos=tokens[-1] in eos, rounds=index, execution_invariants_passed=True,
            **{k: report[k] - v for k, v in counts_before.items()},
            preselected_probes=[dict(round=r, status='collected' if r < index else 'unreached')
                               for rid, r in sorted(selected) if rid == case['id']]))
        append_json(out / 'private-outputs.jsonl', dict(id=case['id'], tokens=tokens))
        save(out, report)
        for round_index, emitted, path, payload in probes:
            def on_row(metric):
                entry = dict(case=case['index'], round=round_index, **metric)
                report['numerical_probes'].append(entry)
                append_json(out / 'probe-metrics.jsonl', entry)
                save(out, report)
            try:
                compare_probe(model, ids, tokens, emitted, payload, protocol['temperature'], on_row)
            finally:
                torch.save(payload, path.with_suffix('.partial.pt'))
                path.with_suffix('.partial.pt').replace(path)
        print(json.dumps(dict(event='prompt_completed', case=case['index'], output_tokens=len(tokens), rounds=index)), flush=True)


def worker(out):
    binding = json.loads((out / 'run.json').read_text())
    report = new_report(binding)
    report['status'] = 'running'
    save(out, report)
    try:
        source = out / 'source' / 'dspark_qwen'
        if Path(__file__).resolve().parent != source.resolve():
            raise ValueError('Worker must execute from bound source snapshot')
        verify_binding(binding, source)
        import torch
        import transformers
        from transformers import AutoModelForCausalLM
        from .checkpoint import load_checkpoint
        from .config import DraftConfig
        from .model import DSparkDraft
        if not torch.cuda.is_available():
            raise RuntimeError('Real collection requires a coordinated GPU window')
        free, total = torch.cuda.mem_get_info()
        if free < 8 * 1024**3:
            raise RuntimeError('Require 8 GiB free before loading')
        torch.cuda.set_per_process_memory_fraction(6 * 1024**3 / total)
        torch.manual_seed(0)
        write_json(out / 'runtime.json', dict(torch=torch.__version__, transformers=transformers.__version__,
            hip=torch.version.hip, device=torch.cuda.get_device_name(), free_bytes_before_load=free))
        model = AutoModelForCausalLM.from_pretrained(binding['model'], local_files_only=True,
            dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval()
        draft = DSparkDraft(model, DraftConfig.from_dict(binding['draft_config'])).to('cuda').eval()
        load_checkpoint(binding['checkpoint'], draft)
        with torch.no_grad():
            exercise(model, draft, binding, out, report, device='cuda', amp=True)
        report.update(status='completed', execution_checks_passed=True,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated())
        return 0
    except BaseException as exc:
        report.update(status='failed', execution_checks_passed=False, error_type=type(exc).__name__, error=str(exc))
        (out / 'error.txt').write_text(traceback.format_exc())
        return 1
    finally:
        save(out, report)


def launch_worker(out, timeout):
    environment = dict(os.environ)
    environment.pop('PYTHONPATH', None)
    with (out / 'stdout.log').open('wb') as log:
        result = subprocess.run([sys.executable, '-m', 'dspark_qwen.collect_rollout', '--_worker', str(out)],
            cwd=out / 'source', env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    write_json(out / 'worker-exit.json', dict(returncode=result.returncode))
    return result.returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint')
    parser.add_argument('--manifest')
    parser.add_argument('--output')
    parser.add_argument('--timeout-seconds', type=int, default=3600)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--_worker', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._worker:
        return worker(Path(args._worker).resolve())
    if not args.checkpoint or not args.manifest or not args.output or not 1 <= args.timeout_seconds <= 7200:
        parser.error('--checkpoint, --manifest, --output and timeout 1..7200 required')
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False, mode=0o700)
    report = new_report()
    save(out, report)
    try:
        write_json(out / 'request.json', vars(args))
        binding = bind_inputs(args.checkpoint, args.manifest)
        binding['source_sha256'] = snapshot(out)
        binding['binding_sha256'] = digest(binding)
        write_json(out / 'run.json', binding)
        report['binding_sha256'] = binding['binding_sha256']
        save(out, report)
        if args.dry_run:
            report['status'] = 'dry_run_complete'
            save(out, report)
            print(json.dumps(dict(status=report['status'], prompts=len(binding['cases']), gpu_touched=False)))
            return 0
        code = launch_worker(out, args.timeout_seconds)
        status = json.loads((out / 'result.json').read_text())['status']
        if status != ('completed' if code == 0 else 'failed'):
            raise RuntimeError(f'Incomplete worker evidence (exit {code})')
        return code
    except BaseException as exc:
        report = json.loads((out / 'result.json').read_text())
        report.update(status='failed', execution_checks_passed=False, error_type=type(exc).__name__, error=str(exc))
        save(out, report)
        (out / 'launcher-error.txt').write_text(traceback.format_exc())
        return 1


if __name__ == '__main__':
    sys.exit(main())
