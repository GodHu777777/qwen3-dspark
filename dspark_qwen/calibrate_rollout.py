"""CPU-only STS fit/eval over complete, bound fit44/eval43 collections.

Artifacts are private: they retain frozen prompt identities and raw-file hashes.
The worker exit is an OS result waited by the collector launcher; external
controller/launcher OS exits remain separate execution audit evidence.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

from .calibration import (RolloutBlock, calibrated_probabilities, fit_sts,
                          reliability_metrics, DEFAULT_TEMPERATURE_GRID)
from .collect_rollout import verify_binding
from .rollout_protocol import digest, prefix_evidence
from .eval_stochastic_gate import sha256

NUM_BINS = 20
GROUP_COUNTS = {'fit': 44, 'eval': 43}
FILES = ('run.json', 'result.json', 'aggregate.json', 'worker-exit.json', 'runtime.json',
         'private-blocks.jsonl', 'private-rounds.jsonl', 'private-outputs.jsonl')
COUNTERS = ('proposed_positions', 'verified_positions', 'attempted_positions',
            'effective_positions', 'accepted_draft_tokens')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _bad_constant(value):
    raise ValueError('Non-finite JSON number: ' + value)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON key: ' + key)
        result[key] = value
    return result


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=_bad_constant,
                      object_pairs_hook=_unique_object)


def read_lines(path):
    # An absent block/round file is valid only for an independently completed
    # all-zero-block group; the checks below still require every prompt output.
    if not path.exists():
        return []
    return [json.loads(line, parse_constant=_bad_constant, object_pairs_hook=_unique_object)
            for line in path.read_text().splitlines() if line.strip()]


def implementation_identity():
    here = Path(__file__).resolve().parent
    return {name: sha256(here / name) for name in ('calibration.py', 'calibrate_rollout.py')}


def common_identity(binding):
    manifest = binding['manifest']
    return dict(checkpoint_sha256=binding['checkpoint_sha256'],
        checkpoint_metadata_sha256=binding['checkpoint_metadata_sha256'],
        development_records_sha256=manifest['identity']['development_records_sha256'],
        data_identity=manifest['identity'], manifest_file_sha256=binding['manifest_file_sha256'],
        manifest_sha256=manifest['manifest_sha256'], rollout_protocol_sha256=digest(manifest['protocol']),
        probability_policy=manifest['protocol']['probability_policy'],
        collector_source_sha256=binding['source_sha256'], draft_config=binding['draft_config'],
        selection_file_sha256=binding['selection_file_sha256'], selection=binding['selection'])


def load_collection(directory, group):
    """Recheck identities, all completed cases, raw labels and aggregate counters."""
    require(group in GROUP_COUNTS, 'Calibration accepts fit or eval only')
    directory = Path(directory)
    binding, result = read(directory / 'run.json'), read(directory / 'result.json')
    require(binding.get('collection_group') == group, 'Wrong or missing explicit calibration group')
    verify_binding(binding, directory / 'source' / 'dspark_qwen')
    require(result['status'] == 'completed' and result['execution_checks_passed'] is True,
            'Incomplete or failed collection')
    require(result['binding_sha256'] == binding['binding_sha256'], 'Result binding changed')
    require(read(directory / 'worker-exit.json') == {'returncode': 0}, 'Missing successful worker OS exit')
    aggregate = read(directory / 'aggregate.json')
    require(aggregate['status'] == 'completed' and aggregate['execution_checks_passed'] is True and
            aggregate['binding_sha256'] == binding['binding_sha256'], 'Incomplete aggregate')
    cases = binding['cases']
    require(len(cases) == GROUP_COUNTS[group], 'Incomplete frozen group')
    require([c['id'] for c in cases] == [c['id'] for c in binding['manifest']['groups'][group]],
            'Cases differ from frozen group')
    prompt_ids = [c['id'] for c in cases]
    prompt_hashes = [c['prompt_sha256'] for c in cases]
    require(len(set(prompt_ids)) == len(cases) and len(set(prompt_hashes)) == len(cases),
            'Duplicate prompt identity')
    outputs = read_lines(directory / 'private-outputs.jsonl')
    rounds = read_lines(directory / 'private-rounds.jsonl')
    raw_blocks = read_lines(directory / 'private-blocks.jsonl')
    runs = result['runs']
    require(len(outputs) == len(runs) == len(cases), 'All frozen prompts must complete, including zero-block prompts')
    require(len(rounds) == len(raw_blocks) == result['blocks'] == aggregate['blocks'], 'Missing block evidence')
    require(aggregate['completed_prompts'] == len(cases), 'Incomplete aggregate prompt count')
    size = binding['manifest']['protocol']['block_size']
    budget = binding['manifest']['protocol']['max_new_tokens']
    blocks, cursor, totals = [], 0, Counter()
    counts, events = [0]*size, [0]*size
    for index, (case, run, output) in enumerate(zip(cases, runs, outputs)):
        require(case['index'] == run['case'] == index and run['seed'] == case['seed'], 'Case order or seed changed')
        require(run['execution_invariants_passed'] is True, 'Prompt invariants failed')
        require(output['id'] == case['id'], 'Output prompt identity changed')
        tokens = output['tokens']
        require(len(tokens) == run['output_tokens'] and 1 <= len(tokens) <= budget and
                all(type(t) is int and t >= 0 for t in tokens), 'Invalid or missing sampled output')
        require(type(run['rounds']) is int and run['rounds'] >= 0, 'Invalid round count')
        require(run['rounds'] > 0 or len(tokens) == 1, 'Zero-block prompt must retain exactly its initial draw')
        if run['rounds'] == 0:
            require(run['ended_eos'] is True or budget == 1, 'Unfinished zero-block prompt')
        committed, case_totals = [], Counter()
        for number in range(run['rounds']):
            require(cursor < len(rounds), 'Missing prompt rounds')
            row, raw = rounds[cursor], raw_blocks[cursor]
            cursor += 1
            require(row['case'] == index and row['round'] == number, 'Mixed, missing or reordered rounds')
            block = RolloutBlock(**{**raw, 'confidence_logits': tuple(raw['confidence_logits'])})
            require(block.block_id == f"{case['id']}:{number}" and block.prompt_id == case['id'] and
                    block.split == 'validation', 'Block identity or split changed')
            require(block.sampling_mode == 'stochastic' and block.collection_policy == 'full_proposal',
                    'Unsupported or censored collection')
            remaining = budget - 1 - len(committed)
            require(block.proposal_length == block.verified_length == min(size, remaining) and
                    block.proposal_length > 0, 'Missing or censored proposal tail')
            require(len(block.confidence_logits) == block.proposal_length and
                    all(type(x) in (int, float) and math.isfinite(x) for x in block.confidence_logits),
                    'Invalid confidence logits')
            require(list(block.confidence_logits) == row['confidence_logits'], 'Confidence records differ')
            evidence = prefix_evidence(block.proposal_length, block.verified_length,
                block.accepted_prefix_length, row['rejected_index'], block.accepted_eos_position)
            require(row['accepted'] == block.accepted_prefix_length and
                    row['accepted_eos_position'] == block.accepted_eos_position, 'Acceptance records differ')
            require(all(row[k] == v for k, v in evidence.items()), 'Prefix label or denominator changed')
            require(row['finite_and_shape_checked'] is True and row['q_dtype'] == 'torch.float64',
                    'Missing numerical audit')
            require(row['cache_before'] == len(case['prompt_token_ids']) + len(committed) and
                    row['cache_after'] - row['cache_before'] == len(row['committed_tokens']), 'Cache commit mismatch')
            emitted = row['committed_tokens']
            require(0 < len(emitted) <= remaining and
                    len(emitted) == block.accepted_prefix_length + int(row['extra_token_kind'] is not None),
                    'Invalid committed token count')
            require(row['extra_token_kind'] in (None, 'residual', 'bonus'), 'Unknown extra token kind')
            require((row['rejected_index'] is not None) == (row['extra_token_kind'] == 'residual'),
                    'Residual/rejection mismatch')
            require(row['extra_token_kind'] != 'bonus' or block.accepted_prefix_length == block.proposal_length,
                    'Bonus before full acceptance')
            require(block.accepted_eos_position is None or
                    (row['stop_reason'] == 'eos' and row['extra_token_kind'] is None), 'Accepted EOS semantics changed')
            committed.extend(emitted)
            require(row['stop_reason'] == 'round_complete' if number < run['rounds']-1 else
                    row['stop_reason'] in ('eos', 'budget'), 'Prompt termination changed')
            if number == run['rounds']-1:
                require((row['stop_reason'] == 'eos') == run['ended_eos'] and
                        (row['stop_reason'] != 'budget' or len(committed)+1 == budget), 'Final stop mismatch')
            for key in COUNTERS:
                case_totals[key] += block.accepted_prefix_length if key == 'accepted_draft_tokens' else evidence[key]
            for j, label in enumerate(evidence['prefix_labels']):
                counts[j] += 1
                events[j] += label
            blocks.append(block)
        require(tokens[1:] == committed, 'Outputs and committed rounds differ')
        require(all(run[k] == case_totals[k] for k in COUNTERS), 'Prompt counter mismatch')
        totals.update(case_totals)
    require(cursor == len(rounds), 'Unaccounted rounds')
    require(all(result[k] == aggregate[k] == totals[k] for k in COUNTERS), 'Aggregate counter mismatch')
    expected_positions = [dict(position=j, count=n, accepted_prefix_events=events[j])
                          for j, n in enumerate(counts) if n]
    require(result['per_position'] == aggregate['per_position'] == expected_positions, 'Per-position denominator mismatch')
    require(aggregate['output_tokens'] == sum(r['output_tokens'] for r in runs) and
            aggregate['eos_prompts'] == sum(r['ended_eos'] for r in runs), 'Output aggregate mismatch')
    evidence = dict(group=group, binding_sha256=binding['binding_sha256'],
        files_sha256={name: sha256(directory/name) if (directory/name).exists() else None for name in FILES},
        prompt_ids=prompt_ids, prompt_sha256=prompt_hashes, completed_prompts=len(cases),
        zero_block_prompts=sum(r['rounds'] == 0 for r in runs), blocks=len(blocks),
        worker_os_exit_code=0, effective_positions=sum(counts))
    runtime = read(directory / 'runtime.json')
    identity = common_identity(binding)
    identity['runtime'] = {key: runtime[key] for key in ('torch', 'transformers', 'hip', 'device')}
    return dict(identity=identity, evidence=evidence, blocks=blocks, block_size=size)


def position_report(collection, temperatures, constants, fitted_positions):
    blocks, evidence = collection['blocks'], collection['evidence']
    reports = []
    for j in range(collection['block_size']):
        eligible = [b for b in blocks if b.effective_length > j]
        labels = [int(j < b.accepted_prefix_length) for b in eligible]
        unscaled = [calibrated_probabilities(b.confidence_logits[:j+1], [1.]*(j+1))[1][-1] for b in eligible]
        scaled = [calibrated_probabilities(b.confidence_logits[:j+1], temperatures)[1][-1] for b in eligible]
        constant = constants[j]
        constant_metrics = reliability_metrics([constant]*len(labels), labels, num_bins=NUM_BINS) if constant is not None else reliability_metrics([], [], num_bins=NUM_BINS)
        reports.append(dict(position=j, observed_label_count=len(labels),
            block_coverage=len(eligible)/len(blocks) if blocks else None,
            prompts_with_observations=len({b.prompt_id for b in eligible}),
            prompt_coverage=len({b.prompt_id for b in eligible})/evidence['completed_prompts'],
            unscaled=reliability_metrics(unscaled, labels, num_bins=NUM_BINS),
            sts=dict(fitted_on_fit=bool(fitted_positions[j]), **reliability_metrics(scaled, labels, num_bins=NUM_BINS)),
            fit_prevalence_constant=dict(available=constant is not None, probability=constant,
                unavailable_reason=None if constant is not None else 'No effective fit labels at this position',
                **constant_metrics)))
    return reports


def fit_collection(directory, *, temperature_grid=DEFAULT_TEMPERATURE_GRID):
    collection = load_collection(directory, 'fit')
    fit = fit_sts(collection['blocks'], block_size=collection['block_size'],
                  identity=collection['identity'], temperature_grid=temperature_grid, num_bins=NUM_BINS)
    constants = [m['target_mean'] for m in fit['uncalibrated_per_position']]
    fitted = [m['fitted'] for m in fit['per_position']]
    artifact = dict(version=1, kind='frozen_sts_fit44', identity=collection['identity'],
        fit_evidence=collection['evidence'], implementation_sha256=implementation_identity(),
        fit=fit, fit_prefix_prevalence=constants,
        fit_metrics=position_report(collection, fit['temperatures'], constants, fitted),
        scope='Fitting-population metrics only; constants estimated only on fit44; no held-out or serving claim')
    artifact['artifact_sha256'] = digest(artifact)
    return artifact


def evaluate_collection(artifact, fit_directory, eval_directory):
    require(artifact.get('kind') == 'frozen_sts_fit44' and artifact.get('version') == 1, 'Unsupported STS artifact')
    require(digest({k:v for k,v in artifact.items() if k != 'artifact_sha256'}) == artifact['artifact_sha256'],
            'STS artifact changed')
    require(artifact['implementation_sha256'] == implementation_identity(), 'STS implementation changed')
    fit_collection_data = load_collection(fit_directory, 'fit')
    evaluation = load_collection(eval_directory, 'eval')
    require(artifact['identity'] == fit_collection_data['identity'] == evaluation['identity'],
            'Checkpoint/data/protocol/collector source identity differs')
    require(artifact['fit_evidence'] == fit_collection_data['evidence'], 'Fit collection changed after freezing')
    for key in ('prompt_ids', 'prompt_sha256'):
        require(not set(fit_collection_data['evidence'][key]) & set(evaluation['evidence'][key]), 'Fit/eval prompt leakage')
    fit = artifact['fit']
    require(fit['num_bins'] == NUM_BINS and len(fit['temperatures']) == evaluation['block_size'] and
            len(artifact['fit_prefix_prevalence']) == evaluation['block_size'], 'STS dimensions or bins changed')
    # No fitting or hyperparameter choice is performed on eval43.
    result = dict(version=1, kind='frozen_sts_eval43', artifact_sha256=artifact['artifact_sha256'],
        identity=evaluation['identity'], fit_evidence=artifact['fit_evidence'],
        eval_evidence=evaluation['evidence'], num_bins=NUM_BINS,
        per_position=position_report(evaluation, fit['temperatures'], artifact['fit_prefix_prevalence'],
                                     [m['fitted'] for m in fit['per_position']]),
        scope='Prompt-held-out from STS fit only; all dev was teacher-forced monitored; final test remains locked; no serving claim')
    result['report_sha256'] = digest(result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    fit = sub.add_parser('fit')
    fit.add_argument('--collection', required=True)
    fit.add_argument('--temperature-grid', type=float, nargs='+', default=DEFAULT_TEMPERATURE_GRID)
    fit.add_argument('--output', required=True)
    evaluate = sub.add_parser('eval')
    evaluate.add_argument('--artifact', required=True)
    evaluate.add_argument('--fit-collection', required=True)
    evaluate.add_argument('--collection', required=True)
    evaluate.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    if args.command == 'fit':
        result = fit_collection(args.collection, temperature_grid=args.temperature_grid)
    else:
        result = evaluate_collection(read(args.artifact), args.fit_collection, args.collection)
    with Path(args.output).open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(kind=result['kind'], output=str(Path(args.output)), gpu_touched=False)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
