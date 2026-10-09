#!/usr/bin/env python3
"""Read-only stdlib reanalysis of recovered, frozen quality32 rollout evidence."""
import argparse
from collections import Counter
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path

STEPS = (128, 512, 1280)
BINS = ((0, 31), (32, 63), (64, 95), (96, 127))
FILES = ('run.json', 'result.json', 'aggregate.json', 'runtime.json', 'worker-exit.json',
         'private-rounds.jsonl', 'private-blocks.jsonl', 'private-outputs.jsonl')
COUNTERS = ('proposed_positions', 'verified_positions', 'attempted_positions', 'effective_positions', 'accepted_draft_tokens')
LIMITS = ['Distinct checkpoint trajectories; differences are descriptive, not matched-prefix causal effects.',
    'Selected-token alpha is not full-vocabulary per-state overlap.',
    'Rounds within prompts are correlated; no independent-round confidence intervals.',
    'Development quality32 only; no final-test, generation, fitting, policy search or speed extrapolation.',
    'Recovered raw hashes establish current recovered-byte identity, not a historical precommitted raw-file hash.']


def require(ok, message):
    if not ok: raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON key'); result[key] = value
    return result


def loads(text):
    def invalid(value): raise ValueError('Nonfinite JSON constant')
    return json.loads(text, object_pairs_hook=object_pairs, parse_constant=invalid)


def read(path): return loads(Path(path).read_text())
def lines(path): return [loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def prefix_evidence(n, verified, accepted, rejected, accepted_eos):
    require(all(type(x) is int for x in (n, verified, accepted)) and 0 <= accepted <= verified <= n, 'Invalid proposal lengths')
    require(rejected is None or (type(rejected) is int and rejected == accepted and rejected < verified), 'Invalid rejected index')
    require(accepted_eos is None or (type(accepted_eos) is int and accepted_eos == accepted-1 and accepted > 0 and rejected is None), 'Invalid accepted EOS')
    require(accepted == verified or rejected is not None or accepted_eos is not None, 'Unaccounted unaccepted tail')
    effective = verified if accepted_eos is None else accepted_eos+1
    return dict(proposed_positions=n, verified_positions=verified, attempted_positions=accepted+int(rejected is not None),
        effective_positions=effective, prefix_labels=[int(j < accepted) for j in range(effective)])


def numeric(value): return type(value) in (int, float) and math.isfinite(value)


def first_risk(row):
    reasons = []
    p = row.get('selected_p'); q = row.get('selected_q')
    for name, values in (('p', p), ('q', q)):
        if not isinstance(values, list) or not values: reasons.append('missing_selected_'+name)
        elif not numeric(values[0]): reasons.append('nonfinite_or_nonnumeric_selected_'+name)
        if isinstance(values, list) and 'proposal_tokens' in row and len(values) != len(row['proposal_tokens']):
            reasons.append('selected_'+name+'_length_mismatch')
    p0 = p[0] if isinstance(p, list) and p else None
    q0 = q[0] if isinstance(q, list) and q else None
    if numeric(p0) and p0 < 0: reasons.append('negative_selected_p')
    if numeric(q0) and q0 <= 0: reasons.append('nonpositive_selected_q')
    if reasons: return dict(valid=False, missing_or_invalid_reasons=reasons, p0=p0 if numeric(p0) else None, q0=q0 if numeric(q0) else None, p_over_q=None, alpha=None)
    ratio = p0/q0
    # Both selected scalars may be finite while the positive quotient overflows.
    # Preserve a decimal representation instead of dropping a valid alpha=1 row.
    if math.isfinite(ratio): stored = ratio
    else:
        with localcontext() as context:
            context.prec = 50; stored = str(Decimal.from_float(float(p0))/Decimal.from_float(float(q0)))
    return dict(valid=True, missing_or_invalid_reasons=[], p0=p0, q0=q0, p_over_q=stored,
        ratio_float_overflow=not math.isfinite(ratio), alpha=min(1., ratio))


def mean(values): return math.fsum(values)/len(values) if values else None


def ratio_mean(rows):
    if not rows: return None
    with localcontext() as context:
        context.prec = 50
        value = sum((Decimal(str(x['p_over_q'])) for x in rows), Decimal(0))/len(rows)
    result = float(value)
    return result if math.isfinite(result) else str(value)


def summarize(rows):
    valid = [r for r in rows if r['valid']]
    return dict(rounds=len(rows), label_count=len(rows), first_prefix_events=sum(r['first_prefix_label'] for r in rows),
        observed_first_prefix_rate=mean([r['first_prefix_label'] for r in rows]), valid_risk_count=len(valid),
        missing_or_invalid_risk_count=len(rows)-len(valid),
        missing_or_invalid_reasons=dict(Counter(reason for r in rows for reason in r['missing_or_invalid_reasons'])),
        selected_ratio_mean=ratio_mean(valid), selected_ratio_float_overflows=sum(r.get('ratio_float_overflow', False) for r in valid),
        mean_alpha=mean([r['alpha'] for r in valid]), mean_rejection_probability=mean([1-r['alpha'] for r in valid]),
        observed_first_prefix_rate_on_valid_risk=mean([r['first_prefix_label'] for r in valid]))


def equal_prompt(prompts):
    metrics = ('observed_first_prefix_rate', 'mean_alpha', 'mean_rejection_probability', 'observed_first_prefix_rate_on_valid_risk')
    return {key: dict(value=mean([r[key] for r in prompts if r[key] is not None]),
                     available_prompts=sum(r[key] is not None for r in prompts), total_prompts=32) for key in metrics}


def check_collection(directory, step, collector_source):
    directory = Path(directory); source = Path(collector_source)
    for name in FILES: require((directory/name).is_file(), 'Missing artifact: '+name)
    hashes = {name: sha(directory/name) for name in FILES}
    binding, result, aggregate = (read(directory/name) for name in ('run.json', 'result.json', 'aggregate.json'))
    require(digest({k:v for k,v in binding.items() if k != 'binding_sha256'}) == binding['binding_sha256'], 'Binding digest differs')
    require(Path(binding['checkpoint']).name == f'step-{step:06d}', 'Checkpoint step binding differs')
    manifest = binding['manifest']; protocol = manifest['protocol']
    require(digest({k:v for k,v in manifest.items() if k != 'manifest_sha256'}) == manifest['manifest_sha256'], 'Panel digest differs')
    require(binding.get('collection_group', 'quality') == 'quality', 'Only historical quality32 allowed')
    require(protocol['max_new_tokens'] == 128 and protocol['block_size'] == 7 and protocol['temperature'] == 1 and
        protocol['probability_policy'] == 'float64_softmax_normalize_cdf_v1' and protocol['collection_policy'] == 'full_proposal', 'Original quality protocol required')
    require({p.name:sha(p) for p in source.glob('*.py')} == binding['source_sha256'], 'Historical source inventory/hash differs')
    panel = directory.parent/'panel.private.json'
    require(panel.is_file() and sha(panel) == binding['manifest_file_sha256'] and read(panel) == manifest, 'Recovered panel bytes differ')
    for document in (result, aggregate):
        require(document['status'] == 'completed' and document['execution_checks_passed'] is True and
            document['binding_sha256'] == binding['binding_sha256'], 'Incomplete or unbound collection')
    require(read(directory/'worker-exit.json') == {'returncode':0}, 'Successful worker exit required')
    cases, runs = binding['cases'], result['runs']
    require(len(cases) == len(runs) == len(manifest['groups']['quality']) == 32, 'All32 quality cases required')
    require(len({c['id'] for c in cases}) == len({c['prompt_sha256'] for c in cases}) == 32, 'Duplicate quality prompt')
    rounds, blocks, outputs = (lines(directory/name) for name in ('private-rounds.jsonl', 'private-blocks.jsonl', 'private-outputs.jsonl'))
    require(len(rounds) == len(blocks) == result['blocks'] == aggregate['blocks'], 'Missing or extra round/block records')
    require(len(outputs) == aggregate['completed_prompts'] == 32, 'Missing prompt output')
    rows = []; prompts = []; cursor = 0; totals = Counter(); counts = [0]*7; events = [0]*7
    for ordinal, (case, run, output) in enumerate(zip(cases, runs, outputs)):
        require(case['index'] == run['case'] == ordinal and case['split'] == 'validation' and case['seed'] == run['seed'], 'Case order/split/seed differs')
        require({k:case[k] for k in manifest['groups']['quality'][ordinal]} == manifest['groups']['quality'][ordinal] and
            digest(case['prompt_token_ids']) == case['prompt_sha256'], 'Case differs from quality panel')
        require(run['execution_invariants_passed'] is True and output['id'] == case['id'], 'Output/run identity differs')
        tokens = output['tokens']; budget = protocol['max_new_tokens']; size = protocol['block_size']
        require(len(tokens) == run['output_tokens'] and 1 <= len(tokens) <= budget and all(type(t) is int and t >= 0 for t in tokens), 'Invalid output token evidence')
        require(type(run['rounds']) is int and run['rounds'] >= 0 and type(run['ended_eos']) is bool, 'Invalid run coverage')
        require(run['ended_eos'] or len(tokens) == budget, 'Output stopped before budget without EOS')
        require(run['rounds'] > 0 or (len(tokens) == 1 and run['ended_eos']), 'Zero-round prompt must be initial EOS')
        committed = []; progress = 1; case_totals = Counter(); own = []
        for number in range(run['rounds']):
            require(cursor < len(rounds), 'Missing round'); row, block = rounds[cursor], blocks[cursor]; cursor += 1
            require(row['case'] == ordinal and row['round'] == number, 'Mixed/missing/reordered rounds')
            require(block['block_id'] == f"{case['id']}:{number}" and block['prompt_id'] == case['id'] and
                block['split'] == 'validation' and block['sampling_mode'] == 'stochastic' and block['collection_policy'] == 'full_proposal', 'Block identity/policy differs')
            n = len(row['proposal_tokens']); remaining = budget-progress
            require(n == block['proposal_length'] == block['verified_length'] == row['verified_proposal_length'] == min(size, remaining) and n > 0, 'Proposal coverage/censoring differs')
            require(all(type(t) is int and t >= 0 for t in row['proposal_tokens']), 'Invalid proposal token evidence')
            require(row['accepted'] == block['accepted_prefix_length'] and row['accepted_eos_position'] == block['accepted_eos_position'], 'Block/round acceptance differs')
            evidence = prefix_evidence(n, row['verified_proposal_length'], row['accepted'], row['rejected_index'], row['accepted_eos_position'])
            require(all(row[k] == value for k,value in evidence.items()), 'Prefix labels/denominators differ')
            require(row['prefix_labels'] and row['prefix_labels'][0] == int(row['accepted'] >= 1), 'First-prefix label differs')
            require(row['confidence_logits'] == block['confidence_logits'] and len(row['confidence_logits']) == n and
                all(numeric(x) for x in row['confidence_logits']), 'Confidence coverage differs')
            require(row['finite_and_shape_checked'] is True and row['q_dtype'] == 'torch.float64', 'Missing original numerical audit')
            emitted = row['committed_tokens']; accepted = row['accepted']; extra = row['extra_token_kind']
            require(0 < len(emitted) <= remaining and len(emitted) == accepted+int(extra is not None) and
                emitted[:accepted] == row['proposal_tokens'][:accepted], 'Committed acceptance/token count differs')
            require(extra in (None, 'residual', 'bonus') and (row['rejected_index'] is not None) == (extra == 'residual') and
                (extra != 'bonus' or accepted == n), 'Extra-token/rejection semantics differ')
            require(row['accepted_eos_position'] is None or (row['stop_reason'] == 'eos' and extra is None), 'Accepted EOS semantics differ')
            require(progress == 1+len(committed) == row['cache_before']-len(case['prompt_token_ids'])+1 and
                row['cache_after'] == row['cache_before']+len(emitted), 'Output-progress/cache reconstruction differs')
            require(1 <= progress <= 127, 'Emitted-before progress outside original domain')
            require(row['stop_reason'] == 'round_complete' if number < run['rounds']-1 else
                row['stop_reason'] in ('eos', 'budget'), 'Round termination differs')
            if number == run['rounds']-1:
                require((row['stop_reason'] == 'eos') == run['ended_eos'] and
                    (row['stop_reason'] != 'budget' or progress+len(emitted) == budget), 'Final termination differs')
            risk = first_risk(row)
            value = dict(checkpoint_step=step, ordinal=ordinal, round=number, emitted_before=progress,
                progress_bin=f'{BINS[progress//32][0]}-{BINS[progress//32][1]}', first_prefix_label=int(accepted >= 1), **risk)
            own.append(value); rows.append(value); committed.extend(emitted); progress += len(emitted)
            for key in COUNTERS: case_totals[key] += accepted if key == 'accepted_draft_tokens' else evidence[key]
            for j,label in enumerate(evidence['prefix_labels']): counts[j] += 1; events[j] += label
        require(tokens[1:] == committed and progress == run['output_tokens'], 'Output/committed/progress mismatch')
        require(all(run[k] == case_totals[k] for k in COUNTERS), 'Per-prompt aggregate differs'); totals.update(case_totals)
        prompt = dict(ordinal=ordinal, status='available', output_tokens=run['output_tokens'], eos=run['ended_eos'],
            budget_reached=run['output_tokens'] == budget, zero_round=run['rounds'] == 0, expected_rounds=run['rounds'], **summarize(own))
        prompt['progress_bins'] = []
        for lo,hi in BINS:
            current = [r for r in own if lo <= r['emitted_before'] <= hi]
            prompt['progress_bins'].append(dict(bin=f'{lo}-{hi}', status='available' if current else 'no_round',
                no_round_reason=None if current else ('initial_eos' if run['rounds'] == 0 else 'terminated_before_bin' if run['output_tokens'] <= lo else 'no_proposal_after_crossing'), **summarize(current)))
        prompts.append(prompt)
    require(cursor == len(rounds), 'Unaccounted round records')
    require(all(result[k] == aggregate[k] == totals[k] for k in COUNTERS), 'Overall aggregate differs')
    positions = [dict(position=j,count=n,accepted_prefix_events=events[j]) for j,n in enumerate(counts) if n]
    require(result['per_position'] == aggregate['per_position'] == positions, 'Per-position aggregate differs')
    require(aggregate['output_tokens'] == sum(r['output_tokens'] for r in runs) and aggregate['eos_prompts'] == sum(r['ended_eos'] for r in runs), 'Output/EOS aggregate differs')
    require(all(sha(directory/name) == value for name,value in hashes.items()), 'Input changed during analysis')
    summary = dict(checkpoint_step=step, status='available', files_sha256=hashes, binding_sha256=binding['binding_sha256'],
        all32_ordinals_retained=True, completed_prompts=32, zero_round_prompts=sum(p['zero_round'] for p in prompts),
        eos_prompts=sum(p['eos'] for p in prompts), budget_reached_prompts=sum(p['budget_reached'] for p in prompts),
        output_tokens=sum(p['output_tokens'] for p in prompts), pooled=summarize(rows), equal_prompt=equal_prompt(prompts), prompts=prompts)
    summary['progress_bins'] = []
    for index,(lo,hi) in enumerate(BINS):
        grouped = [r for r in rows if lo <= r['emitted_before'] <= hi]; per_prompt = [p['progress_bins'][index] for p in prompts]
        summary['progress_bins'].append(dict(bin=f'{lo}-{hi}', total_prompts=32,
            prompts_with_rounds=sum(p['rounds'] > 0 for p in per_prompt), prompts_with_valid_risk=sum(p['valid_risk_count'] > 0 for p in per_prompt),
            pooled=summarize(grouped), equal_prompt=equal_prompt(per_prompt)))
    summary['low_alpha_prompt_ordinals'] = [p['ordinal'] for p in prompts if p['mean_alpha'] is not None and p['mean_alpha'] < .5]
    summary['observed_below_half_prompt_ordinals'] = [p['ordinal'] for p in prompts if p['observed_first_prefix_rate'] is not None and p['observed_first_prefix_rate'] < .5]
    identity = dict(cases=cases, manifest=manifest, source_sha256=binding['source_sha256'], runtime={k:read(directory/'runtime.json')[k] for k in ('torch','transformers','hip','device')})
    return summary, rows, identity


def evidence_gap(step, error):
    return dict(checkpoint_step=step, status='evidence_gap', reason=error, all32_ordinals_retained=True,
        prompts=[dict(ordinal=i,status='unavailable',reason=error,progress_bins=[dict(bin=f'{a}-{b}',status='unavailable') for a,b in BINS]) for i in range(32)])


def analyze(raw_root, collector_source, output):
    raw_root, output = Path(raw_root), Path(output); output.mkdir(parents=True, exist_ok=False, mode=0o700)
    summaries = []; private = []; identities = []
    for step in STEPS:
        try:
            summary, rows, identity = check_collection(raw_root/str(step)/'collection', step, collector_source)
            if identities: require(identity == identities[0], 'Checkpoint panel/source/runtime identity differs')
            identities.append(identity); summaries.append(summary); private.extend(rows)
        except (ValueError, KeyError, TypeError, OSError, IndexError) as exc:
            summaries.append(evidence_gap(step, type(exc).__name__+': '+str(exc)))
    comparisons = []
    for left,right in zip(summaries,summaries[1:]):
        if left['status'] != 'available' or right['status'] != 'available': continue
        prompt_changes = []
        for a,b in zip(left['prompts'],right['prompts']):
            prompt_changes.append(dict(ordinal=a['ordinal'], alpha_delta=None if a['mean_alpha'] is None or b['mean_alpha'] is None else b['mean_alpha']-a['mean_alpha'],
                observed_rate_delta=None if a['observed_first_prefix_rate'] is None or b['observed_first_prefix_rate'] is None else b['observed_first_prefix_rate']-a['observed_first_prefix_rate']))
        comparisons.append(dict(from_step=left['checkpoint_step'],to_step=right['checkpoint_step'],distinct_trajectory_populations=True,prompt_changes=prompt_changes))
    available = [s for s in summaries if s['status'] == 'available']
    result = dict(status='completed' if len(available)==3 else 'evidence_gap', method='selected_first_token_descriptive_risk',
        progress_definition='Before this proposal:1 initial target output plus all prior committed round tokens; independently checked against cache_before-prompt_length+1 and final outputs',
        progress_bins=[f'{a}-{b}' for a,b in BINS], equal_prompt_definition='Mean of per-prompt metric over prompts with a nonempty corresponding denominator; available/32 explicitly reported',
        checkpoints=summaries, comparisons=comparisons, stable_below_half_alpha_ordinals=sorted(set.intersection(*(set(s['low_alpha_prompt_ordinals']) for s in available))) if len(available)==3 else None,
        limits=LIMITS, analyzer_sha256=sha(__file__))
    (output/'summary.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    with (output/'private-round-risk.jsonl').open('x') as stream:
        for row in private: stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root',required=True,type=Path);parser.add_argument('--collector-source',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args(argv);result=analyze(args.raw_root,args.collector_source,args.output)
    print(json.dumps(dict(status=result['status'],checkpoints=[dict(step=s['checkpoint_step'],status=s['status']) for s in result['checkpoints']])))
    return 0 if result['status']=='completed' else 1


if __name__=='__main__':raise SystemExit(main())
