"""Frozen expanded-development grouping and native stochastic protocol (stdlib only)."""
import argparse
import hashlib
import json
import unicodedata
from pathlib import Path

from .eval_stochastic_gate import sha256, write_json

SALT = 'dspark-expanded-dev-sts-v1'
POLICY = 'float64_softmax_normalize_cdf_v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def prompt_seed(record_id, base_seed=20261009):
    return (base_seed + int(hashlib.sha256(record_id.encode()).hexdigest(), 16)) % (2**63)


def read_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    seen, prompts, text_prompts = set(), set(), set()
    for row in rows:
        rid = row['id']
        if not isinstance(rid, str) or not rid or rid in seen:
            raise ValueError('Missing or duplicate record ID')
        seen.add(rid)
        if row['split'] not in ('train', 'validation'):
            raise ValueError('Final test or unknown split is forbidden')
        if type(row['accepted']) is not bool:
            raise ValueError('accepted must be a boolean')
        tokens = row['prompt_token_ids']
        if not tokens or any(type(t) is not int or t < 0 for t in tokens):
            raise ValueError('Invalid prompt token IDs')
        key = digest(tokens)
        if key in prompts:
            raise ValueError('Duplicate actual prompt token identity')
        prompts.add(key)
        if 'messages' in row:
            users = [m['content'] for m in row['messages'] if m['role'] == 'user']
            # Dataset identities are normalized first-user prompts. Do not trust ID alone.
            if not users or not isinstance(users[0], str):
                raise ValueError('Missing first user prompt')
            key = ' '.join(unicodedata.normalize('NFKC', users[0]).split())
            if key in text_prompts:
                raise ValueError('Duplicate normalized prompt text identity')
            text_prompts.add(key)
    dev = [row for row in rows if row['split'] == 'validation' and row['accepted']]
    if len(dev) != 119:
        raise ValueError('Require exactly 119 accepted development prompts; no filtering/replacement')
    return dev


def input_identity(config):
    generation_path = Path(config['generation_manifest'])
    summary_path = generation_path.with_name('summary.json')
    generation = json.loads(generation_path.read_text())
    summary = json.loads(summary_path.read_text())
    records_hash = sha256(config['records'])
    if records_hash != summary['output_sha256']['records.jsonl']:
        raise ValueError('Development records differ from generation summary')
    fingerprint = generation['model_files_sha256']
    if ('config.json' not in fingerprint or not any(n.endswith('.safetensors') for n in fingerprint)):
        raise ValueError('Target fingerprint must include config and weights')
    for name, expected in fingerprint.items():
        if Path(name).is_absolute() or '..' in Path(name).parts or sha256(Path(config['model']) / name) != expected:
            raise ValueError('Target/tokenizer fingerprint mismatch')
    return dict(development_records_sha256=records_hash, target_fingerprint=fingerprint,
                generation_manifest_sha256=sha256(generation_path), generation_summary_sha256=sha256(summary_path))


def build_manifest(config, base_seed=20261009):
    if type(base_seed) is not int or not 0 <= base_seed < 2**63:
        raise ValueError('base_seed must be a nonnegative 63-bit integer')
    identity = input_identity(config)
    dev = read_rows(config['records'])
    quality = dev[:32]
    rest = sorted(dev[32:], key=lambda r: (hashlib.sha256((SALT + '\0' + r['id']).encode()).hexdigest(), r['id']))
    groups = dict(quality=quality, fit=rest[:44], eval=rest[44:])
    block_size = config['draft']['block_size']
    if type(block_size) is not int or not 1 <= block_size <= 16:
        raise ValueError('Require block_size 1..16')
    manifest = dict(version=1, identity=identity, salt=SALT, base_seed=base_seed,
        protocol=dict(probability_policy=POLICY, temperature=1.0, filtering='none', max_new_tokens=128,
            block_size=block_size, collection_policy='full_proposal', sampling_mode='stochastic',
            target_forward_dtype='bfloat16', draft_parameters='float32', draft_autocast='bfloat16',
            backend='sdpa', admission='full block capped only by remaining output budget'),
        groups={name: [dict(id=r['id'], record_sha256=digest(r), prompt_sha256=digest(r['prompt_token_ids']),
                           seed=prompt_seed(r['id'], base_seed)) for r in rows] for name, rows in groups.items()},
        tv_probes=[dict(id=r['id'], round=0) for r in quality[:2]],
        scope='Quality selects checkpoint. STS eval is prompt-held-out from fit only; all dev used in teacher-forced monitoring. Final test locked.')
    manifest['manifest_sha256'] = digest(manifest)
    return manifest


def verify_manifest(manifest, config):
    expected = build_manifest(config, manifest['base_seed'])
    if manifest != expected:
        raise ValueError('Frozen panel/protocol/data identity changed')
    return {r['id']: r for r in read_rows(config['records'])}


def prefix_evidence(proposed, verified, accepted, rejected_index, accepted_eos_position):
    """Attempted uniforms and effective cumulative-prefix labels are different populations."""
    if any(type(x) is not int for x in (proposed, verified, accepted)) or not 0 <= accepted <= verified <= proposed:
        raise ValueError('Invalid proposal/verification/acceptance lengths')
    if rejected_index is not None and (rejected_index != accepted or rejected_index >= verified):
        raise ValueError('Rejection must be the first unaccepted verified position')
    if accepted_eos_position is not None and (accepted_eos_position != accepted - 1 or accepted == 0 or rejected_index is not None):
        raise ValueError('Accepted EOS must end the accepted draft prefix')
    if accepted < verified and rejected_index is None and accepted_eos_position is None:
        raise ValueError('Unaccepted verified tail requires rejection or accepted EOS')
    effective = verified if accepted_eos_position is None else accepted_eos_position + 1
    return dict(proposed_positions=proposed, verified_positions=verified,
                attempted_positions=accepted + int(rejected_index is not None),
                effective_positions=effective, prefix_labels=[int(j < accepted) for j in range(effective)])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--base-seed', type=int, default=20261009)
    args = parser.parse_args(argv)
    manifest = build_manifest(json.loads(Path(args.config).read_text()), args.base_seed)
    # Exclusive creation prevents silently replacing a previously frozen panel.
    with Path(args.output).open('x') as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(manifest_sha256=manifest['manifest_sha256'], groups={k: len(v) for k,v in manifest['groups'].items()})))


if __name__ == '__main__':
    main()
