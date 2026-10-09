#!/usr/bin/env python3
"""CPU-only tokenize one newly authored public synthetic non-thinking prompt."""
import argparse
import hashlib
import json
from pathlib import Path

SYNTHETIC_PROMPT = 'Write a short explanation of why a library keeps an index of its books. Use plain language and include one example.'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    model, output = Path(args.model).resolve(), Path(args.output)
    if output.exists():
        raise ValueError('Fresh preparation directory required')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True, trust_remote_code=False)
    tokens = tokenizer.apply_chat_template([dict(role='user', content=SYNTHETIC_PROMPT)],
        tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
    request = dict(prompt_token_ids=tokens, seed=20261009)
    output.mkdir(parents=True)
    (output / 'request.json').write_text(json.dumps(request, indent=2) + '\n')
    model_files = {str(p): sha(p) for p in sorted(model.iterdir())
        if p.is_file() and p.suffix in ('.json', '.safetensors', '.model', '.tiktoken', '.txt')}
    record = dict(prompt=SYNTHETIC_PROMPT, prompt_kind='new_public_synthetic_not_dataset_or_final_test',
        enable_thinking=False, add_generation_prompt=True, seed=20261009, prompt_tokens=len(tokens),
        request_sha256=sha(output / 'request.json'), model_files=model_files,
        prepare_source_sha256=sha(Path(__file__)), tokenizer_class=type(tokenizer).__name__)
    (output / 'preparation.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(dict(status='cpu_prepared_only', prompt_tokens=len(tokens),
        request_sha256=record['request_sha256'], model_file_count=len(model_files))))


if __name__ == '__main__':
    main()
