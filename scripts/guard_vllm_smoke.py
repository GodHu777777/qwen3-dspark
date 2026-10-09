#!/usr/bin/env python3
"""Linux stdlib supervisor for one frozen vLLM smoke, with no automatic retry."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import urllib.request


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def process_record(pid):
    try:
        text = Path(f'/proc/{pid}/stat').read_text()
        fields = text[text.rindex(')') + 2:].split()
        return dict(pid=int(pid), state=fields[0], ppid=int(fields[1]),
                    pgrp=int(fields[2]), start_ticks=int(fields[19]))
    except (FileNotFoundError, ProcessLookupError):
        return None


def same_process(record, *, live=False):
    current = process_record(record['pid'])
    return bool(current and current['start_ticks'] == record['start_ticks'] and
                (not live or current['state'] != 'Z'))


def enable_subreaper():
    if not sys.platform.startswith('linux'):
        raise RuntimeError('Linux /proc supervisor required')
    if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'Cannot enable child subreaper')


def discover_owned(owned, root_pid):
    records = []
    for entry in Path('/proc').iterdir():
        if entry.name.isdigit():
            record = process_record(int(entry.name))
            if record:
                records.append(record)
    changed = True
    while changed:
        changed = False
        for record in records:
            pid = record['pid']
            if pid in owned or pid == os.getpid():
                continue
            # As subreaper, orphaned descendants are adopted by this supervisor.
            parent = owned.get(record['ppid'])
            if record['ppid'] == os.getpid() or (parent is not None and same_process(parent, live=True)):
                owned[pid] = record
                changed = True


def cleanup(owned, worker, *, grace=10.):
    actions = []
    # Signal only observed identities; never broad executable-name matching.
    for sig, duration in ((signal.SIGTERM, grace), (signal.SIGKILL, 2.)):
        discover_owned(owned, worker.pid)
        for record in tuple(owned.values()):
            if same_process(record, live=True):
                try:
                    os.kill(record['pid'], sig)
                    actions.append(dict(pid=record['pid'], start_ticks=record['start_ticks'], signal=int(sig)))
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            worker.poll()  # Popen owns and records its actual child wait status.
            discover_owned(owned, worker.pid)
            for pid, record in tuple(owned.items()):
                if pid != worker.pid:
                    try:
                        os.waitpid(pid, os.WNOHANG)
                    except ChildProcessError:
                        pass
            if not any(same_process(record) for record in owned.values()):
                return actions, []
            time.sleep(.05)
    return actions, [record for record in owned.values() if same_process(record)]


class ASRGuard:
    def __init__(self, pid, url, *, pre_free_bytes=10 * 1024**3, running_free_bytes=2 * 1024**3):
        self.pid, self.url = pid, url
        self.identity = process_record(pid)
        if not self.identity or self.identity['state'] == 'Z':
            raise RuntimeError('Preserved ASR is absent')
        self.pre_free_bytes, self.running_free_bytes = pre_free_bytes, running_free_bytes

    def snapshot(self, owned, *, before=False):
        if not same_process(self.identity, live=True):
            raise RuntimeError('Preserved ASR process identity changed')
        with urllib.request.urlopen(self.url, timeout=3) as response:
            health = json.load(response)
        owners = subprocess.run(['fuser', '/dev/kfd'], capture_output=True, text=True, timeout=3)
        if owners.returncode not in (0, 1):
            raise RuntimeError('Cannot inspect KFD owners')
        pids = {int(x) for x in owners.stdout.split()}
        allowed = {self.pid} | {p for p, r in owned.items() if same_process(r)}
        memory = []
        for path in Path('/sys/class/drm').glob('card*/device/mem_info_vram_total'):
            total = int(path.read_text())
            if total > 0:
                used = int((path.parent / 'mem_info_vram_used').read_text())
                memory.append(dict(total_bytes=total, used_bytes=used, free_bytes=total-used))
        result = dict(utc=time.time(), asr=health, asr_identity=self.identity,
            kfd_owners=sorted(pids), vram=memory)
        if not health.get('ready') or health.get('busy') or pids - allowed or self.pid not in pids:
            raise RuntimeError('ASR health or exclusive owned KFD guard failed: ' + json.dumps(result))
        floor = self.pre_free_bytes if before else self.running_free_bytes
        if not memory or min(x['free_bytes'] for x in memory) < floor:
            raise RuntimeError('VRAM headroom guard failed: ' + json.dumps(result))
        return result


def supervise(command, output, guard, *, timeout=600., poll_seconds=5., grace=10., env=None):
    """Testable process lifecycle; guard is injected for CPU-only timeout tests."""
    enable_subreaper()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    owned, worker = {}, None
    status, error, timed_out, returncode = 'failed', None, False, None
    actions, remaining = [], []
    started = time.monotonic()
    try:
        write(output / 'prestate.json', guard.snapshot({}, before=True))
        with (output / 'worker.log').open('xb') as log:
            worker = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, env=env)
            record = process_record(worker.pid)
            if record:
                owned[worker.pid] = record
            write(output / 'launch.json', dict(controller=process_record(os.getpid()), worker=record,
                command=command, deadline_seconds=timeout, cleanup_grace_seconds=grace))
            deadline, next_health = time.monotonic() + timeout, 0.
            while True:
                discover_owned(owned, worker.pid)
                write(output / 'owned-processes.json', list(owned.values()))
                returncode = worker.poll()
                if time.monotonic() >= deadline:
                    timed_out, status = True, 'timeout'
                    break
                if returncode is not None:
                    status = 'completed' if returncode == 0 else 'worker_failed'
                    break
                if time.monotonic() >= next_health:
                    snapshot = guard.snapshot(owned)
                    with (output / 'health.jsonl').open('a') as stream:
                        stream.write(json.dumps(snapshot) + '\n')
                    next_health = time.monotonic() + poll_seconds
                time.sleep(min(.1, max(0., deadline-time.monotonic())))
    except BaseException:
        error = traceback.format_exc()
        (output / 'controller-error.txt').write_text(error)
    finally:
        if worker is not None:
            actions, remaining = cleanup(owned, worker, grace=grace)
            if worker.poll() is not None:
                returncode = worker.wait()  # Actual OS wait, not worker JSON.
            write(output / 'worker-exit.json', dict(returncode=returncode, timed_out=timed_out))
        release = None
        try:
            release = guard.snapshot({}, before=False)
            write(output / 'poststate.json', release)
        except BaseException:
            error = (error or '') + traceback.format_exc()
            (output / 'release-error.txt').write_text(traceback.format_exc())
        released = not remaining and release is not None
        if error or not released or returncode is None:
            status = 'failed'
        result = dict(status=status, worker_os_exit=returncode, timed_out=timed_out,
            released=released, remaining_owned=remaining, cleanup_actions=actions,
            owned_processes=list(owned.values()), elapsed_seconds=time.monotonic()-started)
        write(output / 'controller-result.json', result)
    return result


def verify_binding(binding):
    required = {'files', 'worker', 'python', 'model', 'request', 'asr_pid', 'asr_url'}
    if set(binding) != required:
        raise ValueError('Exact frozen binding fields required')
    for path, expected in binding['files'].items():
        if sha(path) != expected:
            raise ValueError('Frozen input/source identity mismatch: ' + path)
    for path in (binding['worker'], binding['request'], str(Path(__file__).resolve())):
        if path not in binding['files']:
            raise ValueError('Worker/request/controller must be included in frozen hashes')
    model = Path(binding['model'])
    model_files = {str(p) for p in model.iterdir()
        if p.is_file() and p.suffix in ('.json', '.safetensors', '.model', '.tiktoken', '.txt')}
    # Same config+weights minimum as rollout_protocol, plus tokenizer and every
    # indexed shard: this standalone supervisor never imports model packages.
    required_names = {'config.json', 'tokenizer_config.json', 'tokenizer.json'}
    if not all((model / name).is_file() for name in required_names):
        raise ValueError('Complete model config/tokenizer files required')
    index = model / 'model.safetensors.index.json'
    if index.exists():
        shards = set(json.loads(index.read_text())['weight_map'].values())
        if not shards or any(Path(name).name != name or not name.endswith('.safetensors') or
                             not (model / name).is_file() for name in shards):
            raise ValueError('All indexed weight shards required')
    elif not (model / 'model.safetensors').is_file():
        raise ValueError('Actual model weights required; config-only fingerprint is insufficient')
    if not model_files <= set(binding['files']):
        raise ValueError('Complete model/tokenizer file manifest must be included in frozen hashes')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binding', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    binding = json.loads(Path(args.binding).read_text())
    verify_binding(binding)
    if not args.execute:
        print(json.dumps(dict(status='prepared_only_no_gpu', binding_sha256=sha(args.binding))))
        return 0
    root = Path(args.output)
    command = [binding['python'], '-u', binding['worker'], '--model', binding['model'],
        '--request', binding['request'], '--output', str(root / 'worker'), '--execute']
    env = dict(os.environ, OMP_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONPATH', None)
    for name in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'):
        env.pop(name, None)
    guard = ASRGuard(binding['asr_pid'], binding['asr_url'])
    result = supervise(command, root, guard, env=env)
    try:
        verify_binding(binding)
        write(root / 'post-input-integrity.json', dict(passed=True, binding_sha256=sha(args.binding)))
    except BaseException:
        (root / 'post-integrity-error.txt').write_text(traceback.format_exc())
        return 1
    if result['status'] != 'completed' or not result['released']:
        return 1
    worker_result = json.loads((root / 'worker/result.json').read_text())
    return 0 if worker_result['status'] == 'completed' and worker_result['output_tokens'] == 128 else 1


if __name__ == '__main__':
    raise SystemExit(main())
