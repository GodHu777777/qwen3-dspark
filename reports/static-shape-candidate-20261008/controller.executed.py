"""One authorized pause of one identity-checked generator; always resume it."""
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import time

PID = 3405911
EXPECTED_START = "60852082"
GENERATOR = "/home/nju/services/dspark-qwen/scripts/generate.py"
GENERATED = "/home/nju/data/dspark-qwen/expand-20261009/generated"
GEN_LOG = Path("/home/nju/data/dspark-qwen/expand-20261009/generation.log")
OUT = Path("/home/nju/data/dspark-qwen/static-shape-gpu-candidate-20261008-v1")
OUT.mkdir(parents=True, exist_ok=False)
record = {"generator_pid": PID, "expected_starttime": EXPECTED_START,
          "scope": "one authorized pause; diagnostic is not a throughput benchmark"}
child = None
stopped = False
pidfd = None

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save():
    temp = OUT / "controller.tmp"
    temp.write_text(json.dumps(record, indent=2) + "\n")
    temp.replace(OUT / "controller.json")
def identity():
    fields = Path(f"/proc/{PID}/stat").read_text().rsplit(")", 1)[1].split()
    cmd = Path(f"/proc/{PID}/cmdline").read_bytes().decode().split("\0")
    return {"starttime": fields[19], "state": fields[0], "cmdline": cmd}
def gpu():
    result = subprocess.run(["rocm-smi", "--showuse", "--showmeminfo", "vram", "--json"],
        capture_output=True, text=True, timeout=5, check=True)
    data = json.loads(result.stdout[result.stdout.index("{"):])["card0"]
    return {"busy_percent": int(data["GPU use (%)"]),
        "total_bytes": int(data["VRAM Total Memory (B)"]),
        "used_bytes": int(data["VRAM Total Used Memory (B)"])}

try:
    record["before"] = identity()
    assert record["before"]["starttime"] == EXPECTED_START, "PID starttime changed"
    assert GENERATOR in record["before"]["cmdline"] and GENERATED in record["before"]["cmdline"], "Wrong command/output"
    assert record["before"]["state"] not in ("T", "t", "Z"), "Generator already stopped or dead"
    pidfd = os.pidfd_open(PID)
    assert identity()["starttime"] == EXPECTED_START
    record["log_size_before"] = GEN_LOG.stat().st_size
    record["pause_started_utc"] = now()
    pause_start = time.monotonic()
    # Mark before sending so any subsequent exception still attempts SIGCONT.
    stopped = True
    signal.pidfd_send_signal(pidfd, signal.SIGSTOP)
    for _ in range(50):
        state = identity()["state"]
        if state in ("T", "t"): break
        time.sleep(0.1)
    else: raise RuntimeError("Generator did not reach stopped state")
    record["stopped_state"] = state
    record["drain_samples"] = []
    quiet = 0
    for _ in range(8):
        sample = gpu()
        record["drain_samples"].append(sample)
        quiet = quiet + 1 if sample["busy_percent"] <= 5 else 0
        if quiet >= 2: break
        time.sleep(1)
    if quiet < 2: raise RuntimeError("GPU did not drain; cancelling diagnostic")
    if sample["total_bytes"] - sample["used_bytes"] < 6 * 1024**3:
        raise RuntimeError("Less than 6 GiB free; cancelling diagnostic")
    save()
    env = os.environ.copy()
    env.update(PYTHONPATH="/home/nju/services/dspark-qwen", OMP_NUM_THREADS="4")
    command = ["/home/nju/vllm-env/bin/python", "/tmp/dspark-static-shape-candidate/gpu_probe.py",
        "--failure-report", "/home/nju/data/dspark-qwen/cached-decode-gate-20261008/result.json",
        "--output", str(OUT / "result.json")]
    record["diagnostic_command"] = command
    with (OUT / "diagnostic.log").open("w") as log:
        child = subprocess.Popen(command, cwd="/home/nju/services/dspark-qwen", env=env,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        record["diagnostic_pid"] = child.pid
        save()
        try:
            record["diagnostic_exit"] = child.wait(timeout=210)
        except subprocess.TimeoutExpired:
            record["diagnostic_timeout_seconds"] = 210
            os.killpg(child.pid, signal.SIGKILL)
            record["diagnostic_exit"] = child.wait(timeout=10)
except BaseException as exc:
    record["controller_error"] = repr(exc)
finally:
    try:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=10)
    except BaseException as exc:
        record["diagnostic_cleanup_error"] = repr(exc)
    if stopped and pidfd is not None:
        try:
            signal.pidfd_send_signal(pidfd, signal.SIGCONT)
            record["resumed_utc"] = now()
            record["pause_seconds"] = time.monotonic() - pause_start
        except BaseException as exc:
            record["resume_error"] = repr(exc)
    if pidfd is not None: os.close(pidfd)
    save()

# Read-only confirmation after resuming. Never restart any process.
if stopped and "resume_error" not in record:
    observed = []
    for _ in range(50):
        item = identity()
        assert item["starttime"] == EXPECTED_START
        observed.append(item["state"])
        if item["state"] == "R": break
        time.sleep(0.2)
    record["states_after_resume"] = observed
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        size = GEN_LOG.stat().st_size
        if size > record["log_size_before"]:
            record["log_size_after"] = size
            record["log_progress_confirmed_utc"] = now()
            break
        time.sleep(1)
    else:
        record["log_progress_not_yet_observed"] = True
record["controller_finished_utc"] = now()
save()
print(json.dumps({k: v for k, v in record.items() if k in ("diagnostic_exit", "controller_error", "resume_error",
    "pause_seconds", "states_after_resume", "log_size_before", "log_size_after", "log_progress_not_yet_observed")}), flush=True)
