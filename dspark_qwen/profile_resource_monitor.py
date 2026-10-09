"""Independent stdlib diagnostic resource monitor; Linux pidfd-bound worker."""
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import tempfile
import time

LIMITS = dict(max_traces=4, trace_bytes=256*1024**2, total_trace_bytes=1024**3,
              rss_bytes=8*1024**3, monitor_seconds=.25, max_spans=100000)


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,prefix=path.name+'.',suffix='.tmp',delete=False) as stream:
        temporary=Path(stream.name);stream.write(json.dumps(value,allow_nan=False,indent=2)+'\n')
    temporary.replace(path)


def process_record(pid):
    try:
        text=Path(f'/proc/{pid}/stat').read_text();fields=text[text.rindex(')')+2:].split()
        return dict(pid=int(pid),start_ticks=int(fields[19]),state=fields[0])
    except (FileNotFoundError,ProcessLookupError):return None


def same_process(identity):
    actual=process_record(identity['pid'])
    return bool(actual and actual['start_ticks']==identity['start_ticks'] and actual['state']!='Z')


def rss_for(pid):
    return int(Path(f'/proc/{pid}/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')


def current_rss():return rss_for(os.getpid())


class ResourceProbe:
    """Diagnostic-only sampled RSS/file guard. Abort owns only this worker."""
    def __init__(self, directory, result_path, *, limits=None, rss=current_rss, fatal=os._exit, deadline=None):
        self.directory, self.result_path = Path(directory), Path(result_path)
        self.limits = dict(LIMITS if limits is None else limits)
        self.rss, self.fatal, self.deadline = rss, fatal, deadline
        self.max_rss = 0

    def inspect(self):
        rss = self.rss(); self.max_rss = max(self.max_rss, rss)
        files = list(self.directory.glob('*/trace.json*'))
        sizes = {}
        for p in files:
            try: sizes[str(p.relative_to(self.directory))] = p.stat().st_size
            except FileNotFoundError: pass  # Atomic partial -> final rename; next poll sees final.
        value = dict(rss_bytes=rss, max_rss_bytes=self.max_rss, trace_sizes=sizes,
                     trace_total_bytes=sum(sizes.values()), limits=self.limits)
        if self.deadline is not None and time.monotonic() >= self.deadline: value['failure'] = 'cooperative_deadline'
        elif rss > self.limits['rss_bytes']: value['failure'] = 'host_rss_limit'
        elif any(x > self.limits['trace_bytes'] for x in sizes.values()): value['failure'] = 'trace_size_limit'
        elif sum(sizes.values()) > self.limits['total_trace_bytes']: value['failure'] = 'trace_total_limit'
        elif len(sizes) > self.limits['max_traces']: value['failure'] = 'trace_count_limit'
        return value

    def check(self):
        try: value = self.inspect()
        except BaseException as exc: value = dict(failure='resource_monitor_error', error=repr(exc))
        if 'failure' in value:
            try:
                value['diagnostic_states'] = [json.loads(p.read_text()) for p in self.directory.glob('*/status.json')]
                write(self.directory/'resource-failure.json', value)
                try: result = json.loads(self.result_path.read_text())
                except (OSError, ValueError): result = {}
                result.update(status='failed', stage='diagnostic_resource_abort', resource_failure=value)
                write(self.result_path, result)
            except BaseException as exc:
                # Logging itself can fail (full disk, permission, concurrent I/O).
                # It must never turn a hard abort into a silently dead monitor.
                try: os.write(2, ('resource abort evidence error: '+repr(exc)+'\n').encode())
                except BaseException: pass
            finally:
                self.fatal(86)
            raise RuntimeError('Diagnostic resource abort: '+value['failure'])
        return value



class ResourceMonitor:
    """Diagnostic-only subprocess, independent of the worker's GIL/native calls."""
    def __init__(self,directory,result_path,*,state_directory,limits=None,deadline=None):
        self.directory,self.result_path=Path(directory),Path(result_path)
        self.state_directory=Path(state_directory);self.limits=dict(LIMITS if limits is None else limits)
        self.deadline=deadline;self.process=None;self.max_rss=0;self.sequence=0

    def _receive(self,timeout=5):
        ready,_,_=select.select([self.process.stdout],[],[],timeout)
        if not ready:raise RuntimeError('Independent resource monitor handshake timeout')
        line=self.process.stdout.readline()
        if not line:raise RuntimeError('Independent resource monitor exited without acknowledgement')
        value=json.loads(line)
        self.max_rss=max(self.max_rss,value.get('max_rss_bytes',0))
        return value

    def __enter__(self):
        if not sys.platform.startswith('linux') or not hasattr(os,'pidfd_open') or not hasattr(signal,'pidfd_send_signal'):
            raise RuntimeError('Independent resource monitor requires Linux /proc and pidfd support')
        identity=process_record(os.getpid())
        if not identity:raise RuntimeError('Cannot bind resource monitor worker identity')
        self.state_directory.mkdir(parents=True,exist_ok=True)
        config=dict(worker=identity,directory=str(self.directory.resolve()),result_path=str(self.result_path.resolve()),
            state_directory=str(self.state_directory.resolve()),limits=self.limits,deadline=self.deadline)
        path=self.state_directory/'monitor-config.json';write(path,config)
        self.log=(self.state_directory/'monitor-stderr.log').open('wb')
        self.process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--config',str(path.resolve())],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,bufsize=0,
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HIP_VISIBLE_DEVICES='',CUDA_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES=''))
        try:
            value=self._receive()
            if value.get('status')!='ready' or value.get('worker')!=identity:raise RuntimeError('Monitor identity/readiness mismatch')
            self.monitor_identity=value['monitor'];return self
        except BaseException:
            self._stop_failed();raise

    def _request(self,command):
        if self.process.poll() is not None:raise RuntimeError('Independent resource monitor is not alive')
        self.sequence+=1
        self.process.stdin.write((json.dumps(dict(command=command,sequence=self.sequence))+'\n').encode())
        self.process.stdin.flush();value=self._receive()
        if value.get('sequence')!=self.sequence or value.get('status')!=command:
            raise RuntimeError('Independent resource monitor acknowledgement mismatch')
        return value

    def check(self):return self._request('check')

    def _stop_failed(self):
        if self.process is not None and self.process.poll() is None:
            self.process.kill();self.process.wait(timeout=2)
        if hasattr(self,'log'):self.log.close()

    def __exit__(self,*args):
        try:
            value=self._request('stop')
            code=self.process.wait(timeout=2)
            if code!=0 or value.get('failure'):raise RuntimeError('Resource monitor did not stop cleanly')
            write(self.state_directory/'monitor-exit.json',dict(os_exit=code,stop_acknowledged=True,monitor=self.monitor_identity))
        except BaseException:
            self._stop_failed();raise
        finally:
            self.log.close()
            for stream in (self.process.stdin,self.process.stdout):stream.close()


def monitor_main(config_path):
    config=json.loads(Path(config_path).read_text());identity=config['worker'];state=Path(config['state_directory'])
    if not same_process(identity):raise RuntimeError('Resource monitor worker identity changed before pidfd')
    fd=os.pidfd_open(identity['pid'])
    if not same_process(identity):os.close(fd);raise RuntimeError('Worker identity changed during pidfd binding')
    def abort(code):
        # Both original start ticks and the kernel pidfd must still refer to the
        # live worker. Never kill a PID that merely reused its number.
        if same_process(identity):signal.pidfd_send_signal(fd,signal.SIGKILL)
    probe=ResourceProbe(config['directory'],config['result_path'],limits=config['limits'],
                        deadline=config['deadline'],rss=lambda:rss_for(identity['pid']),fatal=abort)
    observed=[];last=None;max_gap=0.;count=0
    def sample():
        nonlocal last,max_gap,count
        now=time.monotonic();gap=now-last if last is not None else 0.;last=now
        max_gap=max(max_gap,gap)
        if not same_process(identity):raise RuntimeError('Bound worker is no longer live')
        if gap>1:
            # Scheduling/disk stalls can still delay an independent process.
            # Such a run fails rather than claiming the <=1s observation contract.
            probe.rss=lambda:(_ for _ in ()).throw(RuntimeError('Monitor sampling gap exceeded one second'))
        value=probe.check();count+=1
        record=dict(monotonic=now,gap_seconds=gap,worker=identity,**value)
        with (state/'monitor-samples.jsonl').open('a') as stream:stream.write(json.dumps(record,allow_nan=False)+'\n')
        return dict(max_rss_bytes=probe.max_rss,samples=count,max_sampling_gap_seconds=max_gap)
    def reply(value):
        sys.stdout.write(json.dumps(value,allow_nan=False)+'\n');sys.stdout.flush()
    try:
        value=sample();monitor=process_record(os.getpid())
        write(state/'monitor-state.json',dict(status='ready',worker=identity,monitor=monitor,**value))
        reply(dict(status='ready',worker=identity,monitor=monitor,**value))
        while True:
            readable,_,_=select.select([sys.stdin],[],[],config['limits']['monitor_seconds'])
            # Always observe/enforce before accepting stop. An over-limit worker
            # cannot turn failure into a successful stop handshake.
            value=sample()
            if not readable:continue
            line=sys.stdin.readline()
            if not line:raise RuntimeError('Resource monitor control pipe closed while worker live')
            request=json.loads(line);command=request['command']
            if command not in ('check','stop'):raise RuntimeError('Unknown monitor command')
            if command=='stop':
                write(state/'monitor-state.json',dict(status='stopped',worker=identity,monitor=monitor,**value))
                reply(dict(status='stop',sequence=request['sequence'],**value));return 0
            reply(dict(status='check',sequence=request['sequence'],**value))
    except BaseException as exc:
        try:
            write(state/'monitor-state.json',dict(status='failed',error=repr(exc),worker=identity,max_rss_bytes=probe.max_rss))
            result_path=Path(config['result_path'])
            try:result=json.loads(result_path.read_text())
            except (OSError,ValueError):result={}
            result.update(status='failed',stage='independent_monitor_abort',error=repr(exc))
            write(result_path,result)
        except BaseException as log_error:
            try:os.write(2,('monitor evidence failure: '+repr(log_error)+'\n').encode())
            except BaseException:pass
        finally:abort(86)
        return 86
    finally:os.close(fd)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    raise SystemExit(monitor_main(args.config))
