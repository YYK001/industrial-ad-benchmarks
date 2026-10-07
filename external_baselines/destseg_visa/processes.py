"""Bounded subprocess monitoring; heartbeats are not training progress."""
import os
from pathlib import Path
import queue
import signal
import subprocess
import threading
import time


def stop_process(process):
    if os.name == 'posix':
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        # Also stop DataLoader children that outlived their parent.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        process.kill()
    process.wait(timeout=10)


def run_process(command, log_path, tag, cancel, *, idle_timeout=900, heartbeat=60, deadline=None):
    """Stop on missing child output, cancellation or a failed child exit."""
    log_path = Path(log_path)
    messages = queue.Queue()
    with log_path.open('a', encoding='utf-8') as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, start_new_session=(os.name == 'posix'))
        def read():
            try:
                for line in process.stdout:
                    messages.put(line)
            finally:
                process.stdout.close()
                messages.put(None)
        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        started = last_output = last_heartbeat = time.monotonic()
        exited_at = None
        eof = False
        def emit(line):
            log.write(line); log.flush()
            print(f'[{tag}] {line}', end='', flush=True)
        emit(f'process started pid={process.pid}; idle limit={idle_timeout}s\n')
        try:
            while True:
                if cancel.is_set():
                    raise RuntimeError(f'{tag}: cancelled after another stage failed')
                try:
                    line = messages.get(timeout=.2)
                    if line is None:
                        eof = True
                    else:
                        last_output = time.monotonic()
                        emit(line)
                except queue.Empty:
                    pass
                now = time.monotonic()
                if deadline is not None and now >= deadline:
                    raise TimeoutError(f'{tag}: session work deadline reached; stopping for archive')
                code = process.poll()
                if code is not None:
                    if exited_at is None:
                        exited_at = now
                    if (eof or now-exited_at > 2) and messages.empty():
                        emit(f'process exited code={code}; elapsed={now-started:.1f}s\n')
                        if code:
                            raise RuntimeError(f'{tag}: exit code {code}; inspect {log_path}')
                        if not eof:
                            stop_process(process)
                        return
                if now-last_output > idle_timeout:
                    raise TimeoutError(f'{tag}: no child output for {idle_timeout}s; inspect {log_path}')
                if now-last_heartbeat >= heartbeat:
                    emit(f'heartbeat: elapsed={now-started:.0f}s; last child output {now-last_output:.0f}s ago\n')
                    last_heartbeat = now
        except BaseException:
            cancel.set()
            stop_process(process)
            raise
        finally:
            reader.join(timeout=1)
