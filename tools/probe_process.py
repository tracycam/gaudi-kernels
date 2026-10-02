"""Bounded cleanup, including children stuck in uninterruptible kernel waits.

SIGKILL is a request, not proof of process termination. Never use an unbounded
wait after it, and preserve a quarantine if the child cannot yet be reaped.
"""
import json
import os
from pathlib import Path
import signal
import subprocess


def identity(pid):
    try:
        text=Path(f'/proc/{pid}/stat').read_text()
        fields=text[text.rfind(')')+2:].split()
        try:boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        except OSError:boot_id=None
        return {'pid':pid,'start_ticks':fields[19],'boot_id':boot_id}
    except (OSError,IndexError):
        return None


def active_quarantine(path):
    """PID reuse or a host reboot must not block a new process."""
    path=Path(path)
    if not path.exists():return None
    record=json.loads(path.read_text())
    if 'processes' in record:
        if any(identity(p['pid'])==p for p in record['processes']):return record
        path.unlink()
        return None
    live=identity(record['process']['pid'])
    if live==record['process']:return record
    path.unlink()
    return None


def quarantine(path,process,reason):
    record={'process':identity(process.pid),'reason':reason}
    if record['process'] is None:return None
    path=Path(path)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(record,indent=2)+'\n')
    temporary.replace(path)
    return record


def stop(process,term_seconds=1,kill_seconds=1):
    record={'pid':process.pid,'signals':[]}
    for sig,timeout in [(signal.SIGTERM,term_seconds),(signal.SIGKILL,kill_seconds)]:
        if process.poll() is not None:break
        try:
            os.killpg(process.pid,sig)
            record['signals'].append(int(sig))
        except ProcessLookupError:
            pass
        try:process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:pass
    record.update(reaped=process.poll() is not None,returncode=process.returncode)
    return record


def query(command,output,timeout=15):
    """Return status as data so a failed preflight can be archived before exit."""
    process=None;record={'command':command,'timeout_s':timeout}
    try:
        with Path(output).open('wb') as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            record['process']=identity(process.pid)
            try:
                process.wait(timeout=timeout)
                record['reason']=None
            except subprocess.TimeoutExpired:
                record['reason']='timeout'
    except OSError as error:
        record.update(reason='spawn_failed',error=str(error))
    except (KeyboardInterrupt,SystemExit) as error:
        record.update(reason='interrupted',interrupt_code=getattr(error,'code',130))
    finally:
        if process is not None:
            record['cleanup']=stop(process) if process.poll() is None else {'reaped':True,'returncode':process.returncode}
            record['returncode']=process.returncode
    return record,process
