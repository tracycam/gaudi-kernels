"""Reversible per-worker CPU experiments; no global governor/device changes."""
import ctypes
import errno
import json
import os
import platform
from pathlib import Path
from gaudi_kernels.serving.host_qos import SchedAttr


def submit_threads(trace_path, pid):
    """Select observed execution owners, rather than guessing thread names."""
    events = json.loads(Path(trace_path).read_text())['traceEvents']
    tids = {int(e['tid']) for e in events
            if e.get('name') == 'NativePlanSubmit' and e.get('ph') == 'B'
            and str(e.get('pid')) == str(pid)}
    if not tids:
        raise ValueError('Trace contains no native submissions from this worker')
    return {pid, *tids}


def configure(worker, mode=None, traces=None):
    if mode not in (None, 'affinity', 'utilization', 'critical', 'physical') or platform.machine() != 'x86_64':
        raise ValueError('Explicit Linux x86_64 CPU diagnostic mode required')
    from gaudi_kernels.engine.context import context
    plan = json.loads((context().startup.run_dir/'host-placement.json').read_text())
    expected = plan['workers'][worker.local_rank]
    local = set(expected['allowed_local_cpus'])
    libc = ctypes.CDLL(None, use_errno=True)

    def get(tid):
        attr = SchedAttr(size=ctypes.sizeof(SchedAttr))
        if libc.syscall(315, tid, ctypes.byref(attr), attr.size, 0):
            raise OSError(ctypes.get_errno(), 'sched_getattr')
        return attr

    def clamp(tid, minimum, maximum):
        attr = get(tid)
        attr.flags |= 0x20 | 0x40  # Preserve existing flags; update clamp only.
        attr.util_min, attr.util_max = minimum, maximum
        if libc.syscall(314, tid, ctypes.byref(attr), 0):
            raise OSError(ctypes.get_errno(), 'sched_setattr')

    state = worker.__dict__.setdefault('_cpu_policy_diagnostic', {
        'threads': {}, 'mode': None,
        'default_clamp': (get(os.getpid()).util_min, get(os.getpid()).util_max)})
    if traces is not None:
        selected = submit_threads(traces[str(worker.rank)], os.getpid())
        if any(not (Path('/proc/self/task')/str(tid)).exists() for tid in selected):
            raise ValueError('Observed submission thread is no longer alive')
        state['critical_threads'] = selected
    critical = state.get('critical_threads', set())
    if mode == 'critical' and not critical:
        raise ValueError('Critical-thread policy requires an in-process submission trace')
    if mode == 'critical' and any(not (Path('/proc/self/task')/str(tid)).exists() for tid in critical):
        raise ValueError('Submission thread changed; collect a new trace')
    peers = [x for x in plan['workers'] if x['numa_node'] == expected['numa_node']]
    groups = {}
    for cpu in sorted(local):
        siblings = tuple(int(v) for v in (Path('/sys/devices/system/cpu')/
            f'cpu{cpu}/topology/thread_siblings_list').read_text().strip().split(','))
        groups.setdefault(siblings, set()).add(cpu)
    physical = sorted(groups.values(), key=min)
    primary_cpus = {min(group) for group in physical}
    index = next(i for i,x in enumerate(peers) if x['local_rank'] == worker.local_rank)
    partition = set().union(*physical[index::len(peers)])
    if not partition:
        raise ValueError('No disjoint physical-core partition')
    touched = []
    try:
        for task in Path('/proc/self/task').iterdir():
            tid = int(task.name)
            try:
                if tid not in state['threads']:
                    attr = get(tid)
                    # Threads born inside an experimental arm inherit its
                    # settings. Restore them to the worker's initial contract.
                    state['threads'][tid] = ((local, *state['default_clamp']) if state['mode']
                                            else (os.sched_getaffinity(tid), attr.util_min, attr.util_max))
                original, minimum, maximum = state['threads'][tid]
                cpus = partition if mode == 'affinity' else primary_cpus if mode == 'physical' else original
                os.sched_setaffinity(tid, cpus)
                boost = mode in ('utilization', 'physical') or (mode == 'critical' and tid in critical)
                clamp(tid, 1024 if boost else minimum, maximum)
                touched.append(tid)
            except OSError as error:
                if error.errno != errno.ESRCH:
                    raise
    except BaseException:
        for tid in touched:
            try:
                original, minimum, maximum = state['threads'][tid]
                os.sched_setaffinity(tid, original); clamp(tid, minimum, maximum)
            except OSError as error:
                if error.errno != errno.ESRCH:
                    raise
        raise
    state['mode'] = mode
    return dict(rank=worker.rank, mode=mode, threads=len(touched),
                critical_threads=sorted(critical),
                boosted_threads=sorted(touched if mode in ('utilization', 'physical') else critical if mode == 'critical' else []),
                cpus=sorted(partition if mode == 'affinity' else primary_cpus if mode == 'physical' else local),
                scope='Per-thread affinity/uclamp only, restored between independent ABBA arms')
