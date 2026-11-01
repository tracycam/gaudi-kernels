"""Reversible worker-owned Linux CPU utilization clamp; no global policy."""
import ctypes
import errno
import os
from pathlib import Path
import platform


class SchedAttr(ctypes.Structure):
    _fields_ = [('size', ctypes.c_uint32), ('policy', ctypes.c_uint32),
                ('flags', ctypes.c_uint64), ('nice', ctypes.c_int32),
                ('priority', ctypes.c_uint32), ('runtime', ctypes.c_uint64),
                ('deadline', ctypes.c_uint64), ('period', ctypes.c_uint64),
                ('util_min', ctypes.c_uint32), ('util_max', ctypes.c_uint32)]


def read_attr(tid):
    attr = SchedAttr(size=ctypes.sizeof(SchedAttr))
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.syscall(315, tid, ctypes.byref(attr), attr.size, 0):
        raise OSError(ctypes.get_errno(), 'sched_getattr')
    return attr


def set_clamp(tid, minimum, maximum):
    attr = read_attr(tid)
    attr.flags |= 0x20 | 0x40
    attr.util_min, attr.util_max = minimum, maximum
    if ctypes.CDLL(None, use_errno=True).syscall(314, tid, ctypes.byref(attr), 0):
        raise OSError(ctypes.get_errno(), 'sched_setattr')


def thread_ids():
    return [int(task.name) for task in Path('/proc/self/task').iterdir()]


def _parse_cpu_list(text):
    result = set()
    for part in text.strip().split(','):
        if not part:
            continue
        bounds = part.split('-')
        if len(bounds) == 1:
            result.add(int(bounds[0]))
        elif len(bounds) == 2 and int(bounds[0]) <= int(bounds[1]):
            result.update(range(int(bounds[0]), int(bounds[1]) + 1))
    return result


def primary_physical_cpus(cpus=None, *, sysfs_root=Path('/sys/devices/system/cpu')):
    """Select one primary logical CPU per physical core (stripping SMT siblings)."""
    if cpus is None:
        cpus = os.sched_getaffinity(0)
    groups = {}
    for cpu in sorted(cpus):
        topo = sysfs_root / f'cpu{cpu}/topology/thread_siblings_list'
        if topo.exists():
            siblings = tuple(sorted(_parse_cpu_list(topo.read_text())))
        else:
            siblings = (cpu,)
        groups.setdefault(siblings, set()).add(cpu)
    selected = {min(group) for group in groups.values() if group}
    return selected or set(cpus)


class WorkerQos:
    def __init__(self, minimum=1024, *, cpus=None):
        if platform.system() != 'Linux' or platform.machine() != 'x86_64':
            raise RuntimeError('Worker CPU utilization clamp requires Linux x86_64')
        if type(minimum) is not int or not 1 <= minimum <= 1024:
            raise ValueError('Worker CPU utilization minimum must be 1..1024')
        self.minimum = minimum
        self.cpus = set(cpus) if cpus is not None else None
        if self.cpus is not None and not self.cpus:
            raise ValueError('Worker CPU affinity set must be non-empty')
        attr = read_attr(os.getpid())
        self.default = (attr.util_min, attr.util_max)
        self.default_cpus = os.sched_getaffinity(os.getpid()) if self.cpus is not None else None
        self.originals = {}
        self.original_cpus = {}
        self.initial_tids = set(thread_ids())
        self.active = True
        try:
            self._apply_threads(self.initial_tids)
        except BaseException:
            self.restore()
            raise

    def _apply_threads(self, tids):
        for tid in sorted(tids, key=lambda tid: tid != os.getpid()):
            try:
                old = read_attr(tid)
                if tid not in self.originals:
                    if self.minimum > old.util_max:
                        raise ValueError('Requested worker minimum exceeds existing CPU utilization cap')
                    self.originals[tid] = (old.util_min, old.util_max)
                if self.cpus is not None and tid not in self.original_cpus:
                    self.original_cpus[tid] = os.sched_getaffinity(tid)
                set_clamp(tid, self.minimum, self.originals[tid][1])
                if self.cpus is not None:
                    os.sched_setaffinity(tid, self.cpus)
            except OSError as error:
                if error.errno not in (errno.ENOENT, errno.ESRCH):
                    raise

    def refresh(self):
        """Re-apply uclamp and CPU affinity to threads spawned after startup (e.g. Synapse/HCCL)."""
        if not self.active:
            return 0
        current = set(thread_ids())
        self._apply_threads(current)
        return len(current)

    def restore(self):
        if not self.active:
            return
        errors = []
        for tid in thread_ids():
            if tid in self.initial_tids and tid not in self.originals:
                continue
            # Threads created after startup inherit the worker's clamp. Restore
            # their parent's original setting as well as the initial threads.
            minimum, maximum = self.originals.get(tid, self.default)
            try:
                set_clamp(tid, minimum, maximum)
                if self.cpus is not None:
                    os.sched_setaffinity(tid, self.original_cpus.get(tid, self.default_cpus))
            except OSError as error:
                if error.errno not in (errno.ENOENT, errno.ESRCH):
                    errors.append(error)
        self.active = False
        if errors:
            raise errors[0]


def apply_optimal_cpu_qos(minimum=1024, *, physical_cores=True):
    """Apply optimal uclamp_min=1024 and NUMA-local primary physical core affinity to current process."""
    available = os.sched_getaffinity(0)
    local_cpus = set(available)
    mod_text = os.environ.get('HLS_MODULE_ID')
    if mod_text is not None and mod_text.isdigit():
        try:
            from gaudi_kernels.serving.host_placement import discover
            from gaudi_kernels.engine.config import HostPlacement
            mod_id = int(mod_text)
            plan = discover(HostPlacement(numa_binding='local', visible_modules=(mod_id,)))
            candidate = set(plan['workers'][0]['allowed_local_cpus']) & available
            if candidate:
                local_cpus = candidate
        except Exception:
            pass
    target_cpus = primary_physical_cpus(local_cpus) if physical_cores else local_cpus
    return WorkerQos(minimum, cpus=target_cpus)
