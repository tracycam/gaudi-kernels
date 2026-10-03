"""Discover module topology and use vLLM's pre-spawn CPU/memory binding."""
import json
import os
from pathlib import Path
import subprocess
import inspect
from functools import wraps

from gaudi_kernels.engine.config import ConfigError


def cpu_set(text):
    result = set()
    for part in text.strip().split(','):
        if not part:
            continue
        bounds = part.split('-')
        if len(bounds) == 1:
            result.add(int(bounds[0]))
        elif len(bounds) == 2 and int(bounds[0]) <= int(bounds[1]):
            result.update(range(int(bounds[0]), int(bounds[1])+1))
        else:
            raise ConfigError('Invalid CPU range: '+part)
    return result


def discover(host, *, pci_root=Path('/sys/bus/pci/devices'), query=None):
    if query is None:
        query = subprocess.check_output(['hl-smi', '--query-aip=module_id,bus_id',
                                         '--format=csv,noheader'], text=True, timeout=15)
    modules = {}
    for line in query.strip().splitlines():
        module, bus = [v.strip() for v in line.split(',')]
        if int(module) in modules:
            raise ConfigError('Duplicate HPU module in topology')
        modules[int(module)] = bus
    available = os.sched_getaffinity(0)
    workers = []
    for rank, module in enumerate(host.visible_modules):
        if module not in modules:
            raise ConfigError('HPU module absent: '+str(module))
        pci = pci_root/modules[module]
        node = int((pci/'numa_node').read_text())
        cpus = cpu_set((pci/'local_cpulist').read_text()) & available
        if node < 0 or not cpus:
            raise ConfigError('No usable CPU/NUMA locality for module '+str(module))
        workers.append({'local_rank': rank, 'module_id': module, 'bus_id': modules[module],
                        'numa_node': node, 'allowed_local_cpus': sorted(cpus)})
    return {'binding': host.numa_binding, 'workers': workers,
            'launcher_cpus': sorted(available),
            'scope': 'PCI sysfs locality keyed by module ID, not hl-smi index'}


def prepare(runtime):
    plan = discover(runtime.selection.engine.runtime.host)
    if plan['binding'] == 'local':
        for node in sorted({w['numa_node'] for w in plan['workers']}):
            # vLLM can fall back to CPU-only binding; reject that silently
            # weakened placement before loading the model.
            subprocess.run(['numactl', f'--cpunodebind={node}', f'--membind={node}', 'true'],
                           check=True, timeout=10)
    (runtime.startup.run_dir/'host-placement.json').write_text(json.dumps(plan, indent=2)+'\n')
    return plan


def vllm_kwargs():
    from gaudi_kernels.engine.context import context
    runtime = context()
    kwargs = {'max_num_batched_tokens':runtime.selection.engine.runtime.prefill_chunk_tokens,
              'enable_chunked_prefill':True}
    if runtime.selection.engine.runtime.runner == 'native':
        kwargs['worker_cls'] = 'gaudi_kernels.serving.worker.NativeHPUWorker'
    if runtime.selection.engine.runtime.native_enabled:
        kwargs['async_scheduling']=False
    if runtime.selection.engine.runtime.host.numa_binding == 'none':
        return kwargs
    plan = json.loads((runtime.startup.run_dir/'host-placement.json').read_text())
    return kwargs | {'numa_bind': True, 'numa_bind_nodes': [w['numa_node'] for w in plan['workers']]}


def install_worker_spawn(module):
    original = module.WorkerProc.make_worker_process
    signature = inspect.signature(original)
    @wraps(original)
    def spawn(*args, **kwargs):
        from gaudi_kernels.engine.context import context
        rank = signature.bind(*args, **kwargs).arguments['local_rank']
        modules = context().selection.engine.runtime.host.visible_modules
        previous = os.environ.get('HLS_MODULE_ID')
        os.environ['HLS_MODULE_ID'] = str(modules[rank])
        try:
            return original(*args, **kwargs)
        finally:
            if previous is None:
                os.environ.pop('HLS_MODULE_ID', None)
            else:
                os.environ['HLS_MODULE_ID'] = previous
    module.WorkerProc.make_worker_process = staticmethod(spawn)


def snapshot(worker):
    from gaudi_kernels.engine.context import context
    runtime = context()
    plan = json.loads((runtime.startup.run_dir/'host-placement.json').read_text())
    expected = plan['workers'][worker.local_rank]
    threads = []
    for task in sorted(Path('/proc/self/task').iterdir()):
        try:
            status = dict(line.split(':', 1) for line in (task/'status').read_text().splitlines() if ':' in line)
            threads.append({'tid': int(task.name), 'name': status['Name'].strip(),
                            'cpus': status['Cpus_allowed_list'].strip(),
                            'memory_nodes': status['Mems_allowed_list'].strip()})
        except FileNotFoundError:
            continue
    locality = {'anon': {}, 'file': {}}
    policies = {}
    mappings = {}
    for line in Path('/proc/self/numa_maps').read_text().splitlines():
        parts = line.split()
        policies[parts[1]] = policies.get(parts[1], 0)+1
        kind = 'file' if any(p.startswith('file=') for p in parts) else 'anon'
        pages_at_nodes = {}
        for part in parts:
            if part.startswith('N') and '=' in part and part[1:part.index('=')].isdigit():
                node, pages = part.split('=')
                locality[kind][node] = locality[kind].get(node, 0)+int(pages)
                pages_at_nodes[node] = int(pages)
        mappings[int(parts[0], 16)] = {'policy': parts[1], 'pages': pages_at_nodes}
    import ctypes
    lib = ctypes.CDLL('libnuma.so.1', use_errno=True)
    # get_mempolicy(NULL,NULL,0,NULL,0) is invalid; query the mode only.
    mode = ctypes.c_int()
    rc = lib.get_mempolicy(ctypes.byref(mode), None, 0, None, 0)
    api = ctypes.CDLL(None)
    api.e1_host_buffer_addresses.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint32]
    api.e1_host_buffer_addresses.restype = ctypes.c_uint32
    amount = api.e1_host_buffer_addresses(None, 0)
    addresses = (ctypes.c_uint64*amount)()
    api.e1_host_buffer_addresses(addresses, amount)
    regions = []
    for line in Path('/proc/self/maps').read_text().splitlines():
        start, end = [int(v, 16) for v in line.split()[0].split('-')]
        regions.append((start, end))
    pinned = []
    for address in addresses:
        region = next((start for start, end in regions if start <= address < end), None)
        pinned.append({'address': address, 'mapping': mappings.get(region)})
    result = {'rank': worker.rank, 'local_rank': worker.local_rank, 'pid': os.getpid(),
              'module_id': os.environ.get('HLS_MODULE_ID'), 'expected': expected, 'threads': threads,
              'memory_policy_mode': mode.value if rc == 0 else None,
              'memory_policy_errno': ctypes.get_errno() if rc else 0,
              'mapping_policies': policies, 'resident_pages': locality, 'pinned_staging': pinned}
    devices = []
    for fd in Path('/proc/self/fd').iterdir():
        try:
            target = fd.readlink()
            if target.parent == Path('/dev/accel') and target.name.startswith('accel') and target.name[5:].isdigit():
                pci = (Path('/sys/class/accel')/target.name/'device').resolve()
                devices.append({'device': str(target), 'bus_id': pci.name,
                                'numa_node': int((pci/'numa_node').read_text())})
        except FileNotFoundError:
            continue
    result['device_fds'] = devices
    result['device_match'] = bool(devices) and {d['bus_id'] for d in devices} == {expected['bus_id']}
    result['all_threads_local'] = bool(threads) and all(cpu_set(t['cpus']) <= set(expected['allowed_local_cpus']) for t in threads)
    result['module_match'] = result['module_id'] == str(expected['module_id'])
    path = runtime.startup.run_dir/f'host-placement-rank{worker.rank}.json'
    path.write_text(json.dumps(result, indent=2)+'\n')
    if not result['device_match'] or not result['module_match']:
        raise RuntimeError('Worker module/PCI identity mismatch: '+str(worker.rank))
    if plan['binding'] == 'local' and (not result['all_threads_local'] or result['memory_policy_mode'] != 2):
        raise RuntimeError('Worker CPU/module/memory binding verification failed: '+str(worker.rank))
    return result
