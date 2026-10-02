"""Opt-in runtime observations, always stopped before timed replay.

Record environment *names and source locations*, never environment values.
Declared GC_KERNEL_PATH entries, mapped libraries, and Python imports are
separate facts. Registered GUIDs or compiled recipe nodes are not described as
actual executed kernels without corresponding device trace evidence.
"""
from collections import Counter
from functools import wraps
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys

_original_getitem = None
_reads = Counter()
_root = None
_phase = "startup"
_hash_cache = {}


def _sha(path):
    stat = path.stat()
    identity = (str(path), stat.st_size, stat.st_mtime_ns)
    if identity not in _hash_cache:
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1 << 20), b''):
                h.update(block)
        _hash_cache[identity] = h.hexdigest()
    return _hash_cache[identity]


def start(root, phase="startup"):
    global _original_getitem, _root, _phase
    if _original_getitem is not None:
        return
    _root = str(Path(root).resolve()) + '/'
    _phase = phase
    original = os._Environ.__getitem__

    def recorded_getitem(self, key):
        frame = sys._getframe(1)
        # Mapping.get()/os.getenv() intermediate frames are not the reader.
        while frame and (frame.f_code.co_filename in ('<frozen os>', '<frozen _collections_abc>')
                         or frame.f_code.co_filename.endswith(('/os.py', '/_collections_abc.py'))):
            frame = frame.f_back
        if frame and isinstance(key, str) and frame.f_code.co_filename.startswith(_root):
            _reads[(key, frame.f_code.co_filename, frame.f_lineno, _phase)] += 1
        return original(self, key)

    _original_getitem = original
    os._Environ.__getitem__ = recorded_getitem


def stop():
    global _original_getitem
    if _original_getitem is not None:
        os._Environ.__getitem__ = _original_getitem
        _original_getitem = None


@contextmanager
def observing(enabled, phase):
    global _phase
    if not enabled:
        yield
        return
    already_active=_original_getitem is not None
    previous=_phase
    start(Path(__file__).resolve().parents[1], phase)
    _phase=phase
    try:yield
    finally:
        _phase=previous
        if not already_active:stop()


def track_selection(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        from .context import context
        enabled = context().startup.inventory
        if enabled:
            start(Path(__file__).resolve().parents[1], "selection")
        try:
            return function(*args, **kwargs)
        finally:
            if enabled:
                stop()
    return wrapped


def snapshot(root, *, mapped_paths=None):
    stop()
    root = Path(root).resolve()
    imports = []
    for name, module in list(sys.modules.items()):
        file = getattr(module, '__file__', None)
        if not file:
            continue
        path = Path(file).resolve()
        if path.is_file() and path.is_relative_to(root):
            imports.append({'module': name, 'file': str(path), 'sha256': _sha(path)})
    if mapped_paths is None:
        mapped_paths = set()
        for line in Path('/proc/self/maps').read_text().splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) == 6 and fields[5].startswith('/'):
                mapped_paths.add(fields[5].removesuffix(' (deleted)'))
    declared = [str(Path(p).resolve()) for p in os.getenv('GC_KERNEL_PATH', '').split(':') if p]
    mapped = {str(Path(p).resolve()) for p in mapped_paths}
    libraries = []
    for path in declared:
        file = Path(path)
        libraries.append({'path': path, 'mapped': path in mapped,
            'sha256': _sha(file) if file.is_file() else None})
    return {
        'imports': sorted(imports, key=lambda entry: entry['module']),
        'environment_reads': [{'name': name, 'file': file, 'line': line, 'phase': phase, 'calls': calls}
            for (name, file, line, phase), calls in sorted(_reads.items())],
        'tpc_libraries': libraries,
        'tracking_active': _original_getitem is not None,
        'scope': 'Observed imports/env reads/library mappings; no inference of executed GUIDs',
    }


def write_snapshot(worker):
    from .context import context
    if not context().startup.inventory:
        return None
    result = snapshot(Path(__file__).resolve().parents[1])
    directory = context().startup.run_dir / 'runtime-inventory'
    directory.mkdir(exist_ok=True)
    path = directory / f'rank{worker.rank}.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(path)
    return {'path': str(path), 'imports': len(result['imports']),
            'environment_names': len({r['name'] for r in result['environment_reads']}),
            'tracking_active': result['tracking_active']}
