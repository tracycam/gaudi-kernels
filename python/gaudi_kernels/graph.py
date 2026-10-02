"""Owned public-API native graphs; no Torch dependency or implicit fallback.

V1 accepts explicitly built/compiled recipes. Framework recording is a separate
producer, not implemented by pretending a Python call is a device graph.
"""
import ctypes as C
import hashlib
import json
from contextlib import contextmanager
from collections import OrderedDict
from pathlib import Path

ABI = 1
INPUT, OUTPUT, TEMP, EXTERNAL_READ, EXTERNAL_RW = range(1, 6)
ALL_REDUCE, ALL_GATHER, REDUCE_SCATTER = range(1, 4)
F32, BF16, I32, U8 = range(1, 5)


class GraphError(RuntimeError):
    def __init__(self, status, message):
        self.status = status
        super().__init__(f"native graph status {status}: {message}")


class _Config(C.Structure):
    _fields_ = [('size', C.c_uint32), ('version', C.c_uint32),
                ('device', C.c_uint32), ('flights', C.c_uint32),
                ('artifact_directory', C.c_char_p)]


class _Binding(C.Structure):
    _fields_ = [('size', C.c_uint32), ('version', C.c_uint32),
                ('tensor', C.c_char_p), ('buffer', C.c_char_p), ('offset', C.c_uint64)]


class _Input(C.Structure):
    _fields_ = [('size', C.c_uint32), ('version', C.c_uint32),
                ('name', C.c_char_p), ('data', C.c_void_p), ('bytes', C.c_uint64)]


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


class Graph:
    def __init__(self, library, directory, *, device, flights=4):
        self.library = Path(library).resolve()
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        self.api = C.CDLL(str(self.library))
        p, s, u, i = C.c_void_p, C.c_char_p, C.c_uint64, C.c_int
        signatures = {
            'create': [C.POINTER(_Config), C.POINTER(p)],
            'buffer': [p, s, u, u, i], 'output': [p, s, u, u], 'recipe': [p, s, p],
            'recipe_load': [p, s, s], 'launch': [p, s, C.POINTER(_Binding), C.c_uint32],
            'comm': [p, s, p], 'comm_on_device': [p, s, p, C.c_uint32],
            'collect': [p, i, s, s, u, s, u, u, i],
            'copy': [p, s, u, s, u, u], 'pool_requirements': [p, C.POINTER(u)],
            'bind_pool': [p, u, u], 'instantiate': [p],
            'replay': [p, C.POINTER(_Input), C.c_uint32, C.POINTER(u)],
            'query': [p, u], 'wait': [p, u], 'read': [p, u, s, p, u],
            'release': [p, u], 'check_external': [p, s, u, u],
            'address': [p, s, C.POINTER(u)], 'report': [p, s], 'destroy': [p],
        }
        for name, args in signatures.items():
            f = getattr(self.api, 'gkg_' + name)
            f.argtypes, f.restype = args, i
        self.api.gkg_error.argtypes = [p]
        self.api.gkg_error.restype = s
        self.handle = p()
        config = _Config(C.sizeof(_Config), ABI, device, flights, str(self.directory).encode())
        self._check(self.api.gkg_create(C.byref(config), C.byref(self.handle)))
        self.device, self.flights = device, flights
        self.buffers, self.recipes, self.commands, self.owners = {}, {}, [], {}
        self.bucket, self.ready = None, False

    def _check(self, status):
        if status:
            message = self.api.gkg_error(self.handle).decode(errors='replace')
            raise GraphError(status, message)

    def buffer(self, name, *, bytes, role, capture_address=0, owner=None):
        if role in (EXTERNAL_READ, EXTERNAL_RW) and owner is None:
            raise ValueError('external buffer requires a retained owner')
        self._check(self.api.gkg_buffer(self.handle, name.encode(), capture_address, bytes, role))
        self.buffers[name] = dict(bytes=bytes, role=role, capture_address=capture_address)
        if owner is not None:
            self.owners[name] = owner
        return self

    def static_input(self, name, *, bytes, capture_address=0):
        return self.buffer(name, bytes=bytes, role=INPUT, capture_address=capture_address)

    def static_output(self, name, *, bytes, capture_address=0):
        if capture_address:
            self._check(self.api.gkg_output(self.handle, name.encode(), capture_address, bytes))
            self.buffers[name] = dict(bytes=bytes, role=OUTPUT, capture_address=capture_address)
            return self
        return self.buffer(name, bytes=bytes, role=OUTPUT, capture_address=capture_address)

    @contextmanager
    def capture(self, stream):
        """Record synchronous host submissions on one public Synapse stream.

        Requires libgkg_capture.so preloaded at process startup. This producer
        has not qualified Lazy framework worker draining; it must not be used
        around a Python forward that only queues work for a later host thread.
        Input uploads, device synchronization and readback belong outside.
        """
        api = C.CDLL(None)
        try:
            begin, end = api.gkg_capture_begin, api.gkg_capture_end
        except AttributeError as e:
            raise RuntimeError('start the process with LD_PRELOAD=libgkg_capture.so') from e
        begin.argtypes, begin.restype = [C.c_void_p, C.c_void_p], C.c_int
        end.argtypes, end.restype = [C.c_void_p, C.c_char_p], C.c_int
        self._check(begin(self.handle, stream))
        try:
            yield self
        except BaseException:
            end(self.handle, str(self.directory / 'capture.json').encode())
            self.close()
            raise
        self._check(end(self.handle, str(self.directory / 'capture.json').encode()))

    def recipe(self, name, handle):
        self._check(self.api.gkg_recipe(self.handle, name.encode(), handle))
        self.recipes[name] = self.directory / f'recipe-{len(self.recipes)}.bin'
        return self

    def recipe_load(self, name, file):
        file = Path(file).resolve()
        self._check(self.api.gkg_recipe_load(self.handle, name.encode(), str(file).encode()))
        self.recipes[name] = file
        return self

    def launch(self, recipe, bindings):
        """bindings: {compiled tensor name: (declared base buffer, byte offset)}."""
        b = (_Binding * len(bindings))(*[
            _Binding(C.sizeof(_Binding), ABI, t.encode(), n.encode(), offset)
            for t, (n, offset) in bindings.items()])
        self._check(self.api.gkg_launch(self.handle, recipe.encode(), b, len(b)))
        self.commands.append(dict(kind='launch', recipe=recipe, bindings=bindings))
        return self

    def comm(self, name, handle, *, owner):
        self._check(self.api.gkg_comm_on_device(self.handle, name.encode(), handle, self.device))
        self.owners['comm:' + name] = owner
        return self

    def collect(self, kind, comm, src, dst, *, count, dtype, src_offset=0, dst_offset=0):
        self._check(self.api.gkg_collect(self.handle, kind, comm.encode(), src.encode(), src_offset,
                                       dst.encode(), dst_offset, count, dtype))
        self.commands.append(dict(kind='collect', collective=kind, comm=comm, src=src, dst=dst,
                                  count=count, dtype=dtype, src_offset=src_offset, dst_offset=dst_offset))
        return self

    def copy(self, src, dst, *, bytes, src_offset=0, dst_offset=0):
        self._check(self.api.gkg_copy(self.handle, src.encode(), src_offset, dst.encode(), dst_offset, bytes))
        self.commands.append(dict(kind='copy', src=src, dst=dst, bytes=bytes,
                                  src_offset=src_offset, dst_offset=dst_offset))
        return self

    def pool_requirements(self):
        value = C.c_uint64()
        self._check(self.api.gkg_pool_requirements(self.handle, C.byref(value)))
        return value.value

    def bind_pool(self, address, bytes, *, owner):
        if owner is None:
            raise ValueError('dedicated arena requires a retained owner')
        self._check(self.api.gkg_bind_pool(self.handle, address, bytes))
        self.owners['pool:dedicated'] = owner
        return self

    def instantiate(self, *, bucket=None, pool_allocator=None):
        """Optional allocator returns (address, capacity_bytes, retained_owner).

        The producer allocates a fresh, dedicated arena; captured intermediates
        must never be used as its backing storage. V1 remains Torch-free.
        """
        if pool_allocator is not None:
            address, bytes, owner = pool_allocator(self.pool_requirements())
            self.bind_pool(address, bytes, owner=owner)
        self._check(self.api.gkg_instantiate(self.handle))
        self.bucket, self.ready = bucket, True
        return self

    def replay(self, inputs, *, async_=True):
        storage = []
        for name, data in inputs.items():
            raw = memoryview(data)
            if not raw.c_contiguous:
                raise ValueError('static input must be contiguous')
            storage.append((name, C.create_string_buffer(raw.tobytes())))
        args = (_Input * len(storage))(*[
            _Input(C.sizeof(_Input), ABI, name.encode(), C.cast(buf, C.c_void_p), len(buf) - 1)
            for name, buf in storage])
        ticket = C.c_uint64()
        self._check(self.api.gkg_replay(self.handle, args, len(args), C.byref(ticket)))
        flight = Flight(self, ticket.value)
        if not async_:
            flight.wait()
        return flight

    def check_external(self, name, address, bytes):
        self._check(self.api.gkg_check_external(self.handle, name.encode(), address, bytes))

    def address(self, name):
        value = C.c_uint64()
        self._check(self.api.gkg_address(self.handle, name.encode(), C.byref(value)))
        return value.value

    def report(self, file=None):
        path = Path(file) if file is not None else self.directory / 'report.json'
        self._check(self.api.gkg_report(self.handle, str(path).encode()))
        return json.loads(path.read_text())

    def fingerprint(self):
        libraries = [self.library, Path('/usr/lib/habanalabs/libSynapse.so'),
                     Path('/usr/lib/habanalabs/libhcl.so'), Path('/usr/lib/habanalabs/libscal.so')]
        return dict(abi=ABI, libraries={p.name: _hash(p) for p in libraries},
                    driver=self.report()['driver'])

    def save(self, directory):
        """Recipes + declarative manifest. External pointers are diagnostics only."""
        import shutil
        target = Path(directory).resolve()
        target.mkdir(parents=True, exist_ok=False)
        recipes = {}
        report = self.report()
        for index, record in enumerate(report['recipes']):
            name, file = record['name'], record['file']
            dest = target / f'recipe-{index}.bin'
            shutil.copyfile(file, dest)
            recipes[name] = dict(file=dest.name, sha256=_hash(dest))
        buffers = {b['name']: {k: b[k] for k in ('bytes', 'role', 'capture_address')}
                   for b in report['buffers']}
        manifest = dict(version=ABI, fingerprint=self.fingerprint(), bucket=self.bucket,
                        flights=self.flights, buffers=buffers, recipes=recipes,
                        commands=report['plan_commands'], communicators=report['communicators'])
        (target / 'plan.json').write_text(json.dumps(manifest, indent=2) + '\n')

    @classmethod
    def load(cls, library, plan, directory, *, device, externals=None, comms=None, pool_allocator=None):
        plan = Path(plan).resolve()
        manifest = json.loads((plan / 'plan.json').read_text())
        if manifest['version'] != ABI:
            raise ValueError('plan ABI mismatch')
        graph = cls(library, directory, device=device, flights=manifest['flights'])
        try:
            if graph.fingerprint() != manifest['fingerprint']:
                raise ValueError('SDK/driver/runtime fingerprint mismatch')
            for name, b in manifest['buffers'].items():
                if b['role'] in (EXTERNAL_READ, EXTERNAL_RW):
                    # Never trust an address saved in another process.
                    address, bytes, owner = (externals or {})[name]
                    if bytes != b['bytes']:
                        raise ValueError('external size mismatch')
                else:
                    address, bytes, owner = 0, b['bytes'], None
                graph.buffer(name, bytes=bytes, role=b['role'], capture_address=address, owner=owner)
            for name, r in manifest['recipes'].items():
                path = (plan / r['file']).resolve()
                if path.parent != plan or _hash(path) != r['sha256']:
                    raise ValueError('recipe path/hash mismatch')
                graph.recipe_load(name, path)
            required = {c['comm'] for c in manifest['commands'] if c['kind'] == 'collect'}
            for name in required:
                handle, owner = (comms or {})[name]
                graph.comm(name, handle, owner=owner)
            if graph.report()['communicators'] != manifest['communicators']:
                raise ValueError('communicator rank/size mismatch')
            for command in manifest['commands']:
                c = dict(command)
                kind = c.pop('kind')
                if kind == 'launch':
                    graph.launch(**c)
                elif kind == 'copy':
                    graph.copy(**c)
                elif kind == 'collect':
                    graph.collect(c.pop('collective'), **c)
                else:
                    raise ValueError('unregistered plan command')
            return graph.instantiate(bucket=manifest['bucket'], pool_allocator=pool_allocator)
        except BaseException:
            graph.close()
            raise

    def close(self):
        if self.handle:
            status = self.api.gkg_destroy(self.handle)
            if status == 8:
                self._check(status)  # BUSY preserves the live handle and owners.
            self.handle = C.c_void_p()
            self.owners.clear()
            if status:
                # destroy releases the handle on non-BUSY errors. Do not query
                # its error string after free.
                raise GraphError(status, 'native destruction failed after releasing the graph')


class Flight:
    def __init__(self, graph, ticket):
        self.graph, self.ticket = graph, ticket

    def query(self):
        status = self.graph.api.gkg_query(self.graph.handle, self.ticket)
        if status == 8:
            return False
        self.graph._check(status)
        return True

    def wait(self):
        self.graph._check(self.graph.api.gkg_wait(self.graph.handle, self.ticket))
        return self

    def result(self, name):
        size = self.graph.buffers[name]['bytes']
        data = C.create_string_buffer(size)
        self.graph._check(self.graph.api.gkg_read(self.graph.handle, self.ticket, name.encode(), data, size))
        return data.raw

    def release(self):
        self.graph._check(self.graph.api.gkg_release(self.graph.handle, self.ticket))


class Cache:
    """Bucket cache. Eviction never destroys a graph with outstanding work."""
    def __init__(self, max_graphs=8):
        if max_graphs < 1:
            raise ValueError('max_graphs must be positive')
        self.max_graphs, self.graphs = max_graphs, OrderedDict()

    def get(self, key, builder):
        if key in self.graphs:
            self.graphs.move_to_end(key)
            return self.graphs[key]
        if len(self.graphs) >= self.max_graphs:
            old = next(iter(self.graphs))
            self.graphs[old].close()  # BUSY leaves the old graph and cache intact.
            del self.graphs[old]
        graph = builder()
        if not graph.ready:
            graph.close()
            raise ValueError('cache builder must return an instantiated graph')
        self.graphs[key] = graph
        return graph
