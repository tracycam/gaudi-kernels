"""Explicit process-local execution settings and SHA-pinned artifact ownership.

Only bootstrap reads a manifest location from the environment. Capture and
replay use these typed objects; changing arithmetic requires a drained runner.
This module imports neither Torch nor vLLM.
"""
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

from .config import ConfigError
from .selection import PolicySelection
from .artifact_pins import PINS


@dataclass(frozen=True)
class Artifact:
    path: Path
    sha256: str

    def verify(self):
        digest = hashlib.sha256()
        with self.path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b''):
                digest.update(chunk)
        if digest.hexdigest() != self.sha256:
            raise ConfigError('Artifact bytes changed: ' + str(self.path))
        return self


@dataclass(frozen=True)
class Startup:
    run_dir: Path
    layers: int = 70
    keep_cpu_oracle: bool = False
    layer_diagnostics: bool = False
    same_input_audit: bool = False
    profile_config: Path | None = None
    inventory: bool = False


@dataclass
class NativeState:
    active: bool = False
    single_rpc: bool = False
    reuse_pages: bool = False
    pad_window: bool = True
    boundary_positions: tuple[int, ...] = ()
    replay_mode: str = 'compact'


@dataclass
class ExecutionContext:
    startup: Startup
    artifacts: object
    selection: PolicySelection
    native: NativeState = field(default_factory=NativeState)
    # This is the qualified 70-g collective shape policy, not a universal AG.
    collective_mode: str = 'auto'

    def has(self, name):
        return name in self.artifacts or any(name+'.'+kind in self.artifacts for kind in ('torch','tpc','host'))

    def artifact(self, name, kind='torch'):
        key = name + '.' + kind
        if key not in self.artifacts:
            raise ConfigError('Required artifact absent: ' + key)
        return self.artifacts[key]

    def path(self, name, kind='torch'):
        return str(self.artifact(name, kind).path)

    def sha(self, name, kind='torch'):
        return self.artifact(name, kind).sha256

    def libraries(self, name, qualified=None):
        result = {}
        for kind in ('tpc', 'torch'):
            artifact = self.artifact(name, kind).verify()
            if qualified is not None and artifact.sha256 != qualified[kind]:
                raise ConfigError('Unqualified artifact: ' + name + '.' + kind)
            result[kind] = {'path': str(artifact.path), 'sha256': artifact.sha256}
        # GC_KERNEL_PATH is an SDK input; actual registration is checked at
        # prepare time by the caller, not inferred from a manifest claim.
        return result


_CONTEXT = None


def context():
    if _CONTEXT is None:
        raise ConfigError('Initialize the explicit serving manifest before importing device bindings')
    return _CONTEXT


def initialize(document, *, directory):
    """Validate all source-independent bindings before loading any extension."""
    global _CONTEXT
    if _CONTEXT is not None:
        raise ConfigError('Execution context already initialized')
    if type(document) is not dict or set(document) != {'schema_version', 'startup', 'artifacts', 'selection'}:
        raise ConfigError('Serving manifest requires schema_version/startup/artifacts/selection')
    if type(document['schema_version']) is not int or document['schema_version'] != 1:
        raise ConfigError('Unsupported serving manifest schema')
    startup = document['startup']
    allowed = {'run_dir', 'layers', 'keep_cpu_oracle', 'layer_diagnostics', 'same_input_audit', 'profile_config', 'inventory'}
    if type(startup) is not dict or set(startup) - allowed or 'run_dir' not in startup:
        raise ConfigError('Unknown/missing startup setting')
    if type(startup['run_dir']) is not str:
        raise ConfigError('startup.run_dir must be a path string')
    if type(startup.get('layers', 70)) is not int or not 1 <= startup.get('layers', 70) <= 70:
        raise ConfigError('startup.layers must be 1..70; partial layers are diagnostic only')
    for name in ('keep_cpu_oracle', 'layer_diagnostics', 'same_input_audit', 'inventory'):
        if type(startup.get(name, False)) is not bool:
            raise ConfigError('startup.' + name + ' must be bool')
    base = Path(directory).resolve(strict=True)
    options = dict(startup)
    options['run_dir'] = Path(options['run_dir']).resolve()
    if options.get('profile_config') is not None:
        if type(options['profile_config']) is not str:
            raise ConfigError('profile_config must be path/null')
        options['profile_config'] = Path(options['profile_config']).resolve(strict=True)
    artifacts = {}
    if type(document['artifacts']) is not dict or not document['artifacts']:
        raise ConfigError('No explicit artifact bindings')
    for name, binding in document['artifacts'].items():
        if type(name) is not str or name.count('.') != 1 or name.rsplit('.', 1)[1] not in ('torch', 'tpc', 'host'):
            raise ConfigError('Artifact name must be group.torch/tpc/host')
        if type(binding) is not dict or set(binding) != {'path', 'sha256'}:
            raise ConfigError('Artifact requires exactly path/sha256')
        if type(binding['path']) is not str or type(binding['sha256']) is not str:
            raise ConfigError('Artifact path/hash must be strings')
        if len(binding['sha256']) != 64 or any(c not in '0123456789abcdef' for c in binding['sha256']):
            raise ConfigError('Artifact SHA256 must be lowercase hexadecimal')
        if name.endswith(('.torch', '.tpc')) or name=='tensor_hold.host':
            if name not in PINS or PINS[name]!=binding['sha256']:
                raise ConfigError('Unqualified executable artifact: '+name)
        path = Path(binding['path'])
        if not path.is_absolute():
            path = base / path
        artifact = Artifact(path.resolve(strict=True), binding['sha256']).verify()
        artifacts[name] = artifact
    candidate = ExecutionContext(Startup(**options), MappingProxyType(artifacts),
                                 PolicySelection.from_dict(document['selection']))
    _CONTEXT = candidate
    return candidate


def load_manifest(path):
    from .config import _unique_object
    path = Path(path).resolve(strict=True)
    document = json.loads(path.read_text(), object_pairs_hook=_unique_object)
    return initialize(document, directory=path.parent)
