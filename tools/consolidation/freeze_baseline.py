#!/usr/bin/env python3
"""Reconstitute the SHA-pinned 70-g run without editing any existing runtime.

The seed supplies files not captured by the historical freezer. Its Python
sources are inventoried separately; they are not claimed to be frozen 70-g
sources. This is deliberately a migration tool, not a new model executor.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(path.read_text())


def safe_relative(name):
    p = Path(name)
    if p.is_absolute() or '..' in p.parts:
        raise ValueError(f'Unsafe manifest path: {name}')
    return p


def destination(name, runtime):
    p = safe_relative(name)
    if p.parts[:2] == ('production-runtime', 'kernels'):
        return runtime / 'kernels/python/gaudi_kernels' / Path(*p.parts[2:])
    if p.parts[:2] == ('production-runtime', 'libraries'):
        return runtime / 'libraries' / Path(*p.parts[2:])
    if p.parts[0] == 'production-runtime':
        return runtime / Path(*p.parts[1:])
    if p.parts[0] in ('precision-source', 'batch-kernel-source'):
        return runtime / 'executor' / Path(*p.parts[1:])
    if p.parts[0] == 'cpu-fp32-oracle':
        return runtime / p
    return runtime / 'executor/executor' / p


def prepare(snapshot, command_file, environment_file, seed, output):
    manifest = load(snapshot / 'source/sha256.json')
    # Validate every historical byte before creating or launching anything.
    for name, expected in manifest.items():
        source = snapshot / 'source' / safe_relative(name)
        if digest(source) != expected:
            raise ValueError(f'Frozen source SHA mismatch: {name}')
    original_command = load(command_file)
    original_environment = load(environment_file)
    if not isinstance(original_command, list) or not all(isinstance(x, str) for x in original_command):
        raise ValueError('Expected an argv list')
    if not isinstance(original_environment, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in original_environment.items()):
        raise ValueError('Expected string environment entries')
    output.mkdir(parents=True, exist_ok=False)
    runtime = output / 'runtime'
    runtime.mkdir()
    for name in ('plugin', 'vllm', 'kernels'):
        shutil.copytree(seed / name, runtime / name,
                        ignore=shutil.ignore_patterns('.git', '__pycache__', '*.pyc', 'artifacts'))
    # Do not copy 12 GB of old case outputs. Only runtime source/build inputs.
    executor = runtime / 'executor'
    executor.mkdir()
    seed_executor = seed / 'executor'
    for source in seed_executor.iterdir():
        if source.is_file():
            shutil.copy2(source, executor / source.name)
    for name in ('tpc', 'precision-tpc', 'precision-ops-build', 'ops-build',
                 'runtime-sources', 'legacy-native', 'torch-build', 'flow-ops-build'):
        source = seed_executor / name
        if source.is_dir():
            shutil.copytree(source, executor / name,
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (executor / 'executor').mkdir()
    seed_leaf = seed_executor / 'executor'
    for source in seed_leaf.iterdir():
        if source.is_file() and source.suffix in ('.py', '.cpp', '.inc', '.sh', '.so', '.md'):
            shutil.copy2(source, executor / 'executor' / source.name)
    for name in ('tpc', 'torch-build', 'kernels', 'flow-ops-build', 'direct-ops-build'):
        source = seed_leaf / name
        if source.is_dir():
            shutil.copytree(source, executor / 'executor' / name,
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    restored = {}
    for name, expected in manifest.items():
        target = destination(name, runtime)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(snapshot / 'source' / name, target)
        if digest(target) != expected:
            raise ValueError(f'Restored source SHA mismatch: {name}')
        restored[str(target.relative_to(runtime))] = expected
    old_runtime = original_environment['GK_RUNTIME_ROOT']
    env = dict(original_environment)
    special_libraries = {
        'GK_BLOCK_REDUCE_ISA_TPC_LIBRARY': 'production-runtime/qkv-neumaier-isa/tpc.so',
        'GK_BLOCK_REDUCE_ISA_TORCH_LIBRARY': 'production-runtime/qkv-neumaier-isa/torch.so',
        'GK_MXFP4_SCALE_TAIL_TPC_LIBRARY': 'production-runtime/gp-scale-tail/tpc.so',
        'GK_MXFP4_SCALE_TAIL_TORCH_LIBRARY': 'production-runtime/gp-scale-tail/torch.so',
    }
    for key, value in list(env.items()):
        if key in special_libraries:
            env[key] = str(destination(special_libraries[key], runtime))
        elif key.endswith('_LIBRARY'):
            candidate = runtime / 'libraries' / Path(value).name
            oracle = runtime / 'cpu-fp32-oracle' / Path(value).name
            if candidate.is_file():
                env[key] = str(candidate)
            elif oracle.is_file():
                env[key] = str(oracle)
            else:
                env[key] = value.replace(old_runtime, str(runtime), 1)
        elif value.startswith(old_runtime + '/') or value == old_runtime:
            env[key] = str(runtime) + value[len(old_runtime):]
    # The profiler JSON was not part of the historical SHA manifest.
    if env.get('E1_PROFILE_CONFIG'):
        source = Path(original_environment['E1_PROFILE_CONFIG'])
        target = Path(env['E1_PROFILE_CONFIG'])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    env['GK_RUNTIME_ROOT'] = str(runtime)
    environment_path = output / 'environment.env'
    environment_path.write_text(''.join(f'{k}={shlex.quote(v)}\n' for k, v in sorted(env.items())))
    command = [original_command[0], str(executor / 'executor/run_model.py'),
               'baseline-70g', *original_command[3:]]
    seed_sources = {str(p.relative_to(runtime)): digest(p)
                    for p in runtime.rglob('*.py') if p.is_file()}
    metadata = {
        'historical_case': str(snapshot), 'historical_command': original_command,
        'historical_environment_sha256': digest(environment_file),
        'historical_manifest_sha256': digest(snapshot / 'source/sha256.json'),
        'restored_files': restored, 'seed_python_sources': seed_sources,
        'seed_only_sources': sorted(set(seed_sources) - set(restored)),
        'command': command, 'environment': env,
        'scope': 'Frozen recorded sources and binaries restored; uncaptured seed sources explicitly listed',
    }
    (output / 'baseline.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return command, environment_path, env


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--command', type=Path, required=True)
    p.add_argument('--environment', type=Path, required=True)
    p.add_argument('--seed', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--run', action='store_true')
    a = p.parse_args()
    command, environment_path, env = prepare(*(x.resolve() for x in
        (a.snapshot, a.command, a.environment, a.seed, a.output)))
    print(json.dumps({'command': command, 'environment_file': str(environment_path)}, indent=2), flush=True)
    if a.run:
        # The existing supervisor owns locks, idle checks, timeout and cleanup.
        child_env = {**os.environ, **env, 'GK_ENV_FILE': str(environment_path)}
        raise SystemExit(subprocess.call(command, env=child_env))


if __name__ == '__main__':
    main()
