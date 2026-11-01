"""Benchmark-only launcher; historical control RPCs never enter serving boot."""
import argparse
from pathlib import Path
import sys
import os
import json

from gaudi_kernels.serving.launch import prepare_launch


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--plugin-source', type=Path, required=True)
    parser.add_argument('--vllm-source', type=Path, required=True)
    parser.add_argument('--benchmark-module', required=True)
    parser.add_argument('--python', default=sys.executable)
    args, remainder = parser.parse_known_args()
    if remainder and remainder[0] == '--':
        remainder = remainder[1:]
    python, command, environment = prepare_launch(manifest=args.manifest, plugin_source=args.plugin_source, vllm_source=args.vllm_source,
           module=args.benchmark_module, python=args.python, arguments=remainder,
           bootstrap=Path(__file__).resolve().parent / 'startup', output_argument=True)
    from gaudi_kernels.engine.context import context
    runtime = context()
    if runtime.has('prepared_fault.host'):
        if args.benchmark_module != 'tools.validation.executor.bridge_collective_test' or '--fault-prefix' not in remainder:
            raise ValueError('Launch failure injection requires the explicit two-rank prefix test')
        environment['LD_PRELOAD'] = runtime.path('prepared_fault', 'host')
    if runtime.has('submission_timing.host'):
        if args.benchmark_module != 'tools.validation.executor.bridge_model_test' or '--submission-timing' not in remainder:
            raise ValueError('API observer requires an explicit diagnostic run')
        environment['LD_PRELOAD'] = runtime.path('submission_timing', 'host')
    if runtime.has('prequeue_gate.host'):
        if args.benchmark_module != 'tools.validation.executor.bridge_model_test' or '--prequeue-library' not in remainder:
            raise ValueError('The gate provider is admitted only by the explicit prequeue diagnostic')
        original = runtime.artifact('down', 'tpc')
        wrapper = runtime.artifact('prequeue_gate', 'host')
        paths = environment['GC_KERNEL_PATH'].split(':')
        if paths.count(str(original.path)) != 1:
            raise ValueError('Gate provider requires exactly one original down database')
        environment['GC_KERNEL_PATH'] = ':'.join(str(wrapper.path) if p == str(original.path) else p for p in paths)
        if runtime.has('prequeue_events.host'):
            environment['LD_PRELOAD'] = runtime.path('prequeue_events', 'host')
        (runtime.startup.run_dir/'diagnostic-tpc-provider.json').write_text(json.dumps(dict(
            original=str(original.path), original_sha256=original.sha256,
            wrapper=str(wrapper.path), wrapper_sha256=wrapper.sha256,
            scope='Diagnostic registration wrapper; original model ELFs delegated unchanged'),indent=2)+'\n')
    os.execve(python, command, environment)


if __name__ == '__main__':
    main()
