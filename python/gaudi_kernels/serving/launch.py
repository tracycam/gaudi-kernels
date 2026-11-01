"""Launch repository-owned serving code with explicit artifacts and SDK env."""
import argparse
import os
from pathlib import Path
import sys
from gaudi_kernels.engine.context import load_manifest
SDK_ENVIRONMENT = {'PT_HPU_LAZY_MODE': '1', 'PT_HPU_ENABLE_LAZY_COLLECTIVES': 'true', 'PT_HPU_LAZY_COLLECTIVES_HOLD_TENSORS': '1', 'PT_HPU_LAZY_ACC_PAR_MODE': '0', 'PT_HPU_ENABLE_DIV_PRECISE': '1', 'VLLM_HPU_FORCE_CHANNEL_FP8': 'true', 'VLLM_CONTIGUOUS_PA': 'false', 'VLLM_SKIP_WARMUP': '1', 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'}

def prepare_launch(*, manifest, plugin_source, vllm_source, module, arguments=(), python=sys.executable, bootstrap=None,
           output_argument=False):
    manifest = manifest.resolve(strict=True)
    runtime = load_manifest(manifest)
    model = getattr(getattr(getattr(runtime, 'selection', None), 'engine', None), 'model', 'mimo-v2.6-pro')
    if bootstrap is None and model == 'mimo-v2.6-pro' and runtime.startup.layers != 70:
        raise ValueError('Production MiMo launch requires all 70 layers; use validation launcher for reduced models')
    runtime.startup.run_dir.mkdir(parents=True, exist_ok=False)
    from gaudi_kernels.serving.host_placement import prepare
    placement = prepare(runtime)
    package = Path(__file__).resolve().parents[2]
    repository = package.parent
    if bootstrap is None:
        bootstrap = Path(__file__).resolve().parent / 'startup'
    environment = {k: v for (k, v) in os.environ.items() if not k.startswith(('GK_', 'UNIFIED_', 'E1_', 'NATIVE_', 'TOKEN_', 'FLOW_', 'COMM_', 'BUDGET_'))}
    environment.update(SDK_ENVIRONMENT)
    if 'HLS_MODULE_ID' in environment:
        raise RuntimeError('TP launcher must not inherit a single HLS_MODULE_ID')
    environment['HABANA_VISIBLE_MODULES'] = ','.join(str(w['module_id']) for w in placement['workers'])
    if placement['binding'] == 'local':
        environment['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    environment['GK_RUNTIME_MANIFEST'] = str(manifest)
    environment['PYTHONPATH'] = os.pathsep.join((str(p.resolve(strict=True)) for p in (bootstrap, package, repository, plugin_source, vllm_source)))
    # Framework owns submission. Do not inherit historical execution hooks.
    environment.pop('LD_PRELOAD', None)
    environment['GC_KERNEL_PATH'] = ':'.join((str(binding.path) for (name, binding) in runtime.artifacts.items() if name.endswith('.tpc')))
    if len(environment['GC_KERNEL_PATH'].split(':')) > 15:
        raise RuntimeError('Gaudi2 TPC database limit exceeded')
    environment['VLLM_CONFIG_HIDDEN_LAYERS'] = str(runtime.selection.engine.runtime.graph_layer_interval)
    if runtime.startup.profile_config is not None:
        environment['HABANA_PROFILE'] = '1'
        environment['HABANA_PROF_CONFIG'] = str(runtime.startup.profile_config)
    else:
        environment.pop('HABANA_PROFILE', None)
        environment.pop('HABANA_PROF_CONFIG', None)
    environment['HABANA_LOGS'] = str(runtime.startup.run_dir / 'habana-logs')
    command = [python, '-m', module]
    if output_argument:
        command.extend(('--out', str(runtime.startup.run_dir)))
    command.extend(arguments)
    if module == 'vllm.entrypoints.cli.main':
        if any(arg == '--max-num-batched-tokens' or arg.startswith('--max-num-batched-tokens=') or
               arg == '--no-enable-chunked-prefill' for arg in arguments):
            raise ValueError('Set prefill chunk size in the typed runtime configuration')
        command.extend(['--max-num-batched-tokens',
                        str(runtime.selection.engine.runtime.prefill_chunk_tokens), '--enable-chunked-prefill'])
    if runtime.selection.engine.runtime.runner == 'native' and module == 'vllm.entrypoints.cli.main':
        worker_cls = ('gaudi_kernels.serving.worker.MimoHPUWorker'
                      if model == 'mimo-v2.6-pro'
                      else 'gaudi_kernels.serving.worker.FrameworkHPUWorker')
        command.extend(['--worker-cls', worker_cls])
    if placement['binding'] == 'local' and module == 'vllm.entrypoints.cli.main':
        command.extend(['--numa-bind', '--numa-bind-nodes',
                        *[str(w['numa_node']) for w in placement['workers']]])
    return python, command, environment


def launch(**kwargs):
    python, command, environment = prepare_launch(**kwargs)
    os.execve(python, command, environment)

def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--plugin-source', type=Path, required=True)
    parser.add_argument('--vllm-source', type=Path, required=True)
    parser.add_argument('--module', default='vllm.entrypoints.cli.main')
    parser.add_argument('--python', default=sys.executable)
    args, remainder = parser.parse_known_args()
    if remainder and remainder[0] == '--':
        remainder = remainder[1:]
    launch(manifest=args.manifest, plugin_source=args.plugin_source, vllm_source=args.vllm_source,
           module=args.module, python=args.python, arguments=remainder)
if __name__ == '__main__':
    main()
