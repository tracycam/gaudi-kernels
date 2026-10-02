"""Opt-in isolated communicator experiment; does not edit deployed plugin."""
from gaudi_kernels.engine.context import context as execution_context
import importlib.abc
import importlib.machinery
import os
import sys
if execution_context().startup.inventory:
    from gaudi_kernels.engine.inventory import start
    start(__import__('pathlib').Path(__file__).resolve().parents[1])
MODE = 'skip-moe-output-view'
if MODE in ('baseline', 'no-ar-mark', 'cache-tensors', 'skip-moe-output-view'):

    class Finder(importlib.abc.MetaPathFinder):

        def find_spec(self, fullname, path, target=None):
            targets = {'vllm.v1.executor.multiproc_executor', 'vllm_gaudi.distributed.device_communicators.hpu_communicator', 'vllm_gaudi.v1.worker.hpu_worker', 'vllm_gaudi.v1.worker.hpu_model_runner', 'vllm.model_executor.model_loader.default_loader', 'vllm_gaudi.extension.ops', 'vllm.model_executor.layers.fused_moe.router.gate_linear'}
            if execution_context().selection.engine.runtime.runner == 'native':
                targets -= {'vllm_gaudi.v1.worker.hpu_worker', 'vllm_gaudi.v1.worker.hpu_model_runner'}
            if MODE == 'skip-moe-output-view':
                targets.add('vllm_gaudi.ops.hpu_mxfp4')
            if execution_context().has('block_fp8'):
                targets.add('vllm_gaudi.ops.hpu_fp8')
            if execution_context().has('norm'):
                targets.add('vllm_gaudi.ops.hpu_layernorm')
            if execution_context().has('swa'):
                targets.add('vllm_gaudi.attention.backends.hpu_attn')
            if fullname not in targets:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
            if spec is None:
                return None
            original = spec.loader

            class Loader(importlib.abc.Loader):

                def create_module(self, module_spec):
                    return original.create_module(module_spec)

                def exec_module(self, module):
                    if fullname == 'vllm_gaudi.ops.hpu_mxfp4' and MODE == 'skip-moe-output-view':
                        source = original.get_source(fullname)
                        needle = 'return output.view('
                        assert source.count(needle) == 2, 'Frozen MXFP4 return sites changed'
                        source = source.replace(needle, 'return _comm_output_view(output, ')
                        route_needle = 'topk_weights = topk_weights.to(x.dtype)'
                        assert source.count(route_needle) == 2, 'Frozen route cast sites changed'
                        source = source.replace(route_needle, 'topk_weights = topk_weights.to(_precision_route_dtype(x))')
                        from gaudi_kernels.serving.executor.precision_runtime import route_dtype
                        module.__dict__['_precision_route_dtype'] = route_dtype

                        def output_view(output, *shape):
                            return output if tuple(output.shape) == tuple(shape) else output.view(*shape)
                        module.__dict__['_comm_output_view'] = output_view
                        exec(compile(source, spec.origin, 'exec'), module.__dict__)
                        print(f'COMM_EXP_REWRITE pid={os.getpid()} module={fullname} return_sites=2', flush=True)
                    else:
                        original.exec_module(module)
                    if fullname == 'vllm.v1.executor.multiproc_executor':
                        from gaudi_kernels.serving.host_placement import install_worker_spawn
                        install_worker_spawn(module)
                    if fullname == 'vllm_gaudi.ops.hpu_fp8' and execution_context().has('block_fp8'):
                        if execution_context().has('reduce_isa'):
                            from gaudi_kernels.serving.executor.qkv_neumaier_isa_runtime import prepare
                            prepare()
                        from gaudi_kernels.production_integration import install_block_fp8
                        install_block_fp8(module, enabled=True, scope='qkv', policy='bf16_fp32', extension_path=execution_context().path('block_fp8', 'torch'), keep_cpu_oracle=execution_context().startup.keep_cpu_oracle)
                    if fullname == 'vllm_gaudi.ops.hpu_layernorm' and execution_context().has('norm'):
                        import torch
                        torch.ops.load_library(execution_context().path('norm', 'torch'))
                        from gaudi_kernels.vllm_norm import install_vllm_norm
                        install_vllm_norm('vendor')
                        if execution_context().has('norm_grid24'):
                            from gaudi_kernels.serving.executor.norm_grid24_runtime import prepare as prepare_grid24
                            prepare_grid24()
                    if fullname == 'vllm_gaudi.attention.backends.hpu_attn' and execution_context().has('swa'):
                        import torch
                        if execution_context().has('qkv_post'):
                            torch.ops.load_library(execution_context().path('qkv_post', 'torch'))
                        torch.ops.load_library(execution_context().path('swa', 'torch'))
                        if execution_context().has('swa_av'):
                            from gaudi_kernels.serving.executor.swa_av_runtime import prepare as prepare_swa_av
                            prepare_swa_av()
                        if execution_context().has('swa_batch'):
                            from gaudi_kernels.serving.executor.swa_batch_runtime import prepare as prepare_swa_batch
                            prepare_swa_batch()
                        from gaudi_kernels.vllm_swa_install import install_vllm_swa
                        install_vllm_swa('vendor')
                    if fullname.endswith(('extension.ops', 'router.gate_linear', 'hpu_communicator')):
                        from gaudi_kernels.serving.executor.precision_runtime import install as install_precision
                        install_precision(module)
                    if fullname.endswith('model_loader.default_loader') and int(str(execution_context().startup.layers)) < 70:
                        import re
                        limit = int(str(execution_context().startup.layers))
                        original_weights = module.DefaultModelLoader.get_all_weights

                        def filtered_weights(self, *args, **kwargs):
                            for (name, tensor) in original_weights(self, *args, **kwargs):
                                match = re.search('(?:^|\\.)layers\\.(\\d+)\\.', name)
                                if match is None or int(match[1]) < limit:
                                    yield (name, tensor)
                        module.DefaultModelLoader.get_all_weights = filtered_weights
                    if fullname.endswith('hpu_mxfp4') and '1' == '1':
                        from gaudi_kernels.serving.executor.native_model import install
                        install(module)
                    if fullname.endswith('hpu_model_runner') and '1' == '1' and ('0' == '0'):
                        original_graphs = module.HPUModelRunner._use_graphs

                        def native_graphs(self, attn_metadata, batch_size):
                            if attn_metadata is not None and attn_metadata.is_prompt:
                                return False
                            return original_graphs(self, attn_metadata, batch_size)
                        module.HPUModelRunner._use_graphs = native_graphs
                    if fullname.endswith('hpu_model_runner') and (MODE == 'cache-tensors' or ('1' == '1' and '1' == '1')):

                        from gaudi_kernels.serving.model_adapter import adapter_class
                        adapter_type = adapter_class(module.HpuModelAdapter) if execution_context().has('qkv_post') else module.HpuModelAdapter

                        def maybe_wrap(*args, **kwargs):
                            adapter = adapter_type(*args, **kwargs)
                            return module.htorch.hpu.wrap_in_hpu_graph(adapter, disable_tensor_cache=False, asynchronous=None == '1') if module.htorch.utils.internal.is_lazy() else adapter
                        module._maybe_wrap_in_hpu_graph = maybe_wrap
                    if fullname.endswith('hpu_communicator') and MODE == 'no-ar-mark':
                        import torch.distributed as dist

                        def all_reduce(self, input_):
                            dist.all_reduce(input_, group=self.device_group)
                            return input_
                        module.HpuCommunicator.all_reduce = all_reduce
                    if fullname.endswith('hpu_model_runner'):
                        original_prepare = module.HPUModelRunner._prepare_dummy_scenario
                        original_add = module.HPUModelRunner._add_dummy_request

                        def offset_add(self, requests, scheduled, num_computed_tokens, total_tokens, scheduled_tokens, is_prompt, block_id=0):
                            if getattr(self, '_comm_offset_warmup', False) and (not is_prompt):
                                assert num_computed_tokens == total_tokens and scheduled_tokens == 1
                                assert num_computed_tokens > 1
                                num_computed_tokens -= 1
                                total_tokens -= 1
                            return original_add(self, requests, scheduled, num_computed_tokens, total_tokens, scheduled_tokens, is_prompt, block_id)

                        def prepare_with_offset(self, prompt_cfg, decode_cfg):
                            result = original_prepare(self, prompt_cfg, decode_cfg)
                            if decode_cfg and (not prompt_cfg) and self.interleaved_sliding_window and (self.speculative_config is None):
                                self._comm_offset_warmup = True
                                try:
                                    original_prepare(self, prompt_cfg, decode_cfg)
                                finally:
                                    self._comm_offset_warmup = False
                                print(f'COMM_SWA_OFFSET_WARMUP pid={os.getpid()} cfg={decode_cfg}', flush=True)
                            return result
                        module.HPUModelRunner._add_dummy_request = offset_add
                        module.HPUModelRunner._prepare_dummy_scenario = prepare_with_offset
                    if fullname.endswith('hpu_model_runner'):
                        from gaudi_kernels.serving.executor.native_backend import install as install_native_backend
                        install_native_backend(module)
                        from gaudi_kernels.serving.executor.production_quality import install_runner as install_quality
                        install_quality(module)
                    if fullname.endswith('hpu_worker'):
                        from gaudi_kernels.serving.executor.production_quality import configure_capture, configure_policy, snapshot
                        module.HPUWorker.production_quality_capture = configure_capture
                        module.HPUWorker.production_policy = configure_policy
                        module.HPUWorker.production_snapshot = snapshot
                        from gaudi_kernels.serving.executor.native_backend import configure as native_configure, summary as native_summary, configure_profile
                        module.HPUWorker.native_configure = native_configure
                        module.HPUWorker.native_summary = native_summary
                        module.HPUWorker.native_profile = configure_profile
                        from gaudi_kernels.serving.host_placement import snapshot as host_snapshot
                        module.HPUWorker.host_placement_snapshot = host_snapshot
                    print(f'COMM_EXP_INSTALLED mode={MODE} pid={os.getpid()} module={fullname}', flush=True)
            spec.loader = Loader()
            return spec
    sys.meta_path.insert(0, Finder())
