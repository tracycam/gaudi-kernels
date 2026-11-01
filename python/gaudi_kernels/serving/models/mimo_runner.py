"""MiMo operator binding above the model-independent framework runner."""
import vllm_gaudi.v1.worker.hpu_model_runner as hpu
from dataclasses import replace
from gaudi_kernels.engine.context import context
from gaudi_kernels.serving.graph_options import GraphOptions
from gaudi_kernels.serving.model_adapter import adapter_class
from gaudi_kernels.serving.runner import FrameworkHPUModelRunner


class MimoHPUModelRunner(FrameworkHPUModelRunner):
    def speculative_row_budget(self):
        return context().selection.engine.runtime.speculative_verify_rows

    def _use_graphs(self, attn_metadata, batch_size):
        # The tuned reference prefill operators have not passed graph admission.
        # Do not impose this model/operator limitation on other models.
        if attn_metadata is not None and attn_metadata.is_prompt:
            return False
        return super()._use_graphs(attn_metadata, batch_size)

    def graph_options(self):
        options = getattr(self, '_graph_options', None)
        if options is None:
            options = self._graph_options = replace(GraphOptions.from_vllm(self.vllm_config),
                                                    scope=context().selection.engine.runtime.decode_graph)
        return options

    def _wrap_model_adapter(self, *args, **kwargs):
        cls = adapter_class(hpu.HpuModelAdapter) if context().has('qkv_post') else hpu.HpuModelAdapter
        adapter = cls(*args, **kwargs)
        if hpu.htorch.utils.internal.is_lazy():
            return hpu.htorch.hpu.wrap_in_hpu_graph(adapter, disable_tensor_cache=False, asynchronous=False,
                                                   max_graphs=self.graph_options().max_cached_graphs)
        return adapter
