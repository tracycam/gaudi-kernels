"""vLLM worker extension: explicit runner factory and public utility methods."""
from vllm_gaudi.v1.worker.hpu_worker import HPUWorker



class FrameworkHPUWorker(HPUWorker):
    def init_device(self):
        result = super().init_device()
        from gaudi_kernels.engine.context import context
        ctx = context()
        minimum = ctx.selection.engine.runtime.host.worker_util_min
        if minimum:
            from gaudi_kernels.serving.host_qos import WorkerQos, primary_physical_cpus
            cpus = None
            placement_file = getattr(getattr(ctx, 'startup', None), 'run_dir', None)
            if placement_file is not None and (placement_file / 'host-placement.json').exists():
                import json
                plan = json.loads((placement_file / 'host-placement.json').read_text())
                if 0 <= getattr(self, 'local_rank', -1) < len(plan.get('workers', ())):
                    allowed = plan['workers'][self.local_rank].get('allowed_local_cpus')
                    if allowed:
                        cpus = primary_physical_cpus(allowed)
            self._host_qos = WorkerQos(minimum, cpus=cpus)
        communication = context().selection.engine.runtime.communication
        if communication.gather == 'two_stage':
            from vllm.distributed.parallel_state import get_tp_group
            from gaudi_kernels.serving.distributed.rank_gather import (
                GatherLayout, RankGatherTransport, GatherTransportCommunicator)
            tp = get_tp_group()
            # Configuration uses TP-local rank indices; hardware modules are
            # mapped independently by host placement. Never conflate the two.
            ranks = tuple(tp.ranks)
            layout = GatherLayout(
                tuple(tuple(ranks[i] for i in group) for group in communication.local_groups),
                tuple(tuple(ranks[i] for i in group) for group in communication.exchange_groups))
            self.rank_gather_transport = RankGatherTransport(tp.device_group, layout)
            original = tp.device_communicator
            original.rank_gather_transport = self.rank_gather_transport
            tp.device_communicator = GatherTransportCommunicator(original, self.rank_gather_transport)
        return result

    def shutdown(self):
        try:
            return super().shutdown()
        finally:
            qos = getattr(self, '_host_qos', None)
            if qos is not None:
                qos.restore()

    def get_model_runner_cls(self):
        from gaudi_kernels.serving.runner import FrameworkHPUModelRunner
        return FrameworkHPUModelRunner

    def load_model(self, *args, **kwargs):
        result = super().load_model(*args, **kwargs)
        if type(self) is FrameworkHPUWorker:
            from gaudi_kernels.serving.ops_registry import prepare_model_operators
            model = getattr(getattr(self, 'model_runner', None), 'model', None)
            if model is not None:
                prepare_model_operators(model)
        qos = getattr(self, '_host_qos', None)
        if qos is not None:
            qos.refresh()
        return result


class MimoHPUWorker(FrameworkHPUWorker):
    """Reference model operator installation; no alternative execution core."""
    def load_model(self, *args, **kwargs):
        from gaudi_kernels.serving.packed_backend import register
        from gaudi_kernels.serving.policy import configure_policy
        from gaudi_kernels.engine.context import context
        register()
        result = super().load_model(*args, **kwargs)
        configure_policy(self, context().selection.to_dict())
        qos = getattr(self, '_host_qos', None)
        if qos is not None:
            qos.refresh()
        return result

    def get_model_runner_cls(self):
        from gaudi_kernels.serving.models.mimo_runner import MimoHPUModelRunner
        return MimoHPUModelRunner
