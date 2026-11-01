"""Two-stage rank-preserving gather over an explicitly supplied topology.

No reductions, dtype changes, host tensor copies, global monkey patches or
port manipulation. Process groups are constructed once, before graph capture.
The caller owns the transport until every graph using its groups is released.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class GatherLayout:
    local_groups: tuple[tuple[int, ...], ...]
    exchange_groups: tuple[tuple[int, ...], ...]

    def validate(self, ranks):
        ranks = tuple(ranks)
        if not ranks or ranks != tuple(sorted(set(ranks))):
            raise ValueError('Parent ranks must be unique and sorted')
        groups = self.local_groups
        if len(groups) < 2 or not groups[0]:
            raise ValueError('At least two nonempty local groups required')
        width = len(groups[0])
        if any(len(g) != width for g in groups) or tuple(r for g in groups for r in g) != ranks:
            raise ValueError('Local groups must partition parent rank order into equal contiguous blocks')
        exchanges = self.exchange_groups
        if len(exchanges) != width or sorted(r for g in exchanges for r in g) != list(ranks):
            raise ValueError('Exchange groups must partition the same parent ranks')
        for g in exchanges:
            if tuple(sorted(g)) != g or len(g) != len(groups) or any(len(set(g) & set(local)) != 1 for local in groups):
                raise ValueError('Each exchange must contain one rank per local group, in parent order')


class RankGatherTransport:
    def __init__(self, parent_group, layout: GatherLayout):
        import torch.distributed as dist
        ranks = tuple(dist.get_process_group_ranks(parent_group))
        layout.validate(ranks)
        rank = dist.get_rank()
        if rank not in ranks:
            raise ValueError('This process is not a member of the parent group')
        self.parent_group = parent_group
        self.layout = layout
        self.ranks = ranks
        self._groups = []
        self.local_group = self.exchange_group = None
        backend = dist.get_backend(parent_group)
        # Every parent rank must enter this constructor in the same order.
        for phase, groups in (('local', layout.local_groups), ('exchange', layout.exchange_groups)):
            for members in groups:
                group = dist.new_group(ranks=list(members), backend=backend,
                                       use_local_synchronization=True)
                self._groups.append(group)
                if rank in members:
                    setattr(self, phase+'_group', group)

    def all_gather_into_tensor(self, output, input):
        import torch
        import torch.distributed as dist
        if input.ndim < 1 or not input.is_contiguous() or not output.is_contiguous():
            raise ValueError('Gather requires non-scalar contiguous tensors')
        shape = (input.shape[0] * len(self.ranks),) + tuple(input.shape[1:])
        if tuple(output.shape) != shape or output.dtype != input.dtype or output.device != input.device:
            raise ValueError('Output must preserve input dtype/device and concatenate all ranks')
        width = len(self.layout.local_groups[0])
        local = torch.empty((input.shape[0]*width,) + tuple(input.shape[1:]),
                            dtype=input.dtype, device=input.device)
        dist.all_gather_into_tensor(local, input, group=self.local_group)
        dist.all_gather_into_tensor(output, local, group=self.exchange_group)

    def all_reduce_float32(self, input):
        """FP32 sum at both stages; associativity differences are explicit."""
        import torch
        import torch.distributed as dist
        if input.dtype != torch.float32:
            raise ValueError('Two-stage sum requires FP32 intermediate accumulation')
        dist.all_reduce(input, group=self.local_group)
        dist.all_reduce(input, group=self.exchange_group)
        return input


class GatherTransportCommunicator:
    """Instance-owned vLLM adapter; every other operation delegates unchanged."""
    def __init__(self, communicator, transport):
        self.communicator = communicator
        self.transport = transport

    def __getattr__(self, name):
        return getattr(self.communicator, name)

    def all_gather(self, input_, dim=-1):
        import torch
        import habana_frameworks.torch.core as hc
        if not -input_.ndim <= dim < input_.ndim:
            raise ValueError('Invalid gather dimension')
        dim %= input_.ndim
        size = tuple(input_.shape)
        world = len(self.transport.ranks)
        output = torch.empty((size[0]*world,) + size[1:], dtype=input_.dtype, device=input_.device)
        hc.mark_step()
        self.transport.all_gather_into_tensor(output, input_)
        return output.reshape((world,) + size).movedim(0,dim).reshape(
            size[:dim] + (size[dim]*world,) + size[dim+1:])
