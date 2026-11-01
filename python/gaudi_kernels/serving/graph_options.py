"""Application graph scope; independent of model/operator policy manifests."""
from dataclasses import dataclass


@dataclass(frozen=True)
class GraphOptions:
    scope: str = 'forward'
    reuse_inputs: bool = True
    max_input_buckets: int = 128
    max_cached_graphs: int | None = None

    def __post_init__(self):
        if self.scope not in ('forward', 'step') or type(self.scope) is not str:
            raise ValueError('hpu_graph.scope must be forward or step')
        if type(self.reuse_inputs) is not bool:
            raise ValueError('hpu_graph.reuse_inputs must be bool')
        if type(self.max_input_buckets) is not int or self.max_input_buckets < 1:
            raise ValueError('hpu_graph.max_input_buckets must be a positive integer')
        if self.max_cached_graphs is not None and (type(self.max_cached_graphs) is not int or self.max_cached_graphs < 1):
            raise ValueError('hpu_graph.max_cached_graphs must be a positive integer or null')

    @classmethod
    def from_vllm(cls, config):
        additional = getattr(config, 'additional_config', None)
        if additional is None:
            additional = {}
        if type(additional) is not dict:
            raise ValueError('additional_config must be a mapping')
        document = additional.get('hpu_graph', {})
        if type(document) is not dict or set(document) - {'scope', 'reuse_inputs', 'max_input_buckets', 'max_cached_graphs'}:
            raise ValueError('Unknown hpu_graph options')
        return cls(**document)
