"""Explicit shared position boundary for the qualified Lazy QKV consumer."""


def adapter_class(base):
    if not callable(getattr(base, 'flatten_positions', None)):
        raise RuntimeError('HPU model adapter lacks the explicit position boundary interface')

    class CanonicalModelAdapter(base):
        def flatten_positions(self, positions):
            import torch
            # A graph-owned output, shared by every layer. No per-layer clone
            # or synchronization is needed, and I64 keeps the SDK's I32 edge.
            return torch.ops.gaudi_swa128.reshape(positions, [positions.numel()])

    return CanonicalModelAdapter
