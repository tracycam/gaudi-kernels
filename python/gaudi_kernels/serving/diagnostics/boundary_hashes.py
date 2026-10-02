"""Untimed producer-boundary capture without Module hooks or tensor aliasing.

Clones are issued at the actual producer before scratch storage can be reused.
The inactive path imports no Torch and performs no tensor operation. A separate
plain invocation must prove that diagnostics preserve complete logits.
"""
from contextlib import contextmanager
import hashlib
import re
import threading
_LOCAL = threading.local()

def layer_index(name):
    match = re.search('(?:^|\\.)layers\\.(\\d+)(?:\\.|$)', str(name))
    if match is None:
        raise ValueError('Boundary producer has no layer identity: ' + str(name))
    return int(match[1])

def active():
    return getattr(_LOCAL, 'session', None) is not None

def emit(layer, stage, value):
    session = getattr(_LOCAL, 'session', None)
    if session is None:
        return value
    import torch
    if type(layer) is not int:
        layer = layer_index(layer)
    key = (layer, stage)
    if key in session.values:
        raise RuntimeError('Duplicate producer boundary: ' + str(key))
    if not isinstance(value, torch.Tensor) or value.numel() > 65536:
        raise ValueError('Boundary exceeds bounded decode tensor contract')
    session.values[key] = value.detach().clone()
    return value

class Session:

    def __init__(self, model, layers, rank, path):
        (self.layers, self.rank, self.path) = (layers, rank, path)
        self.values = {}
        owners = {}
        for (name, module) in model.named_modules(remove_duplicate=False):
            if type(module).__name__ == 'NativeExpertTP':
                index = layer_index(name)
                if id(module) in owners and owners[id(module)] != index:
                    raise ValueError('One expert module assigned to different model layers')
                owners[id(module)] = index
                module._boundary_layer_index = index
        if set(owners.values()) != set(range(1, layers)):
            raise ValueError('Incomplete dense-layer0/MoE-layer1..L-1 inventory')

    @contextmanager
    def recording(self):
        if active() or self.values:
            raise RuntimeError('Nested or reused boundary session')
        _LOCAL.session = self
        try:
            yield self
        finally:
            _LOCAL.session = None

    def flush(self, token_ids, positions):
        import json
        import torch
        import habana_frameworks.torch.core as hc
        expected = {(layer, stage) for layer in range(self.layers) for stage in ('qkv', 'attention', 'layer_output')}
        expected |= {(layer, 'moe_local') for layer in range(1, self.layers)}
        if set(self.values) != expected:
            raise RuntimeError('Boundary coverage differs: missing=' + str(sorted(expected - set(self.values))) + ', extra=' + str(sorted(set(self.values) - expected)))
        hc.mark_step()
        torch.hpu.synchronize()
        rows = []
        for ((layer, stage), tensor) in sorted(self.values.items()):
            host = tensor.cpu().contiguous()
            rows.append({'layer': layer, 'stage': stage, 'shape': list(host.shape), 'dtype': str(host.dtype), 'sha256': hashlib.sha256(host.view(torch.uint8).numpy().tobytes()).hexdigest()})
        result = {'schema_version': 1, 'rank': self.rank, 'layers': self.layers, 'records': rows, 'input_ids': token_ids.detach().cpu().flatten().tolist(), 'positions': positions.detach().cpu().flatten().tolist(), 'module_hooks_used': False, 'plain_logits_equivalence_required': True, 'scope': 'Untimed decode producers; hashes alone do not qualify sampling equivalence'}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise FileExistsError(self.path)
        self.path.write_text(json.dumps(result, indent=2) + '\n')
        self.values.clear()
        return result
