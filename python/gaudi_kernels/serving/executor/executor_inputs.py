"""Export CPU decode input roles at their model source; no role guesses from IDs."""
import ctypes
import json
import sys


class SemanticTransfers:
    """Transfer roles supplied by the runner interface, without stack inspection."""
    def __init__(self, api, root):
        self.api, self.root, self.records = api, root, []
        self.api.e1_set_semantic_role.argtypes = [ctypes.c_char_p]
        self.collect = None
        self.audit_inputs = None

    def upload(self, role, source, callback, **kwargs):
        if self.audit_inputs is not None:
            self.audit_inputs[role] = source.clone()
        if self.collect is not None:
            self.collect[role] = source
            return source
        if not self.api.e1_is_active():
            return callback(role, source, **kwargs)
        self.records.append({'role': role, 'shape': list(source.shape), 'dtype': str(source.dtype),
                             'values': source.flatten().tolist()})
        self.api.e1_set_semantic_role(role.encode())
        try:
            return callback(role, source, **kwargs)
        finally:
            self.api.e1_set_semantic_role(None)

    def download(self, role, source, callback):
        if not self.api.e1_is_active():
            return callback(role, source)
        self.records.append({'role': role, 'shape': list(source.shape), 'dtype': str(source.dtype),
                             'direction': 'output'})
        self.api.e1_set_semantic_role(role.encode())
        try:
            return callback(role, source)
        finally:
            self.api.e1_set_semantic_role(None)

    def reset(self, root):
        self.root = root
        self.records.clear()

    def __call__(self):
        import torch.distributed as dist
        path = self.root / f'rank{dist.get_rank()}-input-semantics.json'
        path.write_text(json.dumps(self.records, indent=2)+'\n')
        return self.records

def install(module, api, root):
    import torch
    import torch.distributed as dist
    original = module.async_h2d_copy
    roles = {'positions', 'block_list', 'block_usage', 'block_groups', 'slot_mapping', 'window_block_list', 'window_block_usage', 'window_block_groups', 'token_ids', 'logits_indices'}
    records = []
    api.e1_set_semantic_role.argtypes = [ctypes.c_char_p]

    def copy(source, *args, **kwargs):
        if not api.e1_is_active():
            return original(source, *args, **kwargs)
        frame = sys._getframe(1)
        names = [name for name in roles if frame.f_locals.get(name) is source]
        if frame.f_code.co_name != '_create_decode_input_data' or len(names) != 1:
            return original(source, *args, **kwargs)
        assert isinstance(source, torch.Tensor) and source.device.type == 'cpu'
        role = names[0]
        records.append({'role': role, 'shape': list(source.shape), 'dtype': str(source.dtype), 'values': source.flatten().tolist(), 'source_file': frame.f_code.co_filename, 'line': frame.f_lineno})
        api.e1_set_semantic_role(role.encode())
        try:
            return original(source, *args, **kwargs)
        finally:
            api.e1_set_semantic_role(None)
    module.async_h2d_copy = copy

    def save():
        path = root / f'rank{dist.get_rank()}-input-semantics.json'
        path.write_text(json.dumps(records, indent=2) + '\n')
        return records

    def reset(new_root):
        nonlocal root
        root = new_root
        records.clear()
    save.reset = reset
    return save
