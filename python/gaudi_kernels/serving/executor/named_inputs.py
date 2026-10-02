"""CPU tensor encoding at the captured SDK DMA boundary, with a byte gate."""
import ctypes

INPUT_ROLES = frozenset(('positions', 'slot_mapping', 'token_ids', 'logits_indices',
    'block_list', 'block_groups', 'block_usage', 'window_block_list',
    'window_block_groups', 'window_block_usage'))


class InputBinding(ctypes.Structure):
    _fields_ = [('role', ctypes.c_char_p), ('data', ctypes.c_void_p), ('bytes', ctypes.c_uint64)]


class NamedInputs:
    def __init__(self, api, records):
        import torch
        self.api, self.specs = api, {}
        inputs = [r for r in records if r.get('direction') != 'output']
        if len(inputs) != len(INPUT_ROLES) or {r['role'] for r in inputs} != INPUT_ROLES:
            raise ValueError('Capture requires one upload for every named decode input')
        for r in inputs:
            role = r['role']
            count = ctypes.c_uint64()
            if api.native_multi_input_bytes(role.encode(), ctypes.byref(count)):
                raise ValueError('Missing SDK input: ' + role)
            tensor = torch.tensor(r['values'], dtype=getattr(torch, r['dtype'].removeprefix('torch.')))
            dtype = tensor.dtype
            if tensor.numel() * tensor.element_size() != count.value:
                if dtype == torch.int64 and count.value == tensor.numel() * 4:
                    dtype = torch.int32  # Qualified HPU bridge lowers long indices.
                else:
                    raise ValueError('Unsupported SDK input encoding: ' + role)
            self.specs[role] = (tuple(r['shape']), r['dtype'], dtype, count.value)
            encoded = self.encode(role, tensor.reshape(r['shape']))
            if api.native_multi_compare_input(role.encode(), encoded.data_ptr(), count.value):
                raise ValueError('CPU encoding differs from captured DMA: ' + role)

    def encode(self, role, tensor):
        import torch
        shape, source_dtype, dma_dtype, count = self.specs[role]
        if tensor.device.type != 'cpu' or tuple(tensor.shape) != shape or str(tensor.dtype) != source_dtype:
            raise ValueError('Decode input shape/dtype changed: ' + role)
        if tensor.dtype == torch.int64 and dma_dtype == torch.int32:
            if tensor.numel() and (tensor.min().item() < -(2**31) or tensor.max().item() >= 2**31):
                raise ValueError('SDK index overflow: ' + role)
        result = tensor.to(dtype=dma_dtype).contiguous()
        if result.numel() * result.element_size() != count:
            raise ValueError('Decode input byte count changed: ' + role)
        return result

    def bind(self, inputs):
        if set(inputs) != INPUT_ROLES:
            raise ValueError('Incomplete decode input set')
        owners = [(role.encode(), self.encode(role, inputs[role])) for role in sorted(inputs)]
        bindings = (InputBinding * len(owners))(*(
            InputBinding(role, tensor.data_ptr(), tensor.numel() * tensor.element_size())
            for role, tensor in owners))
        return bindings, owners
