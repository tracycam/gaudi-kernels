import ctypes
import unittest
import torch
from gaudi_kernels.serving.executor.named_inputs import INPUT_ROLES, NamedInputs


class SDKCapture:
    def __init__(self, records):
        self.tensors = {r['role']:torch.tensor(r['values'],
            dtype=torch.bfloat16 if r['role'].endswith('usage') else torch.int32) for r in records}

    def native_multi_input_bytes(self, role, out):
        value = self.tensors[role.decode()]
        ctypes.cast(out, ctypes.POINTER(ctypes.c_uint64))[0] = value.numel()*value.element_size()
        return 0

    def native_multi_compare_input(self, role, pointer, count):
        value = self.tensors[role.decode()].view(torch.uint8).numpy().tobytes()
        return 0 if ctypes.string_at(pointer, count) == value else -2


class NamedInputTests(unittest.TestCase):
    def fixtures(self):
        inputs = {name:torch.tensor([1,2], dtype=torch.bfloat16 if name.endswith('usage') else torch.int64)
                  for name in INPUT_ROLES}
        records = [{'role':name, 'shape':list(v.shape), 'dtype':str(v.dtype), 'values':v.tolist()}
                   for name,v in inputs.items()]
        return inputs, records

    def test_actual_dma_bytes_and_integer_lowering(self):
        inputs, records = self.fixtures()
        plan = NamedInputs(SDKCapture(records), records)
        bindings, owners = plan.bind(inputs)
        self.assertEqual(len(bindings), 10)
        self.assertEqual({name.decode():v.dtype for name,v in owners},
            {name:torch.bfloat16 if name.endswith('usage') else torch.int32 for name in inputs})

    def test_shape_overflow_and_missing_input_are_rejected(self):
        inputs, records = self.fixtures()
        plan = NamedInputs(SDKCapture(records), records)
        inputs['positions'] = torch.tensor([2**31,1], dtype=torch.int64)
        with self.assertRaisesRegex(ValueError, 'overflow'):
            plan.bind(inputs)
        inputs['positions'] = torch.tensor([1], dtype=torch.int64)
        with self.assertRaisesRegex(ValueError, 'shape/dtype'):
            plan.bind(inputs)
        del inputs['positions']
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            plan.bind(inputs)

    def test_unexpected_bridge_conversion_fails_capture_byte_gate(self):
        inputs, records = self.fixtures()
        api = SDKCapture(records)
        api.tensors['positions'][0] = 999
        with self.assertRaisesRegex(ValueError, 'differs from captured DMA'):
            NamedInputs(api, records)
