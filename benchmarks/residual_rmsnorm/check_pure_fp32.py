"""CPU-only dtype/control and rounding fault injections for pure RMS branch."""
import argparse
import ast
import json
from pathlib import Path
import sys
import types
import torch

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / 'python'))
from gaudi_kernels import vllm_norm

seen = []


class FloatRMS:
    @staticmethod
    def apply(x, gamma, epsilon):
        seen.append((x.dtype, gamma.dtype, tuple(x.shape)))
        assert x.dtype == gamma.dtype == torch.float32
        return (x * gamma) * torch.rsqrt(x.square().mean(-1, keepdim=True) + epsilon)


package = types.ModuleType('vllm_gaudi.extension.kernels')
package.rms_norm = lambda: FloatRMS
sys.modules[package.__name__] = package
vllm_norm.pure_rmsnorm_bf16 = lambda x,g,e: FloatRMS.apply(x.float(),g.float(),e).bfloat16()
vllm_norm.reshape_bf16_graph = lambda x,shape:x.reshape(shape)
torch.manual_seed(280932)
gamma = (torch.randn(6144) * .2 + 1).bfloat16()
module = types.SimpleNamespace(weight=gamma, variance_epsilon=1e-6)
raw, records = {}, []
for shape in [(1, 6144), (1, 2, 6144), (8, 6144), (1, 513, 6144)]:
    x = torch.randn(shape).bfloat16()
    actual = vllm_norm._pure_fp32(module, x)
    wide = FloatRMS.apply(x.float(), gamma.float(), 1e-6)
    assert actual.dtype == torch.bfloat16 and actual.shape == x.shape
    assert torch.equal(actual.view(torch.int16), wide.bfloat16().view(torch.int16))
    if shape == (1, 6144):
        square_fault = ((x.float() * gamma.float()) * torch.rsqrt(
            (x * x).float().mean(-1, keepdim=True) + 1e-6)).bfloat16()
        numerator_fault = ((x * gamma).float() * torch.rsqrt(
            x.float().square().mean(-1, keepdim=True) + 1e-6)).bfloat16()
        counts = [int((bad.view(torch.int16) != actual.view(torch.int16)).sum())
                  for bad in [square_fault, numerator_fault]]
        assert min(counts) > 0
        raw.update(x=x, gamma=gamma, pre_round=wide, output=actual,
                   bf16_square_fault=square_fault, bf16_numerator_fault=numerator_fault)
        records.append(dict(shape=shape, bf16_square_changed_outputs=counts[0],
                            bf16_numerator_changed_outputs=counts[1]))
zero = torch.zeros(1, 6144, dtype=torch.bfloat16)
zero[:, ::2] = -0.0
actual = vllm_norm._pure_fp32(module, zero)
assert torch.equal(torch.signbit(actual), torch.signbit(zero * gamma))
raw.update(signed_zero=zero, signed_zero_output=actual)
tree = ast.parse((root / 'python/gaudi_kernels/vllm_norm.py').read_text())
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_pure_fp32')
assert not any(isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add) for n in ast.walk(function))
assert not any(isinstance(n, ast.Attribute) and n.attr in
               ('clone', 'reshape', 'view', 'zeros_like', 'detach') for n in ast.walk(function))
torch.save(raw, a.output / 'raw.pt')
report = dict(status='PASS_CPU_ONLY', actual_device_verified=False,
              scope='dtype dispatch, single final BF16 boundary, fault sensitivity, signed zero; no hardware emulation',
              call_dtypes_shapes=[(str(x), str(g), s) for x, g, s in seen], records=records)
(a.output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
