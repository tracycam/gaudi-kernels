"""Offline K32 quantizer comparison against an explicitly supplied vLLM tree.

Only the pure torch reference function is extracted; records its source hash
and tree revision. This neither imports GPU runtime code nor qualifies MME.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import torch
from gaudi_kernels.mxfp4_reference import quantize_mxfp8


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--vllm', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    src = a.vllm/'vllm/model_executor/layers/quantization/utils/mxfp8_utils.py'
    function = next(n for n in ast.parse(src.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == '_mxfp8_e4m3_quantize_torch')
    env = dict(torch=torch, MXFP8_BLOCK_SIZE=32, MXFP8_VALUE_DTYPE=torch.float8_e4m3fn)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(src), 'exec'), env)
    torch.manual_seed(732)
    cases = []
    for m,k in [(1,32),(3,64),(20,256),(512,256),(1024,32)]:
        x = (torch.randn(m,k)*torch.exp2(torch.randint(-14,14,(m,k//32)).repeat_interleave(32,-1).float())).bfloat16()
        q,s = quantize_mxfp8(x)
        reference,scales = env[function.name](x)
        cases.append(dict(m=m, k=k, code_equal=torch.equal(q.view(torch.uint8),reference.view(torch.uint8)),
                          scale_equal=torch.equal(s,scales)))
    result = dict(scope='CPU MXFP8 quantizer comparison, not native FP8 MME qualification',
                  reference_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
                  reference_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.vllm,text=True).strip(),
                  cases=cases)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('x') as f:
        f.write(json.dumps(result,indent=2)+'\n')
    if not all(c['code_equal'] and c['scale_equal'] for c in cases):
        raise SystemExit('quantizer mismatch; result retained')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
