"""CPU dtype witness for the archived Gaudi dynamic activation quantizer.

This checks scale construction, not HPU FP8 conversion or model quality.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[3]
    sources = [
        workspace / "instance-assets/native-executor-20260925/deployment/plugin/vllm_gaudi/extension/ops.py",
        workspace / "tpc-dev/vllm-native-20260925/executor/precision_runtime.py",
    ]
    rows = []
    limit = torch.finfo(torch.float8_e4m3fnuz).max
    for peak in (0.0, 0.001, 1.0, 16.0):
        x = torch.tensor([[peak, -peak / 2]], dtype=torch.bfloat16)
        amax = x.abs().max(dim=-1, keepdim=True).values
        legacy_scale = (amax + 1e-8) / limit
        legacy_reciprocal = 1.0 / legacy_scale
        # Preserve the old epsilon and range; change only calculation dtype.
        f32_scale = (amax.float() + 1e-8) / limit
        f32_reciprocal = 1.0 / f32_scale
        assert legacy_scale.dtype == legacy_reciprocal.dtype == torch.bfloat16
        rows.append({
            "input_bf16": x.float().tolist(),
            "amax_dtype": str(amax.dtype),
            "legacy_scale_dtype": str(legacy_scale.dtype),
            "legacy_reciprocal_dtype": str(legacy_reciprocal.dtype),
            "legacy_scale_exported_f32": legacy_scale.float().item(),
            "f32_scale_same_formula": f32_scale.item(),
            "legacy_reciprocal_exported_f32": legacy_reciprocal.float().item(),
            "f32_reciprocal_same_formula": f32_reciprocal.item(),
            "scale_relative_difference": ((legacy_scale.float() - f32_scale).abs() / f32_scale).item(),
        })
    assert rows[2]["legacy_scale_exported_f32"] != rows[2]["f32_scale_same_formula"]
    result = {
        "scope": "CPU PyTorch dtype/rounding witness; no HPU/GPU execution, FP8 byte or model-quality claim",
        "torch_version": torch.__version__,
        "fp8_max_from_archived_gaudi2_expression": limit,
        "sources": [{"path": str(p.relative_to(workspace)),
                     "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources],
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
