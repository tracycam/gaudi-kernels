"""Single-card BF16 add with partial vectors and bucketed row counts."""

import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import gaudirt


def audit():
    assert "habana_frameworks" not in sys.modules
    with open("/proc/self/maps") as maps:
        assert "habana_pytorch" not in maps.read().lower()


def main():
    torch.random.default_generator.manual_seed(282)
    gaudirt.initialize(hbm_bytes=512 * 1024 * 1024, pinned_bytes=4 * 1024 * 1024)
    for width, rows in ((1, 1), (7, 3), (127, 7), (128, 1), (129, 7),
                        (7168, 7), (22016, 137)):
        left_cpu = torch.randn(rows, width, dtype=torch.bfloat16)
        right_cpu = torch.randn(rows, width, dtype=torch.bfloat16)
        left, right = left_cpu.to("gaudirt:0"), right_cpu.to("gaudirt:0")
        program = gaudirt._C.BucketProgram([1, 1], [[1, 0, 1, -1]])
        device_output = program.run([left, right])
        assert device_output.untyped_storage().nbytes() % 256 == 0
        output = device_output.cpu()
        expected = (left_cpu.float() + right_cpu.float()).to(torch.bfloat16)
        assert torch.isfinite(output).all() and torch.isfinite(expected).all()
        mismatch = torch.count_nonzero(output.view(torch.int16) != expected.view(torch.int16)).item()
        assert mismatch == 0, (width, rows, mismatch)
        print("width", width, "rows", rows, "bit_differences", mismatch, flush=True)
        if width == 129:
            same = program.run([left, left]).cpu()
            expected_same = (left_cpu.float() + left_cpu.float()).to(torch.bfloat16)
            assert torch.equal(same.view(torch.int16), expected_same.view(torch.int16))
    del left, right, device_output, program
    audit()
    gaudirt.empty_cache()
    gaudirt.shutdown()
    audit()
    print("generic BF16 add passed", flush=True)


if __name__ == "__main__":
    main()
