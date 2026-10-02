"""Independent numerical fixtures for the diagnostic, not device qualification."""
import torch
from gaudi_kernels.block_fp8_oracle import reference


def test_distinct_k_block_scales_survive():
    x = torch.ones((1, 256), dtype=torch.bfloat16)
    w = torch.cat((torch.ones((1, 128)), torch.full((1, 128), 2.0)), dim=1).to(torch.float8_e4m3fn)
    scales = torch.tensor([[.5, 2.]])
    assert reference(x, w, scales).item() == 576.0
    assert reference(x, w, scales, native_half=True).item() == 576.0


def test_original_and_adapted_reference_distinguish_lost_subnormal():
    x = torch.zeros((1, 128), dtype=torch.bfloat16); x[0, 0] = 1
    w = torch.zeros((1, 128)); w[0, 0] = 1/512; w[0, 1] = 448
    w = w.to(torch.float8_e4m3fn)
    scales = torch.ones((1, 1))
    assert reference(x, w, scales).item() == 1/512
    assert reference(x, w, scales, native_half=True).item() == 0


def test_tail_and_n_scale_boundary():
    x = torch.zeros((1, 129), dtype=torch.bfloat16); x[0, 128] = 1
    w = torch.ones((129, 129)).to(torch.float8_e4m3fn)
    scales = torch.tensor([[99., 2.], [37., 4.]])
    y = reference(x, w, scales, bias=torch.ones(129))
    assert torch.equal(y[0, :128], torch.full((128,), 3., dtype=torch.bfloat16))
    assert y[0, 128].item() == 5
