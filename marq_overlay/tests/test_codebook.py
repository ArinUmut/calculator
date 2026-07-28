import torch

from marq.quantizer import _weighted_scalar_candidate, signed_bounds


def test_codebooks_are_sign_symmetric():
    assert signed_bounds(2) == (-1, 1)
    assert signed_bounds(3) == (-3, 3)
    assert signed_bounds(4) == (-7, 7)


def test_two_bit_bootstrap_has_no_sign_range_bias():
    torch.manual_seed(13)
    half = torch.rand(64, 128)
    weight = torch.cat([-half, half], dim=0)
    h = torch.ones_like(weight)
    _, _, dequantized = _weighted_scalar_candidate(
        weight, h, bits=2, group_size=128, iterations=8
    )
    assert abs(float(dequantized.mean())) < 1e-4
    assert torch.allclose(dequantized[:64], -dequantized[64:], atol=1e-6, rtol=0)
