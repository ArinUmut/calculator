import torch
from marq.operator import DLRKroneckerOperator
from marq.quantizer import quantize_matrix


def test_quantizer_runs_and_improves_with_bits():
    torch.manual_seed(4)
    w=torch.randn(16,32)*.08
    da=torch.rand(32)+.2; db=torch.rand(16)+.2
    op=DLRKroneckerOperator(da,db,torch.randn(32,3)*.05,torch.randn(16,3)*.05,site_count=32,damping=1e-3)
    q2=quantize_matrix(w,op,bits=2,group_size=16,sweeps=3)
    q4=quantize_matrix(w,op,bits=4,group_size=16,sweeps=3)
    assert q2.codes.shape==w.shape
    assert q4.serialized_bytes>q2.serialized_bytes
    assert q4.surrogate_loss <= q2.surrogate_loss*1.05


def test_vectorized_weighted_bootstrap_matches_reference():
    from marq.quantizer import _weighted_scalar_candidate, signed_bounds
    from marq.scales import group_index

    torch.manual_seed(11)
    weight = torch.randn(7, 19)
    h = torch.rand_like(weight).add_(0.05)
    bits, group_size = 3, 8
    codes, scales, deq = _weighted_scalar_candidate(weight, h, bits, group_size, iterations=12)
    groups = group_index(tuple(weight.shape), group_size)
    qmin, qmax = signed_bounds(bits)
    assert codes.min() >= qmin and codes.max() <= qmax
    assert torch.all(scales > 0)
    assert torch.allclose(deq, scales[groups] * codes.float())
    # Each returned scale must be the weighted LS optimum for its frozen codes.
    for group in range(scales.numel()):
        mask = groups == group
        q = codes[mask].float()
        den = (h[mask] * q.square()).sum()
        expected = (h[mask] * weight[mask] * q).sum() / den if den > 0 else scales[group]
        assert torch.allclose(scales[group], expected.clamp_min(1e-12), rtol=2e-5, atol=2e-6)


def test_vectorized_bootstrap_large_group_count_finishes():
    from marq.quantizer import _weighted_scalar_candidate

    torch.manual_seed(12)
    weight = torch.randn(512, 1024)
    h = torch.rand_like(weight).add_(0.1)
    codes, scales, deq = _weighted_scalar_candidate(weight, h, bits=4, group_size=128, iterations=3)
    assert codes.shape == weight.shape
    assert scales.numel() == 512 * 8
    assert torch.isfinite(deq).all()
