from __future__ import annotations

from dataclasses import dataclass
import math
import torch
from torch import Tensor

from .categorical import optimize_mean_field
from .operator import DLRKroneckerOperator
from .scales import group_index


@dataclass
class QuantizedCandidate:
    bits: int
    group_size: int
    codes: Tensor
    scales: Tensor
    dequantized: Tensor
    surrogate_loss: float
    serialized_bytes: int
    mean_field_history: list[float]


def signed_bounds(bits: int) -> tuple[int, int]:
    if bits < 2 or bits > 8:
        raise ValueError("bits must be in [2,8]")
    return -(1 << (bits - 1)), (1 << (bits - 1)) - 1


def _group_sum(values: Tensor, groups: Tensor, group_count: int) -> Tensor:
    out = torch.zeros(group_count, dtype=values.dtype, device=values.device)
    out.scatter_add_(0, groups.reshape(-1), values.reshape(-1))
    return out


def _fit_scales_for_codes(
    weight: Tensor,
    h: Tensor,
    codes: Tensor,
    groups: Tensor,
    group_count: int,
    *,
    fallback: Tensor | float = 1.0,
) -> Tensor:
    q = codes.to(weight.dtype)
    numerator = _group_sum(h * weight * q, groups, group_count)
    denominator = _group_sum(h * q.square(), groups, group_count)
    fallback_t = torch.as_tensor(fallback, dtype=weight.dtype, device=weight.device).expand(group_count)
    scales = torch.where(
        denominator > 0,
        numerator / denominator.clamp_min(torch.finfo(weight.dtype).tiny),
        fallback_t,
    )
    return scales.clamp_min(1e-12)


def _weighted_scalar_candidate(
    weight: Tensor,
    h: Tensor,
    bits: int,
    group_size: int,
    iterations: int = 8,
) -> tuple[Tensor, Tensor, Tensor]:
    """Vectorized weighted groupwise scalar bootstrap.

    The previous prototype iterated once per group in Python. Qwen3-0.6B has
    millions of groups across its matrices, so that path was structurally
    unusable. This implementation performs one segmented max and two segmented
    sums per Lloyd iteration with scatter kernels.
    """
    m, n = weight.shape
    groups = group_index((m, n), group_size, device=weight.device).to(torch.long)
    flat_groups = groups.reshape(-1)
    group_count = int(flat_groups.max().item()) + 1
    qmin, qmax = signed_bounds(bits)
    denominator = max(abs(qmin), abs(qmax), 1)

    max_abs = torch.zeros(group_count, dtype=weight.dtype, device=weight.device)
    max_abs.scatter_reduce_(
        0, flat_groups, weight.abs().reshape(-1), reduce="amax", include_self=True
    )
    scales = (max_abs / denominator).clamp_min(1e-12)
    h_safe = h.clamp_min(1e-12)

    for _ in range(iterations):
        codes = torch.round(weight / scales[groups]).clamp(qmin, qmax)
        updated = _fit_scales_for_codes(
            weight, h_safe, codes, groups, group_count, fallback=scales
        )
        relative = ((updated - scales).abs() / scales.clamp_min(1e-12)).amax()
        scales = updated
        if float(relative) <= 1e-5:
            break

    codes = torch.round(weight / scales[groups]).clamp(qmin, qmax).to(torch.int16)
    # One final scale solve makes the returned code/scale pair self-consistent.
    scales = _fit_scales_for_codes(
        weight, h_safe, codes, groups, group_count, fallback=scales
    )
    dequantized = scales[groups] * codes.to(weight.dtype)
    return codes, scales, dequantized


def _serialized_bytes(numel: int, bits: int, groups: int, scale_bytes: int = 2) -> int:
    return math.ceil(numel * bits / 8) + groups * scale_bytes


def quantize_matrix(
    weight: Tensor,
    op_pred: DLRKroneckerOperator,
    *,
    bits: int,
    group_size: int = 128,
    anchor: Tensor | None = None,
    linear: Tensor | None = None,
    tau: float = 0.03,
    sweeps: int = 5,
    local_radius: int = 1,
    damping: float | None = None,
) -> QuantizedCandidate:
    """MARQ-DLR-lite quantization for one matrix.

    The shipping path uses a weighted scalar bootstrap followed by exact
    marginalized local categorical refinement under the supplied operator.
    Scales are refit after hardening using the diagonal metric.
    """
    if weight.ndim != 2:
        raise ValueError("weight must be 2D")
    w = weight.float()
    anchor = w if anchor is None else anchor.float()
    linear = torch.zeros_like(w) if linear is None else linear.float()

    # Step operator differs from prediction operator only by damping.
    if damping is None:
        damping = op_pred.damping
    op_step = DLRKroneckerOperator(
        op_pred.diag_a.float(),
        op_pred.diag_b.float(),
        None if op_pred.low_a is None else op_pred.low_a.float(),
        None if op_pred.low_b is None else op_pred.low_b.float(),
        None if op_pred.diag_correction is None else op_pred.diag_correction.float(),
        op_pred.site_count,
        damping,
    )
    h = op_step.diagonal(include_damping=True).clamp_min(1e-12)
    codes, scales, deq = _weighted_scalar_candidate(w, h, bits, group_size)
    groups = group_index(tuple(w.shape), group_size, device=w.device)
    qmin, qmax = signed_bounds(bits)

    # Candidate values are scale times neighboring integer codes.
    offsets = torch.arange(-local_radius, local_radius + 1, device=w.device)
    qvals = (codes[..., None].to(torch.int32) + offsets).clamp(qmin, qmax)
    values = scales[groups][..., None] * qvals.to(w.dtype) - anchor[..., None]
    # Initialize around current code with finite support and BF16-safe entropy.
    k = values.shape[-1]
    probs = torch.full_like(values, 0.05 / max(1, k - 1))
    probs[..., local_radius] = 0.95
    probs = probs / probs.sum(dim=-1, keepdim=True)
    mf = optimize_mean_field(
        values,
        probs,
        linear,
        op_step,
        tau=tau,
        sweeps=sweeps,
        include_damping=True,
    )
    hard_idx = mf.probs.argmax(dim=-1, keepdim=True)
    hard_codes = qvals.gather(-1, hard_idx).squeeze(-1).to(torch.int16)

    # Diagonal-metric scale refit. Full scale QP is available in scales.py,
    # but this block-separable specialization is what scales to Qwen matrices.
    g_count = int(groups.max().item()) + 1
    scales2 = _fit_scales_for_codes(
        w, h, hard_codes, groups, g_count, fallback=scales
    )
    deq2 = scales2[groups] * hard_codes.float()
    e = deq2 - anchor
    loss = float((linear * e).sum() + 0.5 * op_pred.quadratic(e, include_damping=False))
    return QuantizedCandidate(
        bits=bits,
        group_size=group_size,
        codes=hard_codes.cpu(),
        scales=scales2.cpu(),
        dequantized=deq2.to(weight.dtype).cpu(),
        surrogate_loss=loss,
        serialized_bytes=_serialized_bytes(w.numel(), bits, g_count),
        mean_field_history=mf.objective_history,
    )


def refine_quantized_matrix(
    anchor_codes: Tensor,
    anchor_scales: Tensor,
    op_pred: DLRKroneckerOperator,
    linear: Tensor,
    *,
    bits: int,
    group_size: int = 128,
    tau: float = 0.02,
    sweeps: int = 4,
    code_radius: int = 1,
    scale_radius: float = 0.05,
    minimum_scale: float = 1e-12,
    row_block_size: int = 16,
) -> QuantizedCandidate:
    """Fixed-format quantized-anchor MARQ micro-step.

    The integer active set is centered at frozen anchor codes. Soft assignment
    uses H_step; hardening and predicted loss use H_pred. A bounded matrix-free
    scale QP enforces |s-s_anchor| <= scale_radius*s_anchor.
    """
    from .hardening import block_harden
    from .scales import scale_qp_cg

    codes0 = anchor_codes.to(linear.device, torch.int16)
    scales0 = anchor_scales.to(linear.device, torch.float32)
    c = linear.float()
    m, n = codes0.shape
    groups = group_index((m, n), group_size, device=linear.device)
    if int(groups.max()) + 1 != scales0.numel():
        raise ValueError("anchor scale/group mismatch")
    qmin, qmax = signed_bounds(bits)
    anchor = scales0[groups] * codes0.float()
    offsets = torch.arange(-code_radius, code_radius + 1, device=linear.device)
    qvals = (codes0[..., None].to(torch.int32) + offsets).clamp(qmin, qmax)
    values = scales0[groups][..., None] * qvals.float() - anchor[..., None]
    k = values.shape[-1]
    probs = torch.full_like(values, 1e-3 / max(1, k - 1))
    probs[..., code_radius] = 1.0 - 1e-3
    probs /= probs.sum(dim=-1, keepdim=True)
    op_step = DLRKroneckerOperator(
        op_pred.diag_a.float(), op_pred.diag_b.float(),
        None if op_pred.low_a is None else op_pred.low_a.float(),
        None if op_pred.low_b is None else op_pred.low_b.float(),
        None if op_pred.diag_correction is None else op_pred.diag_correction.float(),
        op_pred.site_count, op_pred.damping,
    )
    mf = optimize_mean_field(
        values, probs, c, op_step, tau=tau, sweeps=sweeps, include_damping=True
    )
    hardened = block_harden(
        values, mf.probs, c, op_pred, row_block_size=row_block_size, repair_passes=1
    )
    hard_idx = hardened.indices.to(linear.device)
    codes = qvals.gather(-1, hard_idx[..., None]).squeeze(-1).to(torch.int16)
    lower = (scales0 * (1.0 - scale_radius)).clamp_min(minimum_scale)
    upper = scales0 * (1.0 + scale_radius)
    scales = scale_qp_cg(
        codes.float(), anchor, c, op_step, groups,
        initial=scales0, lower=lower, upper=upper, tolerance=2e-5,
    ).float()
    deq = scales[groups] * codes.float()
    e = deq - anchor
    loss = float((c * e).sum() + 0.5 * op_pred.quadratic(e, include_damping=False))
    g_count = scales.numel()
    return QuantizedCandidate(
        bits=bits,
        group_size=group_size,
        codes=codes.cpu(),
        scales=scales.cpu(),
        dequantized=deq.to(linear.dtype).cpu(),
        surrogate_loss=loss,
        serialized_bytes=_serialized_bytes(codes.numel(), bits, g_count),
        mean_field_history=mf.objective_history + hardened.objective_history,
    )
