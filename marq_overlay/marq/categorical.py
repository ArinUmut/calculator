from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor

from .operator import DLRKroneckerOperator


def _validate(values: Tensor, probs: Tensor) -> None:
    if values.shape != probs.shape or values.ndim != 3:
        raise ValueError("values/probs must have shape [m,n,k]")
    if torch.any(probs < 0):
        raise ValueError("negative probability")
    err = (probs.sum(dim=-1) - 1).abs().max().item()
    if err > 2e-4:
        raise ValueError(f"probabilities do not sum to 1, max error={err}")


def moments(values: Tensor, probs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    _validate(values, probs)
    mean = (probs * values).sum(dim=-1)
    second = (probs * values.square()).sum(dim=-1)
    variance = (second - mean.square()).clamp_min(0)
    return mean, second, variance


def entropy_term(probs: Tensor, tau: float) -> Tensor:
    if tau == 0:
        return probs.new_zeros(())
    p = probs.float()
    return tau * torch.where(p > 0, p * p.log(), torch.zeros_like(p)).sum()


@dataclass
class _MomentState:
    mean: Tensor
    second: Tensor
    hmean: Tensor


def _state(values: Tensor, probs: Tensor, op: DLRKroneckerOperator, *, include_damping: bool) -> _MomentState:
    mean, second, _ = moments(values, probs)
    return _MomentState(mean, second, op.apply(mean, include_damping=include_damping))


def _state_objective(state: _MomentState, probs: Tensor, linear: Tensor, h: Tensor, tau: float) -> Tensor:
    variance = (state.second - state.mean.square()).clamp_min(0)
    value = (linear * state.mean).sum()
    value = value + 0.5 * (state.mean * state.hmean).sum()
    value = value + 0.5 * (h * variance).sum()
    return value + entropy_term(probs, tau)


def expected_objective(values: Tensor, probs: Tensor, linear: Tensor, op: DLRKroneckerOperator, *, tau: float, include_damping: bool = True) -> Tensor:
    state = _state(values, probs, op, include_damping=include_damping)
    h = op.diagonal(include_damping=include_damping)
    return _state_objective(state, probs, linear, h, tau)


def cavity_costs(values: Tensor, probs: Tensor, linear: Tensor, op: DLRKroneckerOperator, *, include_damping: bool = True) -> Tensor:
    state = _state(values, probs, op, include_damping=include_damping)
    h = op.diagonal(include_damping=include_damping)
    field_without_self = linear + state.hmean - h * state.mean
    return values * field_without_self[..., None] + 0.5 * h[..., None] * values.square()


def best_response(costs: Tensor, tau: float) -> Tensor:
    if tau <= 0:
        idx = costs.argmin(dim=-1, keepdim=True)
        return torch.zeros_like(costs).scatter_(-1, idx, 1.0)
    return torch.softmax((-costs / tau).float(), dim=-1).to(costs.dtype)


def _line_derivative(beta: float, p0: Tensor, dp: Tensor, a1: Tensor, a2: Tensor, tau: float, floor: float) -> float:
    out = a1 + 2.0 * a2 * beta
    if tau:
        p = (p0.float() + beta * dp.float()).clamp_min(floor)
        out = out + tau * (dp.float() * p.log()).sum()
    return float(out)


def _line_coefficients(state0: _MomentState, state1: _MomentState, linear: Tensor, h: Tensor) -> tuple[Tensor, Tensor]:
    delta_mean = state1.mean - state0.mean
    delta_second = state1.second - state0.second
    h_delta = state1.hmean - state0.hmean
    a1 = (linear * delta_mean).sum()
    a1 = a1 + (delta_mean * (state0.hmean - h * state0.mean)).sum()
    a1 = a1 + 0.5 * (h * delta_second).sum()
    a2 = 0.5 * (delta_mean * (h_delta - h * delta_mean)).sum()
    return a1, a2


def _select_beta(p0: Tensor, p1: Tensor, a1: Tensor, a2: Tensor, *, tau: float, floor: float, iterations: int) -> float:
    dp = p1 - p0
    if tau <= 0:
        if float(a2) > 0:
            return float(torch.clamp(-a1 / (2.0 * a2), 0.0, 1.0))
        return 1.0 if float(a1 + a2) <= 0 else 0.0
    d0 = _line_derivative(0.0, p0, dp, a1, a2, tau, floor)
    if d0 >= 0:
        return 0.0
    if float(a2) <= 0:
        return 1.0
    d1 = _line_derivative(1.0, p0, dp, a1, a2, tau, floor)
    if d1 <= 0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if _line_derivative(mid, p0, dp, a1, a2, tau, floor) <= 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def exact_parallel_line_search(values: Tensor, p0: Tensor, p1: Tensor, linear: Tensor, op: DLRKroneckerOperator, *, tau: float, include_damping: bool = True, floor: float = 1e-30, iterations: int = 36) -> tuple[Tensor, float]:
    state0 = _state(values, p0, op, include_damping=include_damping)
    state1 = _state(values, p1, op, include_damping=include_damping)
    h = op.diagonal(include_damping=include_damping)
    a1, a2 = _line_coefficients(state0, state1, linear, h)
    beta = _select_beta(p0, p1, a1, a2, tau=tau, floor=floor, iterations=iterations)
    return p0 + beta * (p1 - p0), beta


@dataclass
class MeanFieldResult:
    probs: Tensor
    objective_history: list[float]
    step_history: list[float]


def optimize_mean_field(values: Tensor, probs: Tensor, linear: Tensor, op: DLRKroneckerOperator, *, tau: float, sweeps: int = 8, include_damping: bool = True, tolerance: float = 1e-7) -> MeanFieldResult:
    _validate(values, probs)
    h = op.diagonal(include_damping=include_damping)
    p = probs
    state0 = _state(values, p, op, include_damping=include_damping)
    history = [float(_state_objective(state0, p, linear, h, tau))]
    steps: list[float] = []
    for _ in range(sweeps):
        field_without_self = linear + state0.hmean - h * state0.mean
        costs = values * field_without_self[..., None] + 0.5 * h[..., None] * values.square()
        target = best_response(costs, tau)
        state1 = _state(values, target, op, include_damping=include_damping)
        a1, a2 = _line_coefficients(state0, state1, linear, h)
        beta = _select_beta(p, target, a1, a2, tau=tau, floor=1e-30, iterations=36)
        p = p + beta * (target - p)
        state0 = _MomentState(
            state0.mean + beta * (state1.mean - state0.mean),
            state0.second + beta * (state1.second - state0.second),
            state0.hmean + beta * (state1.hmean - state0.hmean),
        )
        obj = float(_state_objective(state0, p, linear, h, tau))
        if obj > history[-1] + 1e-5:
            raise RuntimeError(f"line search increased objective: {history[-1]} -> {obj}")
        history.append(obj)
        steps.append(beta)
        if abs(history[-2] - history[-1]) <= tolerance * max(1.0, abs(history[-2])):
            break
    return MeanFieldResult(p, history, steps)
