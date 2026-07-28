from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import Sequence
import math
import numpy as np


@dataclass(frozen=True)
class AllocationCandidate:
    key: str
    format_name: str
    cost_bytes: int
    loss: float
    payload: object | None = None


@dataclass
class AllocationResult:
    selected: list[AllocationCandidate]
    total_cost: int
    total_loss: float
    lambda_value: float


def _pareto_banks(banks: Sequence[Sequence[AllocationCandidate]]) -> list[list[AllocationCandidate]]:
    if not banks or any(not bank for bank in banks):
        raise ValueError("every unit needs candidates")
    result: list[list[AllocationCandidate]] = []
    for bank in banks:
        if any(c.cost_bytes < 0 or not math.isfinite(c.loss) for c in bank):
            raise ValueError("candidate costs/losses must be finite and non-negative in cost")
        by_cost: dict[int, AllocationCandidate] = {}
        for candidate in bank:
            old = by_cost.get(candidate.cost_bytes)
            if old is None or candidate.loss < old.loss:
                by_cost[candidate.cost_bytes] = candidate
        ordered = [by_cost[cost] for cost in sorted(by_cost)]
        keep: list[AllocationCandidate] = []
        best_loss = float("inf")
        for candidate in ordered:
            if candidate.loss < best_loss:
                keep.append(candidate)
                best_loss = candidate.loss
        result.append(keep)
    return result


def exact_mckp_allocate(banks: Sequence[Sequence[AllocationCandidate]], budget_bytes: int, *, max_dense_states: int = 500_000) -> AllocationResult:
    """Exact multiple-choice knapsack under a serialized-byte inequality."""
    pareto = _pareto_banks(banks)
    base_cost = sum(bank[0].cost_bytes for bank in pareto)
    if base_cost > budget_bytes:
        raise ValueError(f"budget infeasible: minimum {base_cost} > {budget_bytes}")
    positive_deltas = [
        candidate.cost_bytes - bank[0].cost_bytes
        for bank in pareto for candidate in bank[1:]
        if candidate.cost_bytes > bank[0].cost_bytes
    ]
    if not positive_deltas:
        selected = [bank[0] for bank in pareto]
        return AllocationResult(selected, base_cost, sum(c.loss for c in selected), float("nan"))
    quantum = positive_deltas[0]
    for delta in positive_deltas[1:]:
        quantum = gcd(quantum, delta)
    capacity = (budget_bytes - base_cost) // quantum
    choices = []
    for bank in pareto:
        base = bank[0]
        choices.append([
            ((candidate.cost_bytes - base.cost_bytes) // quantum, candidate.loss - base.loss, index)
            for index, candidate in enumerate(bank)
        ])

    if capacity <= max_dense_states:
        dp = np.full(capacity + 1, np.inf, dtype=np.float64)
        dp[0] = 0.0
        parent_cost = np.full((len(pareto), capacity + 1), -1, dtype=np.int32)
        parent_choice = np.full((len(pareto), capacity + 1), -1, dtype=np.int16)
        for unit, unit_choices in enumerate(choices):
            nxt = np.full_like(dp, np.inf)
            for delta_cost, delta_loss, candidate_index in unit_choices:
                if delta_cost > capacity:
                    continue
                proposed = dp[: capacity + 1 - delta_cost] + delta_loss
                target = nxt[delta_cost:]
                improve = proposed < target
                if np.any(improve):
                    target[improve] = proposed[improve]
                    locations = np.nonzero(improve)[0] + delta_cost
                    parent_cost[unit, locations] = locations - delta_cost
                    parent_choice[unit, locations] = candidate_index
            dp = nxt
        feasible = np.flatnonzero(np.isfinite(dp))
        if feasible.size == 0:
            raise RuntimeError("exact DP produced no feasible state")
        cursor = int(feasible[np.argmin(dp[feasible])])
        indices = [0] * len(pareto)
        for unit in range(len(pareto) - 1, -1, -1):
            candidate_index = int(parent_choice[unit, cursor])
            if candidate_index < 0:
                raise RuntimeError("broken dense-DP backpointer")
            indices[unit] = candidate_index
            cursor = int(parent_cost[unit, cursor])
    else:
        frontiers: list[dict[int, tuple[float, int, int]]] = []
        previous: dict[int, float] = {0: 0.0}
        for unit_choices in choices:
            raw: dict[int, tuple[float, int, int]] = {}
            for old_cost, old_loss in previous.items():
                for delta_cost, delta_loss, candidate_index in unit_choices:
                    new_cost = old_cost + delta_cost
                    if new_cost > capacity:
                        continue
                    new_loss = old_loss + delta_loss
                    old = raw.get(new_cost)
                    if old is None or new_loss < old[0]:
                        raw[new_cost] = (new_loss, old_cost, candidate_index)
            pruned: dict[int, tuple[float, int, int]] = {}
            best_loss = float("inf")
            for cost in sorted(raw):
                entry = raw[cost]
                if entry[0] < best_loss:
                    pruned[cost] = entry
                    best_loss = entry[0]
            frontiers.append(pruned)
            previous = {cost: entry[0] for cost, entry in pruned.items()}
        cursor = min(previous, key=lambda cost: previous[cost])
        indices = [0] * len(pareto)
        for unit in range(len(pareto) - 1, -1, -1):
            _, old_cost, candidate_index = frontiers[unit][cursor]
            indices[unit] = candidate_index
            cursor = old_cost

    selected = [pareto[unit][index] for unit, index in enumerate(indices)]
    total_cost = sum(candidate.cost_bytes for candidate in selected)
    if total_cost > budget_bytes:
        raise RuntimeError("exact allocator violated budget")
    return AllocationResult(selected, total_cost, sum(c.loss for c in selected), float("nan"))


def lagrangian_allocate(banks: Sequence[Sequence[AllocationCandidate]], budget_bytes: int, *, iterations: int = 80) -> AllocationResult:
    """Fast dual + greedy allocator retained as an explicit approximation."""
    pareto = _pareto_banks(banks)
    min_cost = sum(bank[0].cost_bytes for bank in pareto)
    if min_cost > budget_bytes:
        raise ValueError(f"budget infeasible: minimum {min_cost} > {budget_bytes}")
    def choose(lam: float) -> list[int]:
        return [min(range(len(bank)), key=lambda i: bank[i].loss + lam * bank[i].cost_bytes) for bank in pareto]
    lo, hi = 0.0, 1.0
    while sum(pareto[u][i].cost_bytes for u, i in enumerate(choose(hi))) > budget_bytes:
        hi *= 2.0
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        idx = choose(mid)
        cost = sum(pareto[u][i].cost_bytes for u, i in enumerate(idx))
        if cost > budget_bytes:
            lo = mid
        else:
            hi = mid
    lam = hi
    idx = choose(lam)
    cost = sum(pareto[u][i].cost_bytes for u, i in enumerate(idx))
    while True:
        best = None
        for unit, bank in enumerate(pareto):
            i = idx[unit]
            for j in range(i + 1, len(bank)):
                delta_cost = bank[j].cost_bytes - bank[i].cost_bytes
                gain = bank[i].loss - bank[j].loss
                if delta_cost <= 0 or cost + delta_cost > budget_bytes or gain <= 0:
                    continue
                proposal = (gain / delta_cost, gain, -delta_cost, unit, j)
                if best is None or proposal > best:
                    best = proposal
        if best is None:
            break
        _, _, _, unit, j = best
        cost += pareto[unit][j].cost_bytes - pareto[unit][idx[unit]].cost_bytes
        idx[unit] = j
    selected = [pareto[u][i] for u, i in enumerate(idx)]
    return AllocationResult(selected, cost, sum(c.loss for c in selected), lam)
