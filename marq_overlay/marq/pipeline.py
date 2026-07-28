from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence
import json
import torch
from torch import Tensor

from .allocation import AllocationCandidate, AllocationResult, exact_mckp_allocate
from .calibration import CurvatureRecord
from .quantizer import quantize_matrix, refine_quantized_matrix
from .store import CandidateStore, StoredCandidate


@dataclass(frozen=True)
class BootstrapResult:
    allocation: AllocationResult
    selected: dict[str, StoredCandidate]
    target_budget_bytes: int


def build_candidate_bank(weights: Mapping[str, Tensor], curvature: Mapping[str, CurvatureRecord], store: CandidateStore, *, bits: Sequence[int] = (2, 3, 4, 5), group_size: int = 128, tau: float = 0.03, sweeps: int = 4, device: torch.device | str = "cpu") -> None:
    device = torch.device(device)
    for name, record in curvature.items():
        if name not in weights or weights[name].ndim != 2:
            continue
        op = record.operator(device=device)
        weight = weights[name].to(device)
        for bits_value in bits:
            candidate = quantize_matrix(weight, op, bits=bits_value, group_size=group_size, tau=tau, sweeps=sweeps)
            store.add(name, f"q{bits_value}_g{group_size}", candidate)
        del weight, op


def allocate_candidate_bank(store: CandidateStore, *, budget_bytes: int) -> BootstrapResult:
    units = sorted(store.banks())
    banks: list[list[AllocationCandidate]] = []
    lookup: dict[tuple[str, str], StoredCandidate] = {}
    for unit in units:
        bank: list[AllocationCandidate] = []
        for entry in store.banks()[unit]:
            lookup[(unit, entry.format_name)] = entry
            bank.append(AllocationCandidate(unit, entry.format_name, entry.serialized_bytes, entry.surrogate_loss, entry))
        banks.append(bank)
    allocation = exact_mckp_allocate(banks, budget_bytes)
    selected = {candidate.key: lookup[(candidate.key, candidate.format_name)] for candidate in allocation.selected}
    return BootstrapResult(allocation, selected, budget_bytes)


def apply_selection(model: torch.nn.Module, store: CandidateStore, selected: Mapping[str, StoredCandidate], *, dtype: torch.dtype | None = None) -> None:
    modules = dict(model.named_modules())
    with torch.no_grad():
        for name, entry in selected.items():
            module = modules[name]
            if not isinstance(module, torch.nn.Linear):
                raise TypeError(f"{name} is not a Linear module")
            dequantized = store.dequantize(entry, dtype=dtype or module.weight.dtype)
            module.weight.copy_(dequantized.to(module.weight.device, module.weight.dtype))


def refine_fixed_formats(store: CandidateStore, selected: Mapping[str, StoredCandidate], curvature: Mapping[str, CurvatureRecord], gradients: Mapping[str, Tensor], output_store: CandidateStore, *, scale_radius: float = 0.05, code_radius: int = 1, tau: float = 0.02, sweeps: int = 4, device: torch.device | str = "cpu") -> float:
    total_model_delta = 0.0
    device = torch.device(device)
    for name, entry in selected.items():
        payload = store.load(entry)
        op = curvature[name].operator(device=device)
        linear = gradients[name].to(device)
        candidate = refine_quantized_matrix(payload["codes"].to(device), payload["scales"].to(device), op, linear, bits=entry.bits, group_size=entry.group_size, tau=tau, sweeps=sweeps, code_radius=code_radius, scale_radius=scale_radius)
        output_store.add(name, entry.format_name, candidate)
        total_model_delta += candidate.surrogate_loss
    return total_model_delta


def save_selection(result: BootstrapResult, path: str | Path) -> None:
    data = {
        "target_budget_bytes": result.target_budget_bytes,
        "total_cost": result.allocation.total_cost,
        "total_loss": result.allocation.total_loss,
        "lambda": result.allocation.lambda_value,
        "selected": {
            name: {
                "format": entry.format_name,
                "bits": entry.bits,
                "group_size": entry.group_size,
                "serialized_bytes": entry.serialized_bytes,
                "surrogate_loss": entry.surrogate_loss,
                "payload_path": entry.payload_path,
            }
            for name, entry in result.selected.items()
        },
    }
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True))
