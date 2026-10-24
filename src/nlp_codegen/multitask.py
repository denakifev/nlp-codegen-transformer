import math
import random

import torch
from torch import nn
from torch.nn import functional as F

from nlp_codegen.targets import SIZES


class MultitaskModel(nn.Module):
    def __init__(self, core, structure=True, grounding=True):
        super().__init__()
        self.core = core
        width = core.embedding.embedding_dim
        self.structure = nn.Sequential(nn.Linear(width, max(1, width // 2)), nn.GELU(),
                                       nn.Linear(max(1, width // 2), sum(SIZES))) if structure else None
        self.grounding = nn.Linear(width, 1) if grounding else None

    @property
    def active_tasks(self):
        return ("code",) + (("structure",) if self.structure is not None else ()) + (("grounding",) if self.grounding is not None else ())

    def forward(self, source, decoder, tasks=None):
        selected = self.active_tasks if tasks is None else tuple(tasks)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("Select distinct nonempty tasks")
        if not set(selected) <= set(self.active_tasks):
            raise ValueError("Selected task is unknown or its head is disabled")
        core = self.core
        memory, source_mask = core.encode(source)
        hidden = core.embedding_dropout(core.embedding(decoder))
        length = decoder.size(1)
        causal = torch.ones(length, length, device=decoder.device, dtype=torch.bool).tril()
        valid = decoder.ne(core.pad_id)
        self_mask = causal[None, None] & valid[:, None, None, :]
        rotary = core.rotary(length, hidden.device, hidden.dtype)
        for layer in core.decoder:
            hidden = layer(hidden, memory, self_mask, source_mask, rotary)
        hidden = core.decoder_norm(hidden)
        output = {}
        if "code" in selected:
            output["logits"] = F.linear(hidden, core.embedding.weight)
        if "structure" in selected:
            pooled = (hidden * valid[..., None]).sum(1) / valid.sum(1).clamp_min(1)[:, None]
            output["structure_logits"] = self.structure(pooled)
        if "grounding" in selected:
            output["grounding_logits"] = self.grounding(hidden).squeeze(-1)
        return output


def task_loss_sums(outputs, batch):
    target = batch["target"]
    result = {}
    if "logits" in outputs:
        logits = outputs["logits"]
        if logits.shape[:2] != target.shape:
            raise ValueError("Code logits and targets have different shapes")
        result["code"] = (F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), target.reshape(-1),
                                         ignore_index=0, reduction="sum"), int(target.ne(0).sum()))
    if "structure_logits" in outputs:
        logits, labels = outputs["structure_logits"], batch["structure"]
        if labels.shape != (target.size(0), len(SIZES)) or logits.shape != (target.size(0), sum(SIZES)):
            raise ValueError("Invalid structure shapes")
        valid = labels.ne(-100)
        if not bool((valid == valid[:, :1]).all()):
            raise ValueError("A structure row must be fully labeled or fully masked")
        total = logits.sum() * 0
        offset = 0
        for index, size in enumerate(SIZES):
            total = total + F.cross_entropy(logits[:, offset:offset + size].float(), labels[:, index],
                                           ignore_index=-100, reduction="sum") / len(SIZES)
            offset += size
        result["structure"] = total, int(valid[:, 0].sum())
    if "grounding_logits" in outputs:
        logits, labels = outputs["grounding_logits"], batch["grounding"]
        if logits.shape != target.shape or labels.shape != target.shape:
            raise ValueError("Invalid grounding shapes")
        valid = labels.ne(-100)
        if bool((valid & target.eq(0)).any()) or bool((valid & labels.ne(0) & labels.ne(1)).any()):
            raise ValueError("Grounding labels must be binary on non-PAD positions")
        losses = F.binary_cross_entropy_with_logits(logits.float(), labels.clamp_min(0).float(), reduction="none")
        result["grounding"] = (losses * valid).sum(), int(valid.sum())
    return result


class TaskSampler:
    def __init__(self, enabled=("code", "structure", "grounding"), weights=None, seed=42,
                 auxiliary_tasks_per_step=1, always_code=True, drop_probabilities=None):
        allowed = {"code", "structure", "grounding"}
        if isinstance(enabled, dict):
            if weights is not None:
                raise ValueError("Specify weights once")
            weights = enabled
            enabled = tuple(enabled)
        enabled = tuple(enabled)
        if len(set(enabled)) != len(enabled) or not set(enabled) <= allowed:
            raise ValueError("Invalid enabled tasks")
        weights = {} if weights is None else dict(weights)
        if not set(weights) <= allowed:
            raise ValueError("Unknown task weights")
        self.weights = {}
        for name in enabled:
            value = weights.get(name, 1.0)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError("Task weights must be finite and nonnegative")
            if value > 0:
                self.weights[name] = float(value)
        drop_probabilities = {} if drop_probabilities is None else dict(drop_probabilities)
        if not set(drop_probabilities) <= allowed:
            raise ValueError("Unknown task drop probabilities")
        self.drop_probabilities = {}
        for name in ("code", "structure", "grounding"):
            value = drop_probabilities.get(name, 0.0)
            if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Task drop probabilities must be finite and in [0, 1]")
            if name == "code" and value != 0:
                raise ValueError("The code task cannot be dropped")
            self.drop_probabilities[name] = float(value)
        if type(auxiliary_tasks_per_step) is not int or auxiliary_tasks_per_step < 0:
            raise ValueError("Invalid number of auxiliary tasks")
        if always_code and "code" not in self.weights:
            raise ValueError("The main task must be enabled")
        if not always_code and auxiliary_tasks_per_step == 0:
            raise ValueError("At least one task must be sampled")
        if not self.weights:
            raise ValueError("At least one task must be enabled")
        self.enabled = tuple(self.weights)
        self.auxiliary_tasks_per_step = auxiliary_tasks_per_step
        self.always_code = bool(always_code)
        self.rng = random.Random(seed)
        self.steps = 0
        self.sample_counts = dict.fromkeys(self.enabled, 0)
        self.drop_counts = dict.fromkeys(self.enabled, 0)

    def sample(self, available=None):
        if available is None:
            eligible = list(self.enabled)
        else:
            available = set(available)
            if not available <= {"code", "structure", "grounding"}:
                raise ValueError("Unknown available task")
            eligible = [name for name in self.enabled if name in available]
        if not eligible:
            raise ValueError("No valid enabled tasks are available")
        selected = ["code"] if self.always_code and "code" in eligible else []
        candidates = [name for name in eligible if name not in selected]
        for _ in range(min(self.auxiliary_tasks_per_step, len(candidates))):
            choice = self.rng.random() * sum(self.weights[name] for name in candidates)
            accumulated = 0.0
            chosen = candidates[-1]
            for name in candidates:
                accumulated += self.weights[name]
                if choice < accumulated:
                    chosen = name
                    break
            selected.append(chosen)
            candidates.remove(chosen)
        retained = []
        dropped = []
        for name in selected:
            probability = self.drop_probabilities[name]
            if name != "code" and probability > 0 and self.rng.random() < probability:
                dropped.append(name)
            else:
                retained.append(name)
        if not retained and "code" in eligible:
            retained = ["code"]
        if not retained:
            raise ValueError("No valid tasks remain after auxiliary dropping")
        self.steps += 1
        for name in retained:
            self.sample_counts[name] += 1
        for name in dropped:
            self.drop_counts[name] += 1
        return tuple(retained)

    def state_dict(self):
        return {"version": 2, "enabled": self.enabled, "weights": dict(self.weights),
                "auxiliary_tasks_per_step": self.auxiliary_tasks_per_step,
                "always_code": self.always_code, "drop_probabilities": dict(self.drop_probabilities),
                "rng_state": self.rng.getstate(), "steps": self.steps,
                "sample_counts": dict(self.sample_counts), "drop_counts": dict(self.drop_counts)}

    def load_state_dict(self, state):
        if (state.get("version") != 2 or tuple(state.get("enabled", ())) != self.enabled
                or state.get("weights") != self.weights
                or state.get("auxiliary_tasks_per_step") != self.auxiliary_tasks_per_step
                or state.get("always_code") != self.always_code
                or state.get("drop_probabilities") != self.drop_probabilities):
            raise ValueError("Task sampler configuration changed")
        steps = state["steps"]
        if type(steps) is not int or steps < 0:
            raise ValueError("Invalid task sampler step count")
        counters = {}
        for field in ("sample_counts", "drop_counts"):
            values = state[field]
            if (set(values) != set(self.enabled)
                    or any(type(value) is not int or not 0 <= value <= steps for value in values.values())):
                raise ValueError("Invalid task sampler counters")
            counters[field] = dict(values)
        self.rng.setstate(state["rng_state"])
        self.steps = steps
        self.sample_counts = counters["sample_counts"]
        self.drop_counts = counters["drop_counts"]
