import argparse
from contextlib import nullcontext
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from nlp_codegen.checkpoint import (
    atomic_save, capture_rng_state, export_checkpoint, load_training_checkpoint,
    restore_optimizer_device, restore_rng_state, training_signature,
)
from nlp_codegen.config import load_config
from nlp_codegen.data import PreparedCorpus, ShardBatchSampler
from nlp_codegen.generation import generate_codes
from nlp_codegen.gradients import project_gradients
from nlp_codegen.model import CodeTransformer
from nlp_codegen.multitask import MultitaskModel, TaskSampler, task_loss_sums
from nlp_codegen.online import OnlineBatchSource, OnlineCorpus
from nlp_codegen.targets import SIZES, derive_auxiliary_labels
from nlp_codegen.tracking import CometLogger


class TrainingDataset(Dataset):
    def __init__(self, dataset, tokenizer, auxiliary):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.auxiliary = auxiliary

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row = dict(self.dataset[index])
        if self.auxiliary:
            structure, grounding = derive_auxiliary_labels(
                row["source_ids"], row["target_ids"], self.tokenizer,
            )
        else:
            structure = [-100] * len(SIZES)
            grounding = [-100.0] * len(row["target_ids"])
        row.update(structure=structure, grounding=grounding)
        return row


def collate_records(records):
    sources = [torch.tensor(row["source_ids"], dtype=torch.long) for row in records]
    targets = [torch.tensor(row["target_ids"], dtype=torch.long) for row in records]
    decoders = [torch.cat((torch.tensor([1]), target[:-1])) for target in targets]
    grounding = [torch.tensor(row["grounding"], dtype=torch.float32) for row in records]
    return {
        "source": pad_sequence(sources, batch_first=True, padding_value=0),
        "decoder": pad_sequence(decoders, batch_first=True, padding_value=0),
        "target": pad_sequence(targets, batch_first=True, padding_value=0),
        "structure": torch.tensor([row["structure"] for row in records], dtype=torch.long),
        "grounding": pad_sequence(grounding, batch_first=True, padding_value=-100.0),
        "shard_id": torch.tensor([row["shard_id"] for row in records], dtype=torch.long),
    }


def choose_device(name, local_rank):
    if name == "auto":
        if torch.cuda.is_available():
            name = f"cuda:{local_rank}"
        elif int(os.environ.get("WORLD_SIZE", "1")) > 1:
            name = "cpu"
        else:
            name = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "mps" and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError("Distributed training requires CUDA or CPU")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device


def sum_across_ranks(value):
    if dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def runtime_fingerprint():
    directory = Path(__file__).parent
    names = ("trainer", "model", "decoding", "generation", "multitask", "targets",
             "gradients", "data", "online", "prepare", "checkpoint", "config")
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode())
        digest.update((directory / f"{name}.py").read_bytes())
    return digest.hexdigest()


def set_trainable_scope(model, scope):
    for name, parameter in model.core.named_parameters():
        parameter.requires_grad_(scope == "all" or name.startswith(("decoder.", "decoder_norm.")))
    if not any(parameter.requires_grad for parameter in model.core.parameters()):
        raise ValueError("No trainable model parameters")


def learning_rate_factor(step, warmup, total, minimum, time_progress=0):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = min(1.0, max((step - warmup) / max(1, total - warmup), time_progress))
    return minimum + (1 - minimum) * 0.5 * (1 + math.cos(math.pi * progress))


def flatten_gradients(gradients, parameters):
    return torch.cat([
        (gradient.detach().float() if gradient is not None else torch.zeros_like(parameter)).reshape(-1)
        for gradient, parameter in zip(gradients, parameters)
    ])


def assign_gradients(parameters, vector, scale):
    offset = 0
    for parameter in parameters:
        length = parameter.numel()
        parameter.grad = vector[offset:offset + length].view_as(parameter).mul(scale)
        offset += length


def batch_counts(batches, device):
    values = [
        sum(int(batch["target"].ne(0).sum()) for batch in batches),
        sum(int(batch["structure"][:, 0].ne(-100).sum()) for batch in batches),
        sum(int(batch["grounding"].ne(-100).sum()) for batch in batches),
    ]
    dtype = torch.float32 if device.type == "mps" else torch.float64
    counts = sum_across_ranks(torch.tensor(values, device=device, dtype=dtype))
    return dict(zip(("code", "structure", "grounding"), counts.tolist()))


def update_step(model, batches, optimizer, scaler, selected, denominators, config, projection_rng, device):
    shared = [parameter for parameter in model.core.parameters() if parameter.requires_grad]
    private = {
        task: list(getattr(model, task).parameters()) if task != "code" else []
        for task in selected
    }
    shared_grads = {task: None for task in selected}
    private_grads = {task: None for task in selected}
    numerators = dict.fromkeys(selected, 0.0)
    optimizer.zero_grad(set_to_none=True)
    scale = scaler.get_scale()
    precision = config["train"]["precision"]
    autocast_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    microbatch = int(config["train"]["microbatch_size"])
    for batch in batches:
        for start in range(0, len(batch["source"]), microbatch):
            part = {name: value[start:start + microbatch].to(device, non_blocking=device.type == "cuda")
                    for name, value in batch.items()}
            context = torch.autocast(device.type, dtype=autocast_dtype) if precision != "fp32" else nullcontext()
            with context:
                outputs = model(part["source"], part["decoder"], tasks=selected)
                losses = task_loss_sums(outputs, part)
            for position, task in enumerate(selected):
                numerator, _ = losses[task]
                numerators[task] += float(numerator.detach())
                weight = float(config["loss_weights"][task])
                loss = numerator * (weight / denominators[task])
                parameters = shared + private[task]
                gradients = torch.autograd.grad(
                    scaler.scale(loss), parameters,
                    retain_graph=position + 1 < len(selected), allow_unused=True,
                )
                core_gradient = flatten_gradients(gradients[:len(shared)], shared).div_(scale)
                if shared_grads[task] is None:
                    shared_grads[task] = core_gradient
                else:
                    shared_grads[task].add_(core_gradient)
                if private[task]:
                    head_gradient = flatten_gradients(gradients[len(shared):], private[task]).div_(scale)
                    if private_grads[task] is None:
                        private_grads[task] = head_gradient
                    else:
                        private_grads[task].add_(head_gradient)
    vectors = [sum_across_ranks(shared_grads[task]) for task in selected]
    for task in selected:
        if private_grads[task] is not None:
            sum_across_ranks(private_grads[task])
    finite = all(bool(torch.isfinite(vector).all()) for vector in vectors)
    if finite:
        projected, projection_metrics = project_gradients(
            vectors, mode=config["gradients"]["mode"], generator=projection_rng,
        )
    else:
        projected, projection_metrics = vectors, {"nonfinite_task_gradients": 1}
    assign_gradients(shared, torch.stack(projected).sum(0), scale)
    for task in selected:
        if private_grads[task] is not None:
            assign_gradients(private[task], private_grads[task], scale)
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        float(config["train"]["gradient_clip"]),
        error_if_nonfinite=not scaler.is_enabled(),
    )
    previous_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    skipped = scaler.get_scale() < previous_scale
    dtype = torch.float32 if device.type == "mps" else torch.float64
    total_numerators = sum_across_ranks(torch.tensor(
        [numerators[task] for task in selected], device=device, dtype=dtype,
    ))
    metrics = {}
    for task, value in zip(selected, total_numerators):
        value = float(value) / denominators[task]
        if math.isfinite(value):
            metrics[f"loss/{task}"] = value
        else:
            metrics[f"loss/{task}_nonfinite"] = 1
    metrics.update({f"gradients/{name}": float(value) for name, value in projection_metrics.items()
                    if isinstance(value, (int, float))})
    if bool(torch.isfinite(norm)):
        metrics["gradient_norm"] = float(norm)
    else:
        metrics["gradient_norm_nonfinite"] = 1
    metrics.update(amp_scale=scaler.get_scale(), amp_step_skipped=int(skipped))
    return metrics, skipped


@torch.inference_mode()
def evaluate(core, corpus, config, device, rank, world_size, ema=None):
    count = min(len(corpus.validation), int(config["evaluation"]["max_examples"]))
    if count == 0:
        return {}
    indices = np.random.default_rng(int(config["seed"]) + 1100).permutation(len(corpus.validation))[:count].tolist()
    backup = None
    if ema is not None:
        backup = {name: value.detach().clone() for name, value in core.state_dict().items()}
        core.load_state_dict(ema, strict=True)
    core.eval()
    dtype = torch.float32 if device.type == "mps" else torch.float64
    totals = torch.zeros(4, device=device, dtype=dtype)
    batch_size = int(config["evaluation"]["batch_size"])
    local = indices[rank::world_size]
    for start in range(0, len(local), batch_size):
        records = [corpus.validation[index] for index in local[start:start + batch_size]]
        records = [dict(row, structure=[-100] * len(SIZES), grounding=[-100.] * len(row["target_ids"]))
                   for row in records]
        batch = {name: value.to(device) for name, value in collate_records(records).items()}
        logits = core(batch["source"], batch["decoder"])
        totals[0] += F.cross_entropy(logits.flatten(0, 1).float(), batch["target"].flatten(),
                                     ignore_index=0, reduction="sum")
        valid = batch["target"].ne(0)
        totals[1] += valid.sum()
        totals[2] += (logits.argmax(-1).eq(batch["target"]) & valid).sum()
    generation_count = min(count, int(config["evaluation"]["generation_examples"]))
    generated_indices = indices[:generation_count][rank::world_size]
    if generated_indices:
        records = [corpus.validation[index] for index in generated_indices]
        codes, _ = generate_codes(
            core, [row["source_ids"] for row in records], corpus.tokenizer, device, 0, 1, 2,
            batch_size=batch_size, decoding="beam", num_beams=int(config["evaluation"]["num_beams"]),
            max_new_tokens=int(config["evaluation"]["max_new_tokens"]),
            length_penalty=float(config["evaluation"]["length_penalty"]),
        )
        references = [corpus.tokenizer.decode(row["target_ids"][:-1], skip_special_tokens=True) for row in records]
        totals[3] = sum(code == reference for code, reference in zip(codes, references))
    sum_across_ranks(totals)
    result = {"validation/loss": float(totals[0] / totals[1].clamp_min(1)),
              "validation/token_accuracy": float(totals[2] / totals[1].clamp_min(1)),
              "validation/examples": count}
    if generation_count:
        result["validation/sequence_exact_match_at_1"] = float(totals[3]) / generation_count
    if backup is not None:
        core.load_state_dict(backup, strict=True)
    return result


def run_training(config, max_updates=None):
    run_started = time.monotonic()
    config = copy.deepcopy(config)
    train = config["train"]
    rank, world_size = int(os.getenv("RANK", "0")), int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    device = choose_device(train["device"], local_rank)
    initialized_here = world_size > 1 and not dist.is_initialized()
    if initialized_here:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    tracker = None
    online_source = None
    old_handler = None
    stop_requested = [False]
    try:
        if train["precision"] not in {"fp32", "fp16", "bf16"}:
            raise ValueError("train.precision must be fp32, fp16 or bf16")
        if train["precision"] == "fp16" and device.type != "cuda":
            raise ValueError("FP16 training requires CUDA; use train.precision=fp32 on CPU/MPS")
        if train["precision"] == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise ValueError("This GPU does not support BF16")
        online = config["data"].get("mode", "prepared") == "online"
        if online:
            validation_examples = int(config["evaluation"]["max_examples"]) if int(config["evaluation"]["every"]) > 0 else 0
            corpus = OnlineCorpus(config["data"], validation_examples=validation_examples)
        else:
            corpus = PreparedCorpus(config["data"]["output_dir"])
        if not online and len(corpus.train) == 0:
            raise ValueError("Training split is empty")
        signature = training_signature(config, corpus.fingerprint, corpus.tokenizer_json, world_size)
        runtime = runtime_fingerprint()
        resumed = load_training_checkpoint(train["resume"]) if train["resume"] else None
        if resumed and resumed["signature"] != signature:
            raise ValueError("Resume config, data, tokenizer or world size changed")
        if resumed and resumed.get("runtime_fingerprint") != runtime:
            raise ValueError("Resume training implementation changed")
        previous_elapsed = float(resumed.get("elapsed_seconds", 0.0)) if resumed else 0.0
        total_limit = train.get("max_total_seconds")
        total_limit = float(total_limit) if total_limit is not None else None

        def elapsed_total(synchronize=False):
            value = previous_elapsed + time.monotonic() - run_started
            if synchronize and dist.is_initialized():
                dtype = torch.float32 if device.type == "mps" else torch.float64
                elapsed = torch.tensor(value, device=device, dtype=dtype)
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                value = float(elapsed)
            return value

        output = Path(train["output_dir"])
        if not resumed and (output / "last_training.pt").exists():
            raise ValueError("Output contains a checkpoint; use train.resume or another train.output_dir")
        output.mkdir(parents=True, exist_ok=True)
        seed = int(config["seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model_config = dict(config["model"], vocab_size=int(corpus.metadata["vocab_size"]), pad_id=0, bos_id=1, eos_id=2)
        core = CodeTransformer(model_config)
        if train["init_checkpoint"]:
            initial = torch.load(train["init_checkpoint"], map_location="cpu", weights_only=True)
            if json.loads(initial["tokenizer_json"]) != json.loads(corpus.tokenizer_json):
                raise ValueError("Initial checkpoint and prepared tokenizer differ")
            core.load_state_dict(initial["model_state"], strict=True)
        model = MultitaskModel(core, **config["heads"]).to(device)
        set_trainable_scope(model, train["scope"])
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=float(config["optimizer"]["lr"]), betas=tuple(config["optimizer"]["betas"]),
            eps=float(config["optimizer"]["eps"]), weight_decay=float(config["optimizer"]["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: learning_rate_factor(
                step, int(config["scheduler"]["warmup_steps"]), int(train["max_steps"]),
                float(config["scheduler"]["min_lr_ratio"]),
                time_progress=elapsed_total(synchronize=True) / total_limit if total_limit is not None else 0,
            ),
        )
        scaler = torch.amp.GradScaler("cuda", enabled=train["precision"] == "fp16")
        decay = float(train["ema_decay"])
        ema = {name: value.detach().clone() for name, value in core.state_dict().items()} if decay else None
        task_sampler = TaskSampler(
            enabled=model.active_tasks, weights=config["task_sampling"]["weights"], seed=seed + 400,
            always_code=bool(config["task_sampling"]["always_code"]),
            auxiliary_tasks_per_step=int(config["task_sampling"]["auxiliary_tasks_per_step"]),
            drop_probabilities=config["task_sampling"].get("drop_probabilities"),
        )
        projection_rng = torch.Generator().manual_seed(seed + 900)
        rank_state = resumed["rank_states"][rank] if resumed else None
        sampler_state = rank_state["sampler"] if rank_state else None
        if online:
            online_source = OnlineBatchSource(
                corpus, int(train["batch_size"]), rank, world_size, seed,
                any(config["heads"].values()), collate_records,
                pin_memory=bool(train["pin_memory"]) and device.type == "cuda", state=sampler_state,
            )
            sampler = online_source
            iterator = iter(online_source)
        else:
            sampler = ShardBatchSampler(
                corpus.train, int(train["batch_size"]), rank=rank, world_size=world_size, seed=seed,
                mode=config["data"]["sampling_mode"], state=sampler_state,
            )
            dataset = TrainingDataset(corpus.train, corpus.tokenizer, any(config["heads"].values()))
            loader_options = dict(
                batch_sampler=sampler, collate_fn=collate_records,
                num_workers=int(train["num_workers"]), pin_memory=bool(train["pin_memory"]) and device.type == "cuda",
                generator=torch.Generator().manual_seed(seed + 500 + rank),
            )
            if int(train["num_workers"]) > 0:
                loader_options.update(prefetch_factor=int(train["prefetch_factor"]), persistent_workers=True)
            loader = DataLoader(dataset, **loader_options)
            iterator = iter(loader)
        tracker = CometLogger(config["comet"], experiment_key=resumed.get("experiment_key") if resumed else None, rank=rank)
        tracker.log_parameters(config)
        experiment_key = tracker.key or (resumed.get("experiment_key") if resumed else None)
        experiment_url = tracker.url or (resumed.get("experiment_url") if resumed else None)
        step = int(resumed["step"]) if resumed else 0
        optimizer_steps = int(resumed["optimizer_steps"]) if resumed else 0
        best_loss = float(resumed["best_loss"]) if resumed else float("inf")
        shard_count = corpus.shard_count if online else len(corpus.train.shard_counts)
        shard_seen = list(rank_state["shard_seen"]) if rank_state else [0] * shard_count
        if resumed:
            model.load_state_dict(resumed["model_state"], strict=True)
            optimizer.load_state_dict(resumed["optimizer"])
            restore_optimizer_device(optimizer, device)
            scheduler.load_state_dict(resumed["scheduler"])
            scaler.load_state_dict(resumed["scaler"])
            ema = {name: value.to(device) for name, value in resumed["ema"].items()} if resumed["ema"] is not None else None
            task_sampler.load_state_dict(rank_state["tasks"])
            projection_rng.set_state(rank_state["projection_rng"])
            restore_rng_state(rank_state["rng"])
        else:
            random.seed(seed + rank)
            np.random.seed(seed + rank)
            torch.manual_seed(seed + rank)
        if rank == 0:
            (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
            (output / "run.json").write_text(json.dumps(
                {"world_size": world_size, "signature": signature, "data_fingerprint": corpus.fingerprint,
                 "runtime_fingerprint": runtime, "experiment_key": experiment_key,
                 "experiment_url": experiment_url, "model_config": model_config},
                indent=2,
            ) + "\n")
        session_started, session_steps = time.monotonic(), 0
        if hasattr(signal, "SIGTERM"):
            old_handler = signal.signal(signal.SIGTERM, lambda *_: stop_requested.__setitem__(0, True))

        checkpoint_elapsed = None

        def save_state():
            nonlocal checkpoint_elapsed
            state = {"sampler": sampler.state_dict(), "tasks": task_sampler.state_dict(),
                     "projection_rng": projection_rng.get_state(), "rng": capture_rng_state(device),
                     "shard_seen": shard_seen}
            states = [None] * world_size
            if dist.is_initialized():
                dist.all_gather_object(states, state)
            else:
                states[0] = state
            checkpoint_elapsed = elapsed_total(synchronize=total_limit is not None)
            if rank == 0:
                payload = {
                    "format": "nlp_codegen_training_v1", "step": step, "optimizer_steps": optimizer_steps,
                    "model_state": model.state_dict(), "model_config": model_config,
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "ema": ema, "best_loss": best_loss, "signature": signature,
                    "runtime_fingerprint": runtime, "config": config,
                    "rank_states": states, "tokenizer_json": corpus.tokenizer_json, "metadata": corpus.metadata,
                    "experiment_key": experiment_key, "experiment_url": experiment_url,
                    "elapsed_seconds": checkpoint_elapsed,
                }
                atomic_save(output / "last_training.pt", payload)
                backup = None
                if ema is not None and config["evaluation"]["use_ema"]:
                    backup = {name: value.detach().clone() for name, value in core.state_dict().items()}
                    core.load_state_dict(ema, strict=True)
                export_checkpoint(output / "model.pt", core, model_config, corpus.tokenizer_json,
                                  corpus.metadata, step, weights="ema" if backup is not None else "live")
                if backup is not None:
                    core.load_state_dict(backup, strict=True)
                if config["comet"]["log_checkpoints"]:
                    tracker.log_asset(output / "model.pt")

        last_metrics = {}
        budget_stopped = False
        while step < int(train["max_steps"]):
            if total_limit is not None:
                elapsed_before = elapsed_total(synchronize=True)
                if elapsed_before >= total_limit and (resumed is not None or step > 0):
                    budget_stopped = True
                    break
            model.train()
            batches = [next(iterator) for _ in range(int(train["gradient_accumulation"]))]
            counts = batch_counts(batches, device)
            available = tuple(task for task in model.active_tasks if counts[task] > 0)
            selected = task_sampler.sample(available=available)
            if dist.is_initialized():
                shared_choice = [selected]
                dist.broadcast_object_list(shared_choice, src=0)
                if selected != shared_choice[0]:
                    raise RuntimeError("Task sampling diverged between ranks")
            if not selected:
                raise ValueError("Selected tasks have no valid targets")
            metrics, skipped = update_step(
                model, batches, optimizer, scaler, selected, counts, config, projection_rng, device,
            )
            if not skipped:
                scheduler.step()
                optimizer_steps += 1
                if ema is not None:
                    with torch.no_grad():
                        for name, value in core.state_dict().items():
                            ema[name].lerp_(value.detach(), 1 - decay)
            sampler.commit(len(batches))
            for batch in batches:
                for index, count in zip(*torch.unique(batch["shard_id"], return_counts=True)):
                    shard_seen[int(index)] += int(count)
            step += 1
            session_steps += 1
            metrics.update(step=step, optimizer_steps=optimizer_steps, learning_rate=optimizer.param_groups[0]["lr"])
            metrics.update({f"tasks/{task}_sampled": int(task in selected) for task in model.active_tasks})
            metrics.update({f"tasks/{task}_sampled_total": value for task, value in task_sampler.sample_counts.items()})
            metrics.update({f"tasks/{task}_dropped_total": value for task, value in task_sampler.drop_counts.items()})
            if int(config["evaluation"]["every"]) > 0 and step % int(config["evaluation"]["every"]) == 0:
                validation = evaluate(
                    core, corpus, config, device, rank, world_size,
                    ema if config["evaluation"]["use_ema"] else None,
                )
                metrics.update(validation)
                current_loss = validation.get("validation/loss", float("inf"))
                if current_loss < best_loss:
                    best_loss = current_loss
                    if rank == 0:
                        backup = {name: value.detach().clone() for name, value in core.state_dict().items()}
                        if ema is not None and config["evaluation"]["use_ema"]:
                            core.load_state_dict(ema, strict=True)
                        export_checkpoint(output / "best.pt", core, model_config, corpus.tokenizer_json,
                                          corpus.metadata, step, weights="ema" if ema is not None and config["evaluation"]["use_ema"] else "live")
                        core.load_state_dict(backup, strict=True)
            if step % int(train["log_every"]) == 0 or session_steps == 1:
                seen = sum_across_ranks(torch.tensor(shard_seen, device=device, dtype=torch.long)).tolist()
                metrics.update({f"data/shard_{index}_seen": value for index, value in enumerate(seen)})
                metrics["data/total_examples"] = sum(seen)
                metrics["session/seconds"] = time.monotonic() - session_started
                if rank == 0:
                    tracker.log_metrics(metrics, step=step)
                    with (output / "metrics.jsonl").open("a") as file:
                        file.write(json.dumps(metrics, allow_nan=False) + "\n")
            last_metrics = metrics
            wall_limit = train["max_seconds"]
            stop = stop_requested[0] or (
                wall_limit is not None and time.monotonic() - session_started >= float(wall_limit)
            ) or (max_updates is not None and session_steps >= int(max_updates)) or (
                total_limit is not None and elapsed_total() >= total_limit
            )
            flag = torch.tensor(int(stop), device=device)
            if dist.is_initialized():
                dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            if step % int(train["checkpoint_every"]) == 0 or step == int(train["max_steps"]) or bool(flag):
                save_state()
            if bool(flag):
                break
        if session_steps == 0 or budget_stopped:
            save_state()
        elapsed_seconds = float(checkpoint_elapsed)
        completion_reason = "steps" if step >= int(train["max_steps"]) else (
            "time_budget" if total_limit is not None and elapsed_seconds >= total_limit else None
        )
        result = {"step": step, "optimizer_steps": optimizer_steps, "rank": rank, "world_size": world_size,
                  "checkpoint": str(output / "last_training.pt"), "model": str(output / "model.pt"),
                  "data_fingerprint": corpus.fingerprint, "experiment_key": experiment_key,
                  "experiment_url": experiment_url,
                  "elapsed_seconds": elapsed_seconds, "complete": completion_reason is not None,
                  "completion_reason": completion_reason,
                  "metrics": last_metrics}
        if rank == 0:
            (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    finally:
        if online_source is not None:
            online_source.close()
        if old_handler is not None:
            signal.signal(signal.SIGTERM, old_handler)
        if tracker is not None:
            tracker.close()
        if initialized_here and dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mtl.yaml")
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    from dotenv import load_dotenv
    load_dotenv(Path.cwd() / ".env", override=False)
    result = run_training(load_config(args.config, args.overrides), max_updates=args.max_updates)
    if result["rank"] == 0:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
