from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch

from nlp_codegen.model import CodeTransformer


def capture_rng_state(device=None):
    device = torch.device(device) if device is not None else None
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": None,
        "cuda_device": None,
        "mps": None,
    }
    if (device is not None and device.type == "cuda") or (
        device is None and torch.cuda.is_initialized()
    ):
        index = device.index if device is not None else None
        index = torch.cuda.current_device() if index is None else index
        state["cuda_device"] = index
        state["cuda"] = torch.cuda.get_rng_state(index)
    if device is not None and device.type == "mps":
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("Cannot restore CUDA RNG state without CUDA")
        torch.cuda.set_rng_state(state["cuda"].cpu(), device=state["cuda_device"])
    if state.get("mps") is not None:
        if not torch.backends.mps.is_available():
            raise RuntimeError("Cannot restore MPS RNG state without MPS")
        torch.mps.set_rng_state(state["mps"].cpu())


def atomic_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", prefix=f".{path.name}.", suffix=".tmp",
            dir=path.parent, delete=False,
        ) as file:
            temporary_path = Path(file.name)
            torch.save(payload, file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_training_checkpoint(path):
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Training checkpoint must be a mapping")
    if checkpoint.get("format") != "nlp_codegen_training_v1":
        if checkpoint.get("format") == "nlp_codegen_inference_v1" or (
            "format" not in checkpoint and "model_state" in checkpoint and "tokenizer_json" in checkpoint
        ):
            raise ValueError("Cannot resume from an inference export; provide a training checkpoint")
        raise ValueError("Expected training checkpoint format nlp_codegen_training_v1")
    return checkpoint


def _json_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported training configuration value: {type(value).__name__}")


def training_signature(config, corpus_fingerprint, tokenizer_json, world_size):
    if not isinstance(config, Mapping):
        raise ValueError("Training configuration must be a mapping")
    if type(world_size) is not int or world_size < 1:
        raise ValueError("world_size must be a positive integer")
    configuration = dict(config)
    configuration.pop("comet", None)
    if "train" in configuration:
        train = dict(configuration["train"])
        train.pop("resume", None)
        train.pop("output_dir", None)
        train.pop("init_checkpoint", None)
        configuration["train"] = train
    identity = {
        "config": configuration,
        "corpus_fingerprint": corpus_fingerprint,
        "tokenizer_json": tokenizer_json,
        "world_size": world_size,
    }
    encoded = json.dumps(
        identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False, default=_json_value,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def export_checkpoint(
    path, core, model_config, tokenizer_json, metadata, step, weights="live",
):
    if not isinstance(core, CodeTransformer):
        raise TypeError("Inference export requires a CodeTransformer core")
    model_config = json.loads(json.dumps(dict(model_config), default=_json_value, allow_nan=False))
    metadata = json.loads(json.dumps(dict(metadata), default=_json_value, allow_nan=False))
    state = {name: tensor.detach().cpu() for name, tensor in core.state_dict().items()}
    atomic_save(path, {
        "format": "nlp_codegen_inference_v1",
        "model_config": model_config,
        "model_state": state,
        "tokenizer_json": tokenizer_json,
        "metadata": metadata,
        "step": int(step),
        "weights": weights,
    })


def restore_optimizer_device(optimizer, device):
    def move(value):
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        if isinstance(value, list):
            return [move(item) for item in value]
        if isinstance(value, tuple):
            return tuple(move(item) for item in value)
        return value

    for parameter, state in optimizer.state.items():
        optimizer.state[parameter] = move(state)
