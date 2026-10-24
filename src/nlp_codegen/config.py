from pathlib import Path
import math

from omegaconf import OmegaConf


def load_config(path, overrides=()):
    path = Path(path).resolve()
    base = path.parent / "base.yaml"
    parts = [OmegaConf.load(base)] if base != path else []
    parts.extend((OmegaConf.load(path), OmegaConf.from_dotlist(list(overrides))))
    config = OmegaConf.to_container(OmegaConf.merge(*parts), resolve=True)
    return validate_config(config)


def validate_config(config):
    if config["train"]["resume"] and config["train"]["init_checkpoint"]:
        raise ValueError("Use either train.resume or train.init_checkpoint")
    for key in ("batch_size", "microbatch_size", "gradient_accumulation", "max_steps"):
        if int(config["train"][key]) < 1:
            raise ValueError(f"train.{key} must be positive")
    for name in ("max_seconds", "max_total_seconds"):
        value = config["train"].get(name)
        if value is not None and (isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0):
            raise ValueError(f"train.{name} must be a positive finite number")
    if config["train"]["scope"] not in {"all", "decoder"}:
        raise ValueError("train.scope must be all or decoder")
    if config["gradients"]["mode"] not in {"none", "pcgrad", "orthogonal"}:
        raise ValueError("Unknown gradient projection mode")
    for name, value in config["task_sampling"].get("drop_probabilities", {}).items():
        if name not in {"code", "structure", "grounding"} or not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
            raise ValueError(f"Invalid task dropout: {name}")
        if name == "code" and float(value) != 0:
            raise ValueError("Code task dropout must be zero")
    if config["data"]["source"].get("split", "train") != "train":
        raise ValueError("Preparation accepts only the train split")
    mode = config["data"].get("mode", "prepared")
    if mode not in {"prepared", "online"}:
        raise ValueError("data.mode must be prepared or online")
    if mode == "online":
        if not config["data"].get("tokenizer_path"):
            raise ValueError("Online data requires data.tokenizer_path")
        if config["data"].get("sampling_mode", "equal_shards") != "equal_shards":
            raise ValueError("Online data uses equal_shards sampling")
        for name in ("cpu_workers", "prefetch_batches", "shuffle_buffer", "tokenization_batch_size", "validation_scan_per_shard"):
            if int(config["data"]["online"][name]) < 1:
                raise ValueError(f"data.online.{name} must be positive")
    for key in ("log_every", "checkpoint_every"):
        if int(config["train"][key]) < 1:
            raise ValueError(f"train.{key} must be positive")
    if not 0 <= float(config["train"]["ema_decay"]) < 1:
        raise ValueError("train.ema_decay must be in [0, 1)")
    if float(config["train"]["gradient_clip"]) <= 0:
        raise ValueError("train.gradient_clip must be positive")
    for name, weight in config["loss_weights"].items():
        if not math.isfinite(float(weight)) or float(weight) < 0:
            raise ValueError(f"Invalid loss weight: {name}")
    return config
