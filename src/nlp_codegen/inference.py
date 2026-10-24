import argparse
from collections.abc import Mapping
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch
from tokenizers import Tokenizer

from nlp_codegen.decoding import validate_decoding
from nlp_codegen.generation import generate_codes
from nlp_codegen.model import CodeTransformer


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(path, device="auto", precision="fp32"):
    started = time.perf_counter()
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    device = torch.device(device)
    if precision not in {"fp32", "fp16"}:
        raise ValueError("precision must be fp32 or fp16")
    if precision == "fp16" and device.type == "cpu":
        raise ValueError("CPU inference uses fp32")
    if precision == "fp32":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    dtype = torch.float32 if precision == "fp32" else torch.float16
    synchronize(device)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if "model_config" in checkpoint:
        config = checkpoint["model_config"]
    else:
        spec = checkpoint["model_spec"]
        if not isinstance(spec, Mapping):
            raise ValueError("model_spec must be a mapping")
        if spec.get("_target_") not in {
            "model.CodeTransformer", "src.model.CodeTransformer",
            "nlp_codegen.model.CodeTransformer",
        }:
            raise ValueError("Only CodeTransformer checkpoints are supported")
        config = spec["config"]
    if not isinstance(config, Mapping):
        raise ValueError("model_config must be a mapping")
    config = dict(config)
    model = CodeTransformer(config)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    tokenizer = Tokenizer.from_str(checkpoint["tokenizer_json"])
    tokenizer.encode_special_tokens = True
    if tokenizer.get_vocab_size() != int(config["vocab_size"]):
        raise ValueError("Tokenizer vocabulary does not match model_config")
    metadata = dict(checkpoint.get("metadata", {}))
    for name in ("pad_id", "bos_id", "eos_id"):
        if int(metadata.get(name, config[name])) != int(config[name]):
            raise ValueError(f"Checkpoint metadata does not match {name}")
        metadata[name] = int(config[name])
    architecture = {
        name: int(config[name])
        for name in ("d_model", "d_ff", "num_heads", "vocab_size")
    }
    architecture.update(
        encoder_layers=len(model.encoder),
        decoder_layers=len(model.decoder),
        num_kv_heads=int(config.get("num_kv_heads", config["num_heads"])),
        norm_type=config.get("norm_type", "rmsnorm"),
        norm_eps=float(config.get("norm_eps", 1e-6)),
        parameters=sum(parameter.numel() for parameter in model.parameters()),
    )
    context = {
        "model": model,
        "tokenizer": tokenizer,
        "metadata": metadata,
        "checkpoint_step": checkpoint.get("step", checkpoint.get("checkpoint_step", 0)),
        "weights": checkpoint.get("weights", "live"),
        "device": device,
        "precision": precision,
        "architecture": architecture,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    del checkpoint
    model.to(device=device, dtype=dtype).eval()
    context["checkpoint_sha256"] = file_sha256(path)
    synchronize(device)
    context["load_seconds"] = time.perf_counter() - started
    return context


def load_questions(path):
    questions = []
    with Path(path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("question"), str):
                raise ValueError(f"Line {line_number} must contain a string question")
            questions.append(row["question"])
    return questions


def write_metrics(path, metrics):
    with Path(path).open("w", encoding="utf-8") as file:
        file.write(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
        file.flush()
        os.fsync(file.fileno())
    with Path(path).open(encoding="utf-8") as file:
        if json.load(file) != metrics:
            raise RuntimeError("Metrics readback differs from written values")


def run_inference(
    checkpoint_path,
    test_path,
    output_path,
    device="cuda:0",
    precision="fp32",
    batch_size=32,
    num_beams=5,
    length_penalty=1.0,
    max_new_tokens=512,
    started_at=None,
):
    entered_at = time.perf_counter()
    started = entered_at if started_at is None else float(started_at)
    if not math.isfinite(started) or started > entered_at:
        raise ValueError("started_at must be an earlier time.perf_counter() value")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    validate_decoding("beam", 0.7, 0.9, num_beams, length_penalty)
    checkpoint_path, test_path, output_path = (
        Path(path) for path in (checkpoint_path, test_path, output_path)
    )
    metrics_path = output_path.with_suffix(".metrics.json")
    if {output_path.resolve(), metrics_path.resolve()} & {
        checkpoint_path.resolve(), test_path.resolve()
    }:
        raise ValueError("Output paths must differ from checkpoint and test paths")
    test_started = time.perf_counter()
    questions = load_questions(test_path)
    test_load_seconds = time.perf_counter() - test_started
    context = load_checkpoint(checkpoint_path, device=device, precision=precision)
    tokenizer, metadata = context["tokenizer"], context["metadata"]
    pad_id, bos_id, eos_id = (
        int(metadata[name]) for name in ("pad_id", "bos_id", "eos_id")
    )
    tokenize_started = time.perf_counter()
    encodings = tokenizer.encode_batch(questions, add_special_tokens=False)
    sources = [encoding.ids + [eos_id] for encoding in encodings]
    tokenization_seconds = time.perf_counter() - tokenize_started
    predictions, generation = generate_codes(
        context["model"], sources, tokenizer, context["device"],
        pad_id, bos_id, eos_id, batch_size=batch_size,
        max_new_tokens=max_new_tokens, decoding="beam", num_beams=num_beams,
        length_penalty=length_penalty,
    )
    write_started = time.perf_counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written_digest = hashlib.sha256()
    with output_path.open("wb") as file:
        for question, code in zip(questions, predictions, strict=True):
            line = (json.dumps({"question": question, "code": code}, ensure_ascii=False) + "\n").encode("utf-8")
            file.write(line)
            written_digest.update(line)
        file.flush()
        os.fsync(file.fileno())
    output_write_seconds = time.perf_counter() - write_started
    readback_started = time.perf_counter()
    readback_digest = hashlib.sha256()
    output_rows = 0
    with output_path.open("rb") as file:
        for line in file:
            readback_digest.update(line)
            row = json.loads(line)
            if output_rows >= len(questions) or row != {
                "question": questions[output_rows], "code": predictions[output_rows]
            }:
                raise RuntimeError("Prediction readback differs from generated rows")
            output_rows += 1
    if output_rows != len(questions) or readback_digest.digest() != written_digest.digest():
        raise RuntimeError("Prediction count or SHA-256 readback failed")
    output_readback_seconds = time.perf_counter() - readback_started
    synchronize(context["device"])
    metrics = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": context["checkpoint_sha256"],
        "checkpoint_step": context["checkpoint_step"],
        "weights": context["weights"],
        "test_path": str(test_path),
        "output_path": str(output_path),
        "output_sha256": readback_digest.hexdigest(),
        "output_rows": output_rows,
        "output_bytes": output_path.stat().st_size,
        "examples": len(questions),
        "device": str(context["device"]),
        "precision": context["precision"],
        "architecture": context["architecture"],
        "gpu": context["gpu"],
        "is_t4": bool(context["gpu"] and "T4" in context["gpu"]),
        "batch_size": batch_size,
        "decoding": "beam",
        "num_beams": num_beams,
        "length_penalty": length_penalty,
        "max_new_tokens": max_new_tokens,
        "max_source_tokens": max((len(source) for source in sources), default=0),
        "test_load_seconds": test_load_seconds,
        "load_seconds": context["load_seconds"],
        "tokenization_seconds": tokenization_seconds,
        "output_write_seconds": output_write_seconds,
        "output_readback_seconds": output_readback_seconds,
        "external_started_at": started_at is not None,
        **generation,
    }
    metrics["total_seconds"] = time.perf_counter() - started
    metrics_started = time.perf_counter()
    write_metrics(metrics_path, metrics)
    metrics["initial_metrics_write_seconds"] = time.perf_counter() - metrics_started
    metrics["total_seconds"] = time.perf_counter() - started
    metrics["timing_end"] = "after prediction fsync/readback/hash and initial metrics fsync/readback"
    write_metrics(metrics_path, metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    metrics = run_inference(
        args.checkpoint, args.test, args.output, device=args.device,
        precision=args.precision, batch_size=args.batch_size,
        num_beams=args.num_beams, length_penalty=args.length_penalty,
        max_new_tokens=args.max_new_tokens,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
