import copy
import json
from pathlib import Path

from nlp_codegen.config import load_config
from nlp_codegen.prepare import prepare_data


def prepare_tiny_corpus(root, name="corpus", shard_sizes=(12, 20, 28)):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for shard, size in enumerate(shard_sizes):
        path = root / f"{name}.train.{shard}.jsonl"
        rows = []
        for index in range(size):
            value = index + 2
            if index % 3 == 0:
                question = f"Shard {shard}, task {index}: add {value} and 3."
                code = f"def solve():\n    return {value} + 3\n"
            elif index % 3 == 1:
                question = f"Shard {shard}, task {index}: multiply {value} by 2, then add 4."
                code = f"def solve():\n    value = {value}\n    result = value * 2\n    return result + 4\n"
            else:
                question = f"Shard {shard}, task {index}: divide the sum of {value} and 5 by 2, then subtract 1."
                code = f"def solve():\n    left = {value}\n    right = 5\n    total = left + right\n    result = total / 2\n    return result - 1\n"
            rows.append({"question": question, "code": code})
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        paths.append(str(path))
    data = {
        "seed": 42,
        "output_dir": str(root / name),
        "source": {"kind": "local_jsonl", "split": "train", "input_paths": paths},
        "validation_fraction": 0.0,
        "vocab_size": 384,
        "tokenizer_sample_size": 36,
        "tokenizer_min_frequency": 1,
        "max_source_length": 256,
        "max_target_length": 256,
        "normalize_code": False,
        "preparation_batch_size": 7,
        "sampling_mode": "equal_shards",
    }
    prepare_data(data)
    return data


def tiny_training_config(root, data, name="run", dropout=0.0, max_steps=4):
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "mtl.yaml")
    config["data"] = copy.deepcopy(data)
    config["model"].update(
        d_model=16, d_ff=32, encoder_layers=1, decoder_layers=1,
        num_heads=2, num_kv_heads=2, dropout=dropout,
    )
    config["train"].update(
        output_dir=str(Path(root) / name), device="cpu", precision="fp32",
        max_steps=max_steps, batch_size=4, microbatch_size=2,
        gradient_accumulation=1, num_workers=0, pin_memory=False,
        checkpoint_every=2, log_every=1, ema_decay=0.9,
    )
    config["evaluation"]["every"] = 0
    config["scheduler"]["warmup_steps"] = 1
    return config


def training_fixture(root):
    return tiny_training_config(root, prepare_tiny_corpus(root))
