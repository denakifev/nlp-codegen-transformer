import argparse
import ast
import hashlib
import json
import os
import random
import re
import shutil
import tempfile
import unicodedata
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from .data import FORMAT_VERSION, SPECIAL_TOKENS, PreparedCorpus, canonical_json, file_sha256, fingerprint, validate_tokenizer


DATASET_NAME = "SlavaYus/tinysms_qa_12m"
DATASET_REVISION = "b2724689ff079e9c22efd548fe134bfb29036dbb"


def normalized_question(question):
    return " ".join(unicodedata.normalize("NFKC", question).split())


def is_validation(question, fraction, seed):
    value = f"{seed}\0{normalized_question(question)}".encode("utf-8")
    digest = hashlib.blake2b(value, digest_size=8).digest()
    return int.from_bytes(digest, "big") < int(fraction * (1 << 64))


def normalize_code(code):
    try:
        tree = ast.parse(code, type_comments=True)
        protected = {"inspect", "getsource", "getsourcelines", "currentframe", "getframeinfo", "eval", "exec", "compile", "globals", "locals", "__file__", "__code__", "__traceback__"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in protected:
                return code
            if isinstance(node, ast.Attribute) and node.attr in protected:
                return code
        result = ast.unparse(tree) + "\n"
        if ast.dump(ast.parse(result, type_comments=True), include_attributes=False) != ast.dump(tree, include_attributes=False):
            return code
        return result
    except (SyntaxError, ValueError, TypeError, RecursionError):
        return code


def _settings(config):
    data = dict(config.get("data", config))
    seed = int(data.get("seed", config.get("seed", 42)))
    settings = {
        "seed": seed,
        "validation_fraction": float(data.get("validation_fraction", 0.01)),
        "vocab_size": int(data.get("vocab_size", 16384)),
        "tokenizer_sample_size": int(data.get("tokenizer_sample_size", 120000)),
        "tokenizer_min_frequency": int(data.get("tokenizer_min_frequency", 2)),
        "max_source_length": int(data.get("max_source_length", 512)),
        "max_target_length": int(data.get("max_target_length", 512)),
        "normalize_code": data.get("normalize_code", False),
        "preparation_batch_size": int(data.get("preparation_batch_size", 1024)),
        "question_normalization": "unicode-nfkc-whitespace-v1",
        "split_algorithm": "seeded-blake2b64-threshold-v1",
        "format_version": FORMAT_VERSION,
    }
    if seed < 0 or not 0 <= settings["validation_fraction"] < 1:
        raise ValueError("The split seed or validation fraction is invalid")
    if type(settings["normalize_code"]) is not bool:
        raise ValueError("normalize_code must be a boolean")
    for key in ("tokenizer_sample_size", "tokenizer_min_frequency", "max_source_length", "max_target_length", "preparation_batch_size"):
        if settings[key] < 1:
            raise ValueError(f"{key} must be positive")
    if settings["vocab_size"] < 259:
        raise ValueError("The byte-level vocabulary must contain at least 259 tokens")
    return data, settings


def _source(data):
    source = dict(data.get("source", {}))
    if source.get("split", data.get("split", "train")) != "train":
        raise ValueError("Only the source train split is allowed")
    kind = source.get("kind", data.get("kind", "local_jsonl" if data.get("input_paths") else "huggingface"))
    if kind == "local_jsonl":
        paths = source.get("input_paths", data.get("input_paths"))
        if not isinstance(paths, (list, tuple)) or not paths:
            raise ValueError("local_jsonl requires explicit train input_paths")
        paths = [Path(path).resolve() for path in paths]
        if len(set(paths)) != len(paths) or any(not path.is_file() for path in paths):
            raise ValueError("Train shard paths must be distinct existing files")
        provenance = {
            "kind": kind,
            "split": "train",
            "n_shards": len(paths),
            "shards": [{"shard_id": index, "path": str(path), "sha256": file_sha256(path), "size_bytes": path.stat().st_size} for index, path in enumerate(paths)],
        }
        return provenance, paths
    if kind not in ("huggingface", "hf"):
        raise ValueError("Source kind must be huggingface or local_jsonl")
    dataset_name = source.get("dataset_name", data.get("dataset_name", DATASET_NAME))
    revision = source.get("revision", data.get("revision", DATASET_REVISION))
    if dataset_name != DATASET_NAME or not isinstance(revision, str) or not re.fullmatch(r"[a-fA-F0-9]{40}", revision):
        raise ValueError("The TinySMS source requires an immutable commit revision")
    from datasets import load_dataset
    dataset = load_dataset(DATASET_NAME, split="train", revision=revision, streaming=True)
    n_shards = int(dataset.n_shards)
    if n_shards < 1 or n_shards != int(source.get("expected_shards", data.get("expected_shards", 60))):
        raise ValueError("The source shard count differs from the expected count")
    return {"kind": "huggingface", "dataset_name": dataset_name, "revision": revision, "split": "train", "n_shards": n_shards}, dataset


def _rows(source, opened, shard_id):
    if source["kind"] == "huggingface":
        yield from opened.shard(num_shards=source["n_shards"], index=shard_id, contiguous=True)
        return
    path = opened[shard_id]
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON in train shard {shard_id}, line {number}") from error
    if file_sha256(path) != source["shards"][shard_id]["sha256"]:
        raise ValueError("A source train shard changed during preparation")


def _record(row, settings):
    if not isinstance(row, dict) or not isinstance(row.get("question"), str) or not isinstance(row.get("code"), str):
        raise ValueError("Every source row must contain string question and code")
    if row.get("split", "train") != "train":
        raise ValueError("A local source record declares a non-train split")
    code = normalize_code(row["code"]) if settings["normalize_code"] else row["code"]
    return {"question": row["question"], "code": code}


def _write_json(path, value):
    with path.open("w", encoding="utf-8") as handle:
        handle.write(canonical_json(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _spool(root, source, opened, settings, supplied_tokenizer):
    raw_dir = root / "raw"
    sample_dir = root / "samples"
    raw_dir.mkdir()
    sample_dir.mkdir()
    shard_count = source["n_shards"]
    sample_size = settings["tokenizer_sample_size"]
    if not supplied_tokenizer and sample_size < shard_count:
        raise ValueError("The tokenizer sample budget must cover every source shard")
    quotas = [sample_size // shard_count + int(index < sample_size % shard_count) for index in range(shard_count)]
    statistics = []
    for shard_id in range(shard_count):
        rng = random.Random(settings["seed"] + shard_id * 104729)
        reservoir = []
        counts = {"raw_rows": 0, "train_rows": 0, "validation_rows": 0, "tokenizer_samples": 0}
        with (raw_dir / f"{shard_id:05d}.jsonl").open("w", encoding="utf-8") as target:
            for row in _rows(source, opened, shard_id):
                record = _record(row, settings)
                validation = is_validation(record["question"], settings["validation_fraction"], settings["seed"])
                counts["raw_rows"] += 1
                counts["validation_rows" if validation else "train_rows"] += 1
                target.write(json.dumps(record, ensure_ascii=False) + "\n")
                if not validation and not supplied_tokenizer:
                    index = counts["train_rows"]
                    if len(reservoir) < quotas[shard_id]:
                        reservoir.append(record)
                    else:
                        replacement = rng.randrange(index)
                        if replacement < quotas[shard_id]:
                            reservoir[replacement] = record
            target.flush()
            os.fsync(target.fileno())
        counts["tokenizer_samples"] = len(reservoir)
        with (sample_dir / f"{shard_id:05d}.jsonl").open("w", encoding="utf-8") as target:
            for record in reservoir:
                target.write(json.dumps(record, ensure_ascii=False) + "\n")
        statistics.append(counts)
    if not sum(shard["train_rows"] for shard in statistics):
        raise ValueError("The source training split is empty")
    return statistics


def _tokenizer(root, settings, tokenizer_json):
    if tokenizer_json is not None:
        tokenizer = Tokenizer.from_str(tokenizer_json)
    else:
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=settings["vocab_size"],
            min_frequency=settings["tokenizer_min_frequency"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=list(SPECIAL_TOKENS),
            show_progress=False,
        )
        def texts():
            for path in sorted((root / "samples").glob("*.jsonl")):
                with path.open(encoding="utf-8") as handle:
                    for line in handle:
                        record = json.loads(line)
                        yield record["question"]
                        yield record["code"]
        tokenizer.train_from_iterator(texts(), trainer=trainer)
        tokenizer_json = tokenizer.to_str()
    validate_tokenizer(tokenizer)
    with (root / "tokenizer.json").open("w", encoding="utf-8") as handle:
        handle.write(tokenizer_json)
        handle.flush()
        os.fsync(handle.fileno())
    return tokenizer


def _file_spec(root, path):
    return {"path": str(path.relative_to(root)), "sha256": file_sha256(path), "size_bytes": path.stat().st_size}


class _ShardWriter:
    def __init__(self, root, directory, split):
        self.root = root
        self.count = 0
        self.offsets = {"source": 0, "target": 0}
        self.paths = {}
        self.handles = {}
        self.lengths = {kind: {"min": None, "max": 0, "sum": 0} for kind in ("source", "target")}
        for kind in ("source", "target"):
            for suffix in ("ids", "offsets"):
                key = kind + "_" + suffix
                path = directory / f"{split}.{key}.bin"
                self.paths[key] = path
                self.handles[key] = path.open("wb")
            self.handles[kind + "_offsets"].write(np.asarray([0], dtype="<i8").tobytes())

    def write(self, records):
        if not records:
            return
        for kind in ("source", "target"):
            flat = []
            ends = []
            statistics = self.lengths[kind]
            for record in records:
                ids = record[kind]
                length = len(ids)
                flat.extend(ids)
                self.offsets[kind] += length
                ends.append(self.offsets[kind])
                statistics["min"] = length if statistics["min"] is None else min(statistics["min"], length)
                statistics["max"] = max(statistics["max"], length)
                statistics["sum"] += length
            self.handles[kind + "_ids"].write(np.asarray(flat, dtype="<i4").tobytes())
            self.handles[kind + "_offsets"].write(np.asarray(ends, dtype="<i8").tobytes())
        self.count += len(records)

    def close(self):
        for handle in self.handles.values():
            if not handle.closed:
                handle.flush()
                os.fsync(handle.fileno())
                handle.close()
        return {"count": self.count, "lengths": self.lengths, "files": {key: _file_spec(self.root, path) for key, path in self.paths.items()}}


def _encode_shard(root, shard_id, tokenizer, settings):
    directory = root / f"shard_{shard_id:05d}"
    directory.mkdir()
    writers = {split: _ShardWriter(root, directory, split) for split in ("train", "validation")}
    filtered = {"train": 0, "validation": 0}
    def write(records):
        encoded = tokenizer.encode_batch([text for row in records for text in (row["question"], row["code"])], add_special_tokens=False)
        selected = {"train": [], "validation": []}
        for index, row in enumerate(records):
            ids = {"source": encoded[index * 2].ids, "target": encoded[index * 2 + 1].ids}
            for kind, original in (("source", row["question"]), ("target", row["code"])):
                if any(token < 3 for token in ids[kind]) or tokenizer.decode(ids[kind], skip_special_tokens=False) != original:
                    raise ValueError("The tokenizer must preserve source and code exactly without embedded special tokens")
                ids[kind].append(2)
            split = "validation" if is_validation(row["question"], settings["validation_fraction"], settings["seed"]) else "train"
            if any(len(ids[kind]) > settings["max_" + kind + "_length"] for kind in ("source", "target")):
                filtered[split] += 1
            else:
                selected[split].append(ids)
        for split, values in selected.items():
            writers[split].write(values)
    try:
        with (root / "raw" / f"{shard_id:05d}.jsonl").open(encoding="utf-8") as handle:
            batch = []
            for line in handle:
                batch.append(json.loads(line))
                if len(batch) >= settings["preparation_batch_size"]:
                    write(batch)
                    batch = []
            if batch:
                write(batch)
    finally:
        records = {split: writer.close() for split, writer in writers.items()}
    return {"shard_id": shard_id, "filtered_rows": filtered, **records}


def prepare_data(config):
    data, settings = _settings(config)
    if not data.get("output_dir"):
        raise ValueError("data.output_dir is required")
    output = Path(data["output_dir"]).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.parent / ("." + output.name + ".prepare.lock")
    lock = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(lock)
    staging = None
    try:
        source, opened = _source(data)
        supplied = data.get("tokenizer_path") or None
        tokenizer_json = Path(supplied).read_text(encoding="utf-8") if supplied else None
        if tokenizer_json is not None:
            validate_tokenizer(Tokenizer.from_str(tokenizer_json))
        tokenizer_source = {"kind": "provided", "sha256": hashlib.sha256(tokenizer_json.encode("utf-8")).hexdigest()} if supplied else {"kind": "train_stratified_reservoir"}
        request = fingerprint({"settings": settings, "source": source, "tokenizer_source": tokenizer_source})
        manifest_path = output / "manifest.json"
        if manifest_path.is_file():
            existing = PreparedCorpus(manifest_path)
            if existing.manifest.get("request_fingerprint") != request:
                raise ValueError("The existing output belongs to a different preparation request")
            return manifest_path
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError("The output directory must be empty or contain the same prepared corpus")
        staging = Path(tempfile.mkdtemp(prefix="." + output.name + ".preparing-", dir=output.parent))
        statistics = _spool(staging, source, opened, settings, supplied is not None)
        tokenizer = _tokenizer(staging, settings, tokenizer_json)
        shards = []
        for shard_id, counts in enumerate(statistics):
            shards.append({**_encode_shard(staging, shard_id, tokenizer, settings), **counts})
        shutil.rmtree(staging / "raw")
        shutil.rmtree(staging / "samples")
        manifest = {
            "format_version": FORMAT_VERSION,
            "source": source,
            "preparation": settings,
            "request_fingerprint": request,
            "tokenizer_source": tokenizer_source,
            "tokenizer": _file_spec(staging, staging / "tokenizer.json"),
            "metadata": {"pad_id": 0, "bos_id": 1, "eos_id": 2, "vocab_size": tokenizer.get_vocab_size(), "max_source_length": settings["max_source_length"], "max_target_length": settings["max_target_length"]},
            "counts": {split: sum(shard[split]["count"] for shard in shards) for split in ("train", "validation")},
            "shards": shards,
        }
        manifest["fingerprint"] = fingerprint(manifest)
        _write_json(staging / "manifest.json", manifest)
        PreparedCorpus(staging)
        if output.exists():
            output.rmdir()
        os.rename(staging, output)
        staging = None
        return manifest_path
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        lock_path.unlink(missing_ok=True)


def main():
    from .config import load_config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    data = dict(config["data"])
    data.setdefault("seed", config.get("seed", 42))
    print(prepare_data(data))


if __name__ == "__main__":
    main()
