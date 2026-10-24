import bisect
import hashlib
import json
import operator
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>")
FORMAT_VERSION = 1


def canonical_json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def fingerprint(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_tokenizer(tokenizer):
    if [tokenizer.token_to_id(token) for token in SPECIAL_TOKENS] != [0, 1, 2]:
        raise ValueError("Tokenizer must assign pad=0, bos=1, eos=2")
    tokenizer.encode_special_tokens = True


def _artifact(root, specification, verify_hashes):
    path = (root / specification["path"]).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Artifact path is missing or outside the corpus directory")
    if path.stat().st_size != specification["size_bytes"]:
        raise ValueError(f"Artifact size changed: {specification['path']}")
    if verify_hashes and file_sha256(path) != specification["sha256"]:
        raise ValueError(f"Artifact hash changed: {specification['path']}")
    return path


def _mapped(path, dtype):
    if path.stat().st_size % np.dtype(dtype).itemsize:
        raise ValueError("Array byte length is inconsistent with its dtype")
    if not path.stat().st_size:
        return np.empty(0, dtype=dtype)
    return np.memmap(path, dtype=dtype, mode="r")


def _check_ragged(tokens, offsets, count, max_length, vocab_size):
    if len(offsets) != count + 1 or offsets[0] != 0 or offsets[-1] != len(tokens):
        raise ValueError("Ragged offsets do not match the record count")
    eos_count = 0
    for start in range(0, len(tokens), 1_000_000):
        part = tokens[start:start + 1_000_000]
        if np.any(part < 2) or np.any(part >= vocab_size):
            raise ValueError("Token IDs are outside the allowed vocabulary")
        eos_count += int(np.count_nonzero(part == 2))
    if eos_count != count:
        raise ValueError("Each record must contain exactly one EOS")
    for start in range(0, count, 100_000):
        part = offsets[start:min(start + 100_000, count) + 1]
        lengths = np.diff(part)
        if np.any(lengths < 1) or np.any(lengths > max_length):
            raise ValueError("Ragged lengths violate the configured limits")
        if np.any(tokens[part[1:] - 1] != 2):
            raise ValueError("Each record must end in EOS")


class PreparedSplit:
    def __init__(self, corpus, split, verify_hashes):
        self.split = split
        self._root = corpus.root
        self.manifest_fingerprint = corpus.fingerprint
        self.fingerprint = fingerprint({"corpus": corpus.fingerprint, "split": split})
        self.shard_counts = []
        self.shard_ranges = []
        self._arrays = []
        self._array_specs = []
        self._ends = []
        position = 0
        for shard in corpus.manifest["shards"]:
            record = shard[split]
            count = record["count"]
            if type(count) is not int or count < 0:
                raise ValueError("Shard count must be a nonnegative integer")
            arrays = {}
            for kind in ("source", "target"):
                tokens = _mapped(_artifact(corpus.root, record["files"][kind + "_ids"], verify_hashes), "<i4")
                offsets = _mapped(_artifact(corpus.root, record["files"][kind + "_offsets"], verify_hashes), "<i8")
                _check_ragged(tokens, offsets, count, corpus.metadata["max_" + kind + "_length"], corpus.metadata["vocab_size"])
                arrays[kind + "_ids"] = tokens
                arrays[kind + "_offsets"] = offsets
            self.shard_counts.append(count)
            self.shard_ranges.append((position, position + count))
            position += count
            self._ends.append(position)
            self._arrays.append(arrays)
            self._array_specs.append(record["files"])
        self._length = position

    def __len__(self):
        return self._length

    def __getstate__(self):
        state = dict(self.__dict__)
        state.pop("_arrays")
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._arrays = [
            {name: _mapped(_artifact(self._root, specification, False), "<i4" if name.endswith("_ids") else "<i8")
             for name, specification in files.items()}
            for files in self._array_specs
        ]

    def __getitem__(self, index):
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard_id = bisect.bisect_right(self._ends, index)
        local_index = index - self.shard_ranges[shard_id][0]
        arrays = self._arrays[shard_id]
        result = {"shard_id": shard_id}
        for kind in ("source", "target"):
            start, end = arrays[kind + "_offsets"][local_index:local_index + 2]
            result[kind + "_ids"] = arrays[kind + "_ids"][start:end].tolist()
        return result


class PreparedCorpus:
    def __init__(self, path, verify_hashes=True):
        path = Path(path).resolve()
        manifest_path = path / "manifest.json" if path.is_dir() else path
        self.root = manifest_path.parent
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        content = dict(self.manifest)
        self.fingerprint = content.pop("fingerprint", None)
        if self.manifest.get("format_version") != FORMAT_VERSION or self.fingerprint != fingerprint(content):
            raise ValueError("Invalid corpus manifest format or fingerprint")
        if self.manifest["source"].get("split") != "train":
            raise ValueError("Prepared corpora must originate from the train split")
        shards = self.manifest["shards"]
        if not shards or [shard["shard_id"] for shard in shards] != list(range(len(shards))):
            raise ValueError("Every source shard must appear exactly once in order")
        self.metadata = dict(self.manifest["metadata"])
        self.metadata["fingerprint"] = self.fingerprint
        if [self.metadata.get(name) for name in ("pad_id", "bos_id", "eos_id")] != [0, 1, 2]:
            raise ValueError("Corpus special token IDs are inconsistent")
        tokenizer_path = _artifact(self.root, self.manifest["tokenizer"], verify_hashes)
        self.tokenizer_json = tokenizer_path.read_text(encoding="utf-8")
        self.tokenizer = Tokenizer.from_str(self.tokenizer_json)
        validate_tokenizer(self.tokenizer)
        if self.tokenizer.get_vocab_size() != self.metadata["vocab_size"]:
            raise ValueError("Corpus vocabulary size differs from its tokenizer")
        self.train = PreparedSplit(self, "train", verify_hashes)
        self.validation = PreparedSplit(self, "validation", verify_hashes)
        if not len(self.train):
            raise ValueError("The prepared training split is empty")
        for split in ("train", "validation"):
            if len(getattr(self, split)) != self.manifest["counts"][split]:
                raise ValueError("Split count differs from its shard totals")


class ShardBatchSampler:
    def __init__(self, dataset, batch_size, rank=0, world_size=1, seed=42, mode="equal_shards", state=None, drop_last=True):
        self.dataset = dataset
        values = (batch_size, rank, world_size, seed)
        if any(type(value) is not int for value in values):
            raise ValueError("Sampler sizes, rank and seed must be integers")
        if batch_size < 1 or world_size < 1 or not 0 <= rank < world_size or seed < 0:
            raise ValueError("Sampler sizes, rank or seed are invalid")
        if mode not in ("equal_shards", "uniform_examples"):
            raise ValueError("Unknown shard sampling mode")
        if type(drop_last) is not bool:
            raise ValueError("drop_last must be a boolean")
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.mode = mode
        self.drop_last = bool(drop_last)
        self.global_batch_size = batch_size * world_size
        self.shard_counts = list(dataset.shard_counts)
        if any(type(count) is not int or count < 0 for count in self.shard_counts):
            raise ValueError("Invalid shard counts")
        position = 0
        self.shard_ranges = []
        for count in self.shard_counts:
            self.shard_ranges.append((position, position + count))
            position += count
        if position != len(dataset) or list(dataset.shard_ranges) != self.shard_ranges:
            raise ValueError("Dataset shard ranges are inconsistent")
        if len(dataset) < self.global_batch_size:
            raise ValueError("The corpus is smaller than one global batch")
        active_counts = [count for count in self.shard_counts if count]
        if mode == "equal_shards" and self.global_batch_size > len(active_counts) * min(active_counts):
            raise ValueError("A distinct global batch is incompatible with equal_shards; reduce batch size or use uniform_examples")
        self.epoch_size = len(dataset) if mode == "uniform_examples" else len(active_counts) * max(active_counts)
        remainder = self.epoch_size % self.global_batch_size
        if not self.drop_last and 0 < remainder < world_size:
            raise ValueError("The partial global batch would leave a rank empty; use drop_last=True")
        self.batches_per_epoch = self.epoch_size // self.global_batch_size
        if not self.drop_last and remainder:
            self.batches_per_epoch += 1
        self._config = {
            "dataset_fingerprint": dataset.fingerprint,
            "manifest_fingerprint": getattr(dataset, "manifest_fingerprint", dataset.fingerprint),
            "shard_counts": self.shard_counts,
            "batch_size": batch_size,
            "rank": rank,
            "world_size": world_size,
            "seed": seed,
            "mode": mode,
            "drop_last": self.drop_last,
            "algorithm": "global-pcg64-shard-interleave-v1",
        }
        self._identity = fingerprint(self._config)
        self._consumed = 0
        self._issued = 0
        self._cached_epoch = None
        self._cached_plan = None
        if state is not None:
            if state.get("format_version") != FORMAT_VERSION or state.get("identity") != self._identity or state.get("config") != self._config:
                raise ValueError("Sampler state belongs to a different corpus or configuration")
            consumed = state.get("consumed_batches")
            if type(consumed) is not int or consumed < 0:
                raise ValueError("Invalid consumed batch count")
            if (state.get("epoch"), state.get("offset")) != divmod(consumed, self.batches_per_epoch):
                raise ValueError("Sampler epoch and offset disagree with the consumed count")
            self._consumed = consumed
            self._issued = consumed

    def _plan(self, epoch):
        if self._cached_epoch == epoch:
            return self._cached_plan
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([self.seed, epoch])))
        if self.mode == "uniform_examples":
            plan = rng.permutation(len(self.dataset))
        else:
            active = [index for index, count in enumerate(self.shard_counts) if count]
            active = rng.permutation(active)
            quota = max(self.shard_counts)
            plan = np.empty(quota * len(active), dtype=np.int64)
            for column, shard_id in enumerate(active):
                count = self.shard_counts[shard_id]
                order = rng.permutation(count) + self.shard_ranges[shard_id][0]
                destination = plan[column::len(active)]
                for start in range(0, quota, count):
                    destination[start:start + count] = order[:min(count, quota - start)]
        self._cached_epoch = epoch
        self._cached_plan = plan
        return plan

    def __iter__(self):
        produced = self._consumed
        while True:
            epoch, offset = divmod(produced, self.batches_per_epoch)
            plan = self._plan(epoch)
            start = offset * self.global_batch_size
            batch = plan[start:min(start + self.global_batch_size, len(plan))]
            produced += 1
            self._issued = max(self._issued, produced)
            yield batch[self.rank::self.world_size].tolist()

    def commit(self, n=1):
        if type(n) is not int or n < 0 or self._consumed + n > self._issued:
            raise ValueError("Only issued batches can be committed")
        self._consumed += n

    def state_dict(self):
        epoch, offset = divmod(self._consumed, self.batches_per_epoch)
        return {
            "format_version": FORMAT_VERSION,
            "identity": self._identity,
            "config": json.loads(canonical_json(self._config)),
            "consumed_batches": self._consumed,
            "epoch": epoch,
            "offset": offset,
        }
