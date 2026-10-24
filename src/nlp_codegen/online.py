from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import queue
import random
import threading

import torch

from .data import fingerprint, validate_tokenizer
from .prepare import _record, _rows, _settings, _source, is_validation
from .targets import SIZES, derive_auxiliary_labels
from tokenizers import Tokenizer


def _encode(records, tokenizer, settings):
    encoded = tokenizer.encode_batch([text for row, _ in records for text in (row["question"], row["code"])], add_special_tokens=False)
    for index, (row, position) in enumerate(records):
        result = {"_row_key": position}
        for kind, field in (("source", "question"), ("target", "code")):
            ids = encoded[index * 2 + int(kind == "target")].ids
            if any(value < 3 for value in ids) or tokenizer.decode(ids, skip_special_tokens=False) != row[field]:
                raise ValueError("The tokenizer must preserve question and code exactly")
            result[kind + "_ids"] = ids + [2]
        if all(len(result[kind + "_ids"]) <= settings["max_" + kind + "_length"] for kind in ("source", "target")):
            yield result


class OnlineCorpus:
    def __init__(self, config_data, validation_examples=0):
        self.data, self.settings = _settings(config_data)
        self.online = dict(self.data.get("online", {}))
        self.options = {key: int(self.online.get(key, default)) for key, default in (
            ("cpu_workers", 2), ("prefetch_batches", 2), ("shuffle_buffer", 64),
            ("tokenization_batch_size", 128), ("validation_scan_per_shard", 2048),
        )}
        if any(value < 1 for value in self.options.values()):
            raise ValueError("Online worker, buffer and scan sizes must be positive")
        if type(validation_examples) is not int or validation_examples < 0:
            raise ValueError("validation_examples must be a nonnegative integer")
        tokenizer_path = self.data.get("tokenizer_path")
        if not tokenizer_path:
            raise ValueError("Online mode requires data.tokenizer_path")
        self.tokenizer_json = Path(tokenizer_path).read_text(encoding="utf-8")
        self.tokenizer = Tokenizer.from_str(self.tokenizer_json)
        validate_tokenizer(self.tokenizer)
        self.source, self.opened = _source(self.data)
        self.shard_count = self.source["n_shards"]
        self.fingerprint = fingerprint({
            "format": "online-corpus-v1", "source": self.source, "settings": self.settings,
            "tokenizer_sha256": hashlib.sha256(self.tokenizer_json.encode("utf-8")).hexdigest(),
            "online": self.options, "validation_examples": validation_examples,
        })
        self.metadata = {
            "pad_id": 0, "bos_id": 1, "eos_id": 2, "vocab_size": self.tokenizer.get_vocab_size(),
            "max_source_length": self.settings["max_source_length"],
            "max_target_length": self.settings["max_target_length"], "fingerprint": self.fingerprint,
        }
        self.validation = self._validation(validation_examples) if validation_examples else []

    def _validation(self, limit):
        rng = random.Random(self.settings["seed"] + 1100)
        reservoir, seen = [], 0
        for shard_id in range(self.shard_count):
            batch = []
            def select(records):
                nonlocal seen
                for row in _encode(records, self.tokenizer, self.settings):
                    row.pop("_row_key")
                    row["shard_id"] = shard_id
                    seen += 1
                    if len(reservoir) < limit:
                        reservoir.append(row)
                    else:
                        index = rng.randrange(seen)
                        if index < limit:
                            reservoir[index] = row
            for position, raw in enumerate(_rows(self.source, self.opened, shard_id)):
                if position >= self.options["validation_scan_per_shard"]:
                    break
                row = _record(raw, self.settings)
                if is_validation(row["question"], self.settings["validation_fraction"], self.settings["seed"]):
                    batch.append((row, position))
                if len(batch) >= self.options["tokenization_batch_size"]:
                    select(batch)
                    batch = []
            if batch:
                select(batch)
        return reservoir


class _ShardStream:
    def __init__(self, corpus, shard_id, seed, stop):
        self.corpus, self.shard_id = corpus, shard_id
        self.stop = stop
        self.rng = random.Random(seed + shard_id * 104729)
        self.first_count = None
        self.count = 0
        self.iterator = iter(self._pass())

    def _pass(self):
        corpus, buffer, batch = self.corpus, [], []
        def shuffled(records):
            for row in _encode(records, corpus.tokenizer, corpus.settings):
                row["shard_id"] = self.shard_id
                row["_row_key"] = (self.shard_id, row["_row_key"])
                if len(buffer) < corpus.options["shuffle_buffer"]:
                    buffer.append(row)
                else:
                    index = self.rng.randrange(len(buffer))
                    selected, buffer[index] = buffer[index], row
                    yield selected
        for position, raw in enumerate(_rows(corpus.source, corpus.opened, self.shard_id)):
            if self.stop.is_set():
                return
            row = _record(raw, corpus.settings)
            if is_validation(row["question"], corpus.settings["validation_fraction"], corpus.settings["seed"]):
                continue
            batch.append((row, position))
            if len(batch) >= corpus.options["tokenization_batch_size"]:
                yield from shuffled(batch)
                batch = []
        if batch:
            yield from shuffled(batch)
        self.rng.shuffle(buffer)
        yield from buffer

    def __next__(self):
        try:
            row = next(self.iterator)
        except StopIteration:
            if self.first_count is None:
                self.first_count = self.count
            if not self.count:
                raise
            self.count = 0
            self.iterator = iter(self._pass())
            row = next(self.iterator)
        self.count += 1
        return row


def _pin(value):
    if isinstance(value, torch.Tensor):
        return value.pin_memory()
    if isinstance(value, dict):
        return {key: _pin(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_pin(item) for item in value)
    if isinstance(value, list):
        return [_pin(item) for item in value]
    return value


class OnlineBatchSource:
    def __init__(self, corpus, batch_size, rank, world_size, seed, auxiliary, collate_fn, pin_memory=False, state=None):
        if any(type(value) is not int for value in (batch_size, rank, world_size, seed)):
            raise ValueError("Online batch sizes, rank and seed must be integers")
        if batch_size < 1 or world_size < 1 or not 0 <= rank < world_size or seed < 0:
            raise ValueError("Invalid online batch size, rank, world size or seed")
        self.corpus, self.batch_size, self.rank, self.world_size = corpus, batch_size, rank, world_size
        self.seed, self.auxiliary, self.collate_fn = seed, bool(auxiliary), collate_fn
        self.pin_memory = bool(pin_memory) and torch.cuda.is_available()
        self.config = {
            "corpus_fingerprint": corpus.fingerprint, "batch_size": batch_size, "rank": rank,
            "world_size": world_size, "seed": seed, "auxiliary": self.auxiliary,
            "pin_memory": bool(pin_memory), "algorithm": "roundrobin-buffered-replay-v1",
            "collate_fn": f"{collate_fn.__module__}.{collate_fn.__qualname__}",
        }
        self.identity = fingerprint(self.config)
        self._consumed = 0
        if state is not None:
            if state.get("format_version") != 1 or state.get("identity") != self.identity or state.get("config") != self.config:
                raise ValueError("Online resume source, tokenizer or configuration changed")
            self._consumed = state.get("consumed_batches")
            if type(self._consumed) is not int or self._consumed < 0:
                raise ValueError("Invalid consumed online batch count")
        self._issued = self._consumed
        self._replay_from = self._consumed
        self._stop = threading.Event()
        self._queue = queue.Queue(maxsize=corpus.options["prefetch_batches"])
        self._pool = ThreadPoolExecutor(max_workers=corpus.options["cpu_workers"], thread_name_prefix="online-cpu")
        self._producer = threading.Thread(target=self._produce, name="online-producer", daemon=True)
        self._producer.start()

    def _global_batches(self):
        cursors = [_ShardStream(self.corpus, index, self.seed, self._stop) for index in range(self.corpus.shard_count)]
        order = list(range(len(cursors)))
        rng = random.Random(self.seed)
        rows, unique = [], set()
        global_size = self.batch_size * self.world_size
        while order and not self._stop.is_set():
            rng.shuffle(order)
            active = []
            for shard_id in order:
                if self._stop.is_set():
                    return
                try:
                    row = next(cursors[shard_id])
                    active.append(shard_id)
                except StopIteration:
                    continue
                known = [cursor.first_count for cursor in cursors]
                if all(count is not None for count in known) and sum(known) < global_size:
                    raise ValueError("The eligible training corpus is smaller than one global batch")
                key = row["_row_key"]
                if key in unique:
                    continue
                unique.add(key)
                rows.append(row)
                if len(rows) == global_size:
                    yield rows
                    rows, unique = [], set()
            order = active
        if not self._stop.is_set():
            raise ValueError("The source has no eligible training rows")

    def _collate(self, rows):
        records = []
        for row in rows:
            record = {key: value for key, value in row.items() if key != "_row_key"}
            if self.auxiliary:
                structure, grounding = derive_auxiliary_labels(record["source_ids"], record["target_ids"], self.corpus.tokenizer)
            else:
                structure, grounding = [-100] * len(SIZES), [-100.0] * len(record["target_ids"])
            record.update(structure=structure, grounding=grounding)
            records.append(record)
        batch = self.collate_fn(records)
        return _pin(batch) if self.pin_memory else batch

    def _offer(self, item):
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                pass
        return False

    def _produce(self):
        try:
            for index, rows in enumerate(self._global_batches()):
                if index < self._replay_from:
                    continue
                selected = rows[self.rank::self.world_size]
                if not self._offer(self._pool.submit(self._collate, selected)):
                    return
        except BaseException as error:
            self._offer(error)

    def __iter__(self):
        return self

    def __next__(self):
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if isinstance(item, BaseException):
                self.close()
                raise RuntimeError("Online training source failed") from item
            try:
                batch = item.result()
            except BaseException:
                self.close()
                raise
            self._issued += 1
            return batch
        raise StopIteration

    def commit(self, n=1):
        if type(n) is not int or n < 0 or self._consumed + n > self._issued:
            raise ValueError("Only delivered online batches can be committed")
        self._consumed += n

    def state_dict(self):
        return {"format_version": 1, "identity": self.identity, "config": json.loads(json.dumps(self.config)), "consumed_batches": self._consumed}

    def close(self):
        self._stop.set()
        self._pool.shutdown(wait=False, cancel_futures=True)
        if threading.current_thread() is not self._producer:
            self._producer.join(timeout=0.2)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
