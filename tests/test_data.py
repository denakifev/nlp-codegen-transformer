import copy
import json
import pickle
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import numpy as np

from nlp_codegen.data import PreparedCorpus, ShardBatchSampler, file_sha256
from nlp_codegen.prepare import DATASET_NAME, DATASET_REVISION, is_validation, normalize_code, prepare_data


class DataTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def config(self, sizes=(4, 12, 0), fraction=0.0, name="corpus", rows=None):
        paths = []
        for shard_id, size in enumerate(sizes):
            path = self.root / f"{name}.train.{shard_id}.jsonl"
            records = rows[shard_id] if rows is not None else [{"question": f"Shard {shard_id}, task {index}: compute {index} + 2.", "code": f"def solve():\n    return {index} + 2\n"} for index in range(size)]
            path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
            paths.append(str(path))
        return {
            "seed": 42,
            "output_dir": str(self.root / name),
            "source": {"kind": "local_jsonl", "split": "train", "input_paths": paths},
            "validation_fraction": fraction,
            "vocab_size": 384,
            "tokenizer_sample_size": 12,
            "tokenizer_min_frequency": 1,
            "max_source_length": 256,
            "max_target_length": 256,
            "preparation_batch_size": 3,
            "normalize_code": False,
        }

    def corpus(self):
        return PreparedCorpus(prepare_data(self.config()))

    def test_mmap_roundtrip_provenance_and_empty_shards(self):
        config = self.config()
        manifest_path = prepare_data({"seed": 42, "data": config})
        corpus = PreparedCorpus(manifest_path)
        self.assertEqual(corpus.train.shard_counts, [4, 12, 0])
        self.assertEqual(corpus.validation.shard_counts, [0, 0, 0])
        self.assertEqual(corpus.train.shard_ranges, [(0, 4), (4, 16), (16, 16)])
        self.assertEqual(corpus.metadata["fingerprint"], corpus.fingerprint)
        self.assertEqual([corpus.tokenizer.token_to_id(name) for name in ("<pad>", "<bos>", "<eos>")], [0, 1, 2])
        self.assertIsInstance(corpus.train._arrays[0]["source_ids"], np.memmap)
        for shard_id, source_path in enumerate(config["source"]["input_paths"]):
            source = corpus.manifest["source"]["shards"][shard_id]
            self.assertEqual(source["sha256"], file_sha256(source_path))
            expected = [json.loads(line) for line in Path(source_path).read_text().splitlines()]
            start, _ = corpus.train.shard_ranges[shard_id]
            for index, row in enumerate(expected):
                record = corpus.train[start + index]
                self.assertEqual(record["shard_id"], shard_id)
                for key, text_key in (("source_ids", "question"), ("target_ids", "code")):
                    self.assertEqual(record[key][-1], 2)
                    self.assertNotIn(2, record[key][:-1])
                    self.assertEqual(corpus.tokenizer.decode(record[key][:-1], skip_special_tokens=False), row[text_key])
        self.assertEqual(corpus.train[-1], corpus.train[len(corpus.train) - 1])
        with self.assertRaises(IndexError):
            corpus.train[len(corpus.train)]
        self.assertEqual(prepare_data(config), manifest_path)

    def test_split_groups_and_stratified_tokenizer_are_train_only(self):
        rows = [[{"question": f"Question {shard}-{index}", "code": f"def solve():\n    return {index}\n"} for index in range(70)] for shard in range(3)]
        rows[0].append({"question": "A   repeated question", "code": "def solve():\n    return 1\n"})
        rows[1].append({"question": " A repeated  question ", "code": "def solve():\n    return 1\n"})
        config = self.config(sizes=(0, 0, 0), fraction=0.3, rows=rows)
        from nlp_codegen import prepare
        original = prepare._tokenizer
        observed = []
        def inspect_samples(root, settings, tokenizer_json):
            for shard_id, path in enumerate(sorted((root / "samples").glob("*.jsonl"))):
                values = [json.loads(line) for line in path.read_text().splitlines()]
                self.assertGreater(len(values), 0)
                for row in values:
                    self.assertFalse(is_validation(row["question"], config["validation_fraction"], 42))
                observed.append(shard_id)
            return original(root, settings, tokenizer_json)
        with mock.patch.object(prepare, "_tokenizer", side_effect=inspect_samples):
            corpus = PreparedCorpus(prepare_data(config))
        self.assertEqual(observed, [0, 1, 2])
        expected = Counter("validation" if is_validation(row["question"], 0.3, 42) else "train" for shard in rows for row in shard)
        self.assertEqual(len(corpus.train), expected["train"])
        self.assertEqual(len(corpus.validation), expected["validation"])
        questions = {}
        for name in ("train", "validation"):
            questions[name] = {" ".join(corpus.tokenizer.decode(row["source_ids"][:-1]).split()) for row in getattr(corpus, name)}
        self.assertFalse(questions["train"] & questions["validation"])
        self.assertEqual(is_validation("A   repeated question", 0.3, 42), is_validation(" A repeated  question ", 0.3, 42))

    def test_dataset_pickle_reopens_maps_without_serializing_token_arrays(self):
        corpus = PreparedCorpus(prepare_data(self.config(sizes=(256, 768, 0))))
        payload = pickle.dumps(corpus.train, protocol=pickle.HIGHEST_PROTOCOL)
        array_bytes = sum(value.nbytes for shard in corpus.train._arrays for value in shard.values())
        self.assertGreater(array_bytes, 100000)
        self.assertLess(len(payload), 16000)
        with mock.patch("nlp_codegen.data.file_sha256", side_effect=AssertionError("Worker restoration must not hash the entire corpus")):
            restored = pickle.loads(payload)
        self.assertIsInstance(restored._arrays[0]["source_ids"], np.memmap)
        self.assertEqual(restored.fingerprint, corpus.train.fingerprint)
        self.assertEqual(restored.shard_counts, corpus.train.shard_counts)
        for index in (0, 255, 256, len(restored) - 1):
            self.assertEqual(restored[index], corpus.train[index])

    def test_equal_shards_oversampling_and_global_rank_partition(self):
        corpus = self.corpus()
        samplers = [ShardBatchSampler(corpus.train, 2, rank=rank, world_size=2) for rank in range(2)]
        iterators = [iter(sampler) for sampler in samplers]
        counts = Counter()
        seen = Counter()
        for _ in range(samplers[0].batches_per_epoch):
            batches = [next(iterator) for iterator in iterators]
            self.assertFalse(set(batches[0]) & set(batches[1]))
            combined = batches[0] + batches[1]
            self.assertEqual(len(set(combined)), 4)
            for index in combined:
                counts[corpus.train[index]["shard_id"]] += 1
                seen[index] += 1
            for sampler in samplers:
                sampler.commit()
        self.assertEqual(counts, {0: 12, 1: 12})
        self.assertEqual(set(seen), set(range(len(corpus.train))))
        self.assertEqual([seen[index] for index in range(4)], [3] * 4)
        self.assertEqual([seen[index] for index in range(4, 16)], [1] * 12)

    def test_uniform_examples_visits_every_row_once(self):
        corpus = self.corpus()
        sampler = ShardBatchSampler(corpus.train, 4, mode="uniform_examples")
        iterator = iter(sampler)
        seen = []
        for _ in range(sampler.batches_per_epoch):
            seen.extend(next(iterator))
            sampler.commit()
        self.assertEqual(sorted(seen), list(range(16)))
        self.assertEqual(Counter(corpus.train[index]["shard_id"] for index in seen), {0: 4, 1: 12})

    def test_partial_global_batches_are_distinct_and_nonempty(self):
        corpus = PreparedCorpus(prepare_data(self.config(sizes=(4, 13, 0))))
        samplers = [ShardBatchSampler(corpus.train, 2, rank=rank, world_size=3, mode="uniform_examples", drop_last=False) for rank in range(3)]
        iterators = [iter(sampler) for sampler in samplers]
        seen = []
        for _ in range(samplers[0].batches_per_epoch):
            batches = [next(iterator) for iterator in iterators]
            self.assertTrue(all(batches))
            combined = [index for batch in batches for index in batch]
            self.assertEqual(len(set(combined)), len(combined))
            seen.extend(combined)
        self.assertEqual(sorted(seen), list(range(17)))
        with self.assertRaises(ValueError):
            ShardBatchSampler(corpus.train, 4, world_size=4, mode="uniform_examples", drop_last=False)

    def test_preparation_is_reproducible_and_sample_budget_covers_shards(self):
        config = self.config()
        first = PreparedCorpus(prepare_data(config))
        config["output_dir"] = str(self.root / "second")
        second = PreparedCorpus(prepare_data(config))
        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertEqual(first.tokenizer_json, second.tokenizer_json)
        for index in range(len(first.train)):
            self.assertEqual(first.train[index], second.train[index])
        config["output_dir"] = str(self.root / "insufficient")
        config["tokenizer_sample_size"] = 2
        with self.assertRaises(ValueError):
            prepare_data(config)
        self.assertFalse(Path(config["output_dir"]).exists())

    def test_prefetch_consumed_resume_crosses_epoch_exactly(self):
        corpus = self.corpus()
        sampler = ShardBatchSampler(corpus.train, 2)
        iterator = iter(sampler)
        before = sampler.state_dict()
        prefetched = [next(iterator) for _ in range(6)]
        self.assertEqual(sampler.state_dict(), before)
        sampler.commit(2)
        state = sampler.state_dict()
        self.assertEqual(state["consumed_batches"], 2)
        restored = ShardBatchSampler(corpus.train, 2, state=json.loads(json.dumps(state)))
        resumed = iter(restored)
        self.assertEqual([next(resumed) for _ in range(4)], prefetched[2:])
        self.assertEqual([next(resumed) for _ in range(30)], [next(iterator) for _ in range(30)])
        restored.commit(34)
        expected = restored.state_dict()
        self.assertEqual((expected["epoch"], expected["offset"]), divmod(36, restored.batches_per_epoch))
        with self.assertRaises(ValueError):
            restored.commit()

    def test_changed_sampler_identity_and_invalid_batches_rejected(self):
        corpus = self.corpus()
        sampler = ShardBatchSampler(corpus.train, 2)
        state = sampler.state_dict()
        for changes in ({"batch_size": 4}, {"world_size": 2}, {"seed": 43}, {"mode": "uniform_examples"}, {"drop_last": False}):
            with self.assertRaises(ValueError):
                ShardBatchSampler(corpus.train, state=state, **{"batch_size": 2, **changes})
        changed = copy.copy(corpus.train)
        changed.fingerprint = "changed"
        with self.assertRaises(ValueError):
            ShardBatchSampler(changed, 2, state=state)
        broken = copy.deepcopy(state)
        broken["offset"] += 1
        with self.assertRaises(ValueError):
            ShardBatchSampler(corpus.train, 2, state=broken)
        with self.assertRaises(ValueError):
            ShardBatchSampler(corpus.train, 17)
        with self.assertRaises(ValueError):
            ShardBatchSampler(corpus.train, 9)
        with self.assertRaises(ValueError):
            sampler.commit()

    def test_hash_verification_and_transactional_output(self):
        config = self.config()
        manifest_path = prepare_data(config)
        original = manifest_path.read_bytes()
        modified = copy.deepcopy(config)
        modified["seed"] += 1
        with self.assertRaises(ValueError):
            prepare_data(modified)
        self.assertEqual(manifest_path.read_bytes(), original)
        manifest = json.loads(original)
        artifact = manifest_path.parent / manifest["shards"][0]["train"]["files"]["source_ids"]["path"]
        with artifact.open("r+b") as handle:
            handle.write(np.asarray([3], dtype="<i4").tobytes())
        with self.assertRaises(ValueError):
            PreparedCorpus(manifest_path)
        bad = self.config(sizes=(1,), name="broken", rows=[[{"question": "missing code"}]])
        with self.assertRaises(ValueError):
            prepare_data(bad)
        self.assertFalse(Path(bad["output_dir"]).exists())
        self.assertFalse((self.root / ".broken.prepare.lock").exists())
        self.assertFalse(list(self.root.glob(".broken.preparing-*")))

    def test_no_program_execution_and_local_source_is_offline(self):
        marker = self.root / "execution_marker"
        code = f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\ndef solve():\n    return 1\n"
        config = self.config(sizes=(1,), rows=[[{"question": "One plus zero?", "code": code}]])
        with mock.patch("datasets.load_dataset", side_effect=AssertionError("Local preparation must not use a remote loader")):
            corpus = PreparedCorpus(prepare_data(config))
        self.assertFalse(marker.exists())
        self.assertEqual(corpus.tokenizer.decode(corpus.train[0]["target_ids"][:-1]), code)

    def test_huggingface_loader_receives_only_pinned_train_split(self):
        records = [[{"question": f"Question {index}", "code": "def solve():\n    return 1\n"}] for index in range(2)]
        class FakeSource:
            n_shards = 2
            def shard(inner, num_shards, index, contiguous):
                self.assertEqual(num_shards, 2)
                self.assertTrue(contiguous)
                return iter(records[index])
        config = self.config(sizes=(1,), name="remote")
        config["source"] = {"kind": "huggingface", "revision": DATASET_REVISION, "split": "train", "expected_shards": 2}
        with mock.patch("datasets.load_dataset", return_value=FakeSource()) as loader:
            corpus = PreparedCorpus(prepare_data(config))
        loader.assert_called_once_with(DATASET_NAME, split="train", revision=DATASET_REVISION, streaming=True)
        self.assertEqual(corpus.train.shard_counts, [1, 1])
        invalid = copy.deepcopy(config)
        invalid["output_dir"] = str(self.root / "invalid")
        invalid["source"]["split"] = "validation"
        with mock.patch("datasets.load_dataset", side_effect=AssertionError("Invalid split must be rejected before loading")):
            with self.assertRaises(ValueError):
                prepare_data(invalid)

    def test_provided_tokenizer_and_static_normalization(self):
        first = self.corpus()
        config = self.config(name="reused")
        config["tokenizer_path"] = str(first.root / "tokenizer.json")
        corpus = PreparedCorpus(prepare_data(config))
        self.assertEqual(corpus.tokenizer_json, first.tokenizer_json)
        self.assertEqual(corpus.manifest["tokenizer_source"]["kind"], "provided")
        self.assertEqual(sum(shard["tokenizer_samples"] for shard in corpus.manifest["shards"]), 0)
        self.assertEqual(normalize_code("def solve():\n return 1+2\n"), "def solve():\n    return 1 + 2\n")
        self.assertEqual(normalize_code("invalid python !"), "invalid python !")
        self.assertEqual(normalize_code("print(__file__)"), "print(__file__)")


if __name__ == "__main__":
    unittest.main()
