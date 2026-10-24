import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from helpers import prepare_tiny_corpus, tiny_training_config, training_fixture
from nlp_codegen.checkpoint import capture_rng_state, load_training_checkpoint, restore_rng_state
from nlp_codegen.config import load_config
from nlp_codegen.data import PreparedCorpus
from nlp_codegen.inference import load_checkpoint
from nlp_codegen.model import CodeTransformer
from nlp_codegen.prepare import prepare_data
from nlp_codegen.trainer import run_training


class TrainerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.rng = capture_rng_state("cpu")
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.environment = patch.dict("os.environ", {"RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0"})
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        restore_rng_state(self.rng)
        torch.set_num_threads(self.threads)
        self.temporary.cleanup()

    def assert_state_equal(self, first, second, path="state"):
        if not (isinstance(first, dict) and isinstance(second, dict)):
            self.assertIs(type(first), type(second), path)
        if isinstance(first, torch.Tensor):
            self.assertTrue(torch.equal(first, second), path)
        elif isinstance(first, np.ndarray):
            np.testing.assert_array_equal(first, second, err_msg=path)
        elif isinstance(first, dict):
            self.assertEqual(set(first), set(second), path)
            for key in first:
                self.assert_state_equal(first[key], second[key], f"{path}.{key}")
        elif isinstance(first, (list, tuple)):
            self.assertEqual(len(first), len(second), path)
            for index, (left, right) in enumerate(zip(first, second)):
                self.assert_state_equal(left, right, f"{path}[{index}]")
        else:
            self.assertEqual(first, second, path)

    def test_all_six_experiment_configurations_load(self):
        root = Path(__file__).resolve().parents[1] / "configs"
        expected = {
            "ce": (False, False, "none", "all"),
            "decoder": (False, False, "none", "decoder"),
            "structure": (True, False, "orthogonal", "all"),
            "grounding": (False, True, "orthogonal", "all"),
            "mtl": (True, True, "orthogonal", "all"),
            "mtl_pcgrad": (True, True, "pcgrad", "all"),
        }
        for name, settings in expected.items():
            with self.subTest(name=name):
                config = load_config(root / f"{name}.yaml")
                self.assertEqual((config["heads"]["structure"], config["heads"]["grounding"],
                                  config["gradients"]["mode"], config["train"]["scope"]), settings)
                self.assertTrue(config["task_sampling"]["always_code"])
                self.assertEqual(config["task_sampling"]["auxiliary_tasks_per_step"], 1)
                self.assertEqual(config["data"]["source"]["split"], "train")

    def test_runtime_stop_and_resume_match_uninterrupted_training_bitwise(self):
        data = prepare_tiny_corpus(self.root)
        uninterrupted = tiny_training_config(self.root, data, name="complete", dropout=0.1)
        uninterrupted["task_sampling"]["weights"]["grounding"] = 1.4
        expected_result = run_training(uninterrupted)
        expected = load_training_checkpoint(expected_result["checkpoint"])
        partial_config = copy.deepcopy(uninterrupted)
        partial_config["train"]["output_dir"] = str(self.root / "partial")
        original_config = copy.deepcopy(partial_config)
        partial_result = run_training(partial_config, max_updates=2)
        self.assertEqual(partial_config, original_config)
        partial = load_training_checkpoint(partial_result["checkpoint"])
        self.assertEqual(partial["step"], 2)
        self.assertEqual(partial["config"]["train"]["max_steps"], 4)
        self.assertEqual(partial["signature"], expected["signature"])
        resumed_config = copy.deepcopy(partial_config)
        resumed_config["train"]["resume"] = partial_result["checkpoint"]
        resumed_result = run_training(resumed_config, max_updates=2)
        actual = load_training_checkpoint(resumed_result["checkpoint"])
        self.assertEqual(actual["step"], 4)
        self.assertEqual(actual["optimizer_steps"], 4)
        for key in ("model_state", "optimizer", "scheduler", "scaler", "ema", "rank_states"):
            self.assert_state_equal(actual[key], expected[key], key)
        rank = actual["rank_states"][0]
        self.assertEqual(rank["tasks"]["steps"], 4)
        self.assertEqual(sum(rank["shard_seen"]), 16)
        self.assertNotEqual(rank["sampler"], partial["rank_states"][0]["sampler"])
        initial_projection = torch.Generator().manual_seed(942).get_state()
        self.assertFalse(torch.equal(rank["projection_rng"], initial_projection))
        metrics = [json.loads(line) for line in (self.root / "complete" / "metrics.jsonl").read_text().splitlines()]
        self.assertEqual([row["tasks/code_sampled"] for row in metrics], [1] * 4)
        for row in metrics:
            self.assertEqual(row["tasks/structure_sampled"] + row["tasks/grounding_sampled"], 1)
        self.assertTrue(any(row["tasks/structure_sampled"] for row in metrics))
        self.assertTrue(any(row["tasks/grounding_sampled"] for row in metrics))

    def test_microbatching_preserves_global_objective_and_update(self):
        config = training_fixture(self.root)
        config["task_sampling"]["auxiliary_tasks_per_step"] = 2
        config["train"]["microbatch_size"] = 4
        full_result = run_training(config, max_updates=1)
        full = load_training_checkpoint(full_result["checkpoint"])
        divided = copy.deepcopy(config)
        divided["train"].update(output_dir=str(self.root / "micro"), microbatch_size=1)
        micro_result = run_training(divided, max_updates=1)
        micro = load_training_checkpoint(micro_result["checkpoint"])
        for task in ("code", "structure", "grounding"):
            self.assertAlmostEqual(full_result["metrics"][f"loss/{task}"],
                                   micro_result["metrics"][f"loss/{task}"], places=6)
        for key in full["model_state"]:
            torch.testing.assert_close(micro["model_state"][key], full["model_state"][key],
                                       rtol=2e-5, atol=2e-6, msg=key)
        for key, state in full["optimizer"]["state"].items():
            for field in ("exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(micro["optimizer"]["state"][key][field], state[field],
                                           rtol=5e-4, atol=2e-8, msg=f"{key}.{field}")
        self.assert_state_equal(micro["rank_states"][0]["sampler"], full["rank_states"][0]["sampler"])

    def test_decoder_scope_freezes_encoder_and_embedding_and_exports_core(self):
        config = training_fixture(self.root)
        corpus = PreparedCorpus(config["data"]["output_dir"])
        model_config = dict(config["model"], vocab_size=corpus.metadata["vocab_size"], pad_id=0, bos_id=1, eos_id=2)
        torch.manual_seed(config["seed"])
        initial = CodeTransformer(model_config).state_dict()
        config["train"]["scope"] = "decoder"
        result = run_training(config, max_updates=2)
        checkpoint = load_training_checkpoint(result["checkpoint"])
        changed_decoder = []
        for name, value in initial.items():
            trained = checkpoint["model_state"][f"core.{name}"]
            if name.startswith(("decoder.", "decoder_norm.")):
                changed_decoder.append(not torch.equal(trained, value))
            else:
                self.assertTrue(torch.equal(trained, value), name)
        self.assertTrue(any(changed_decoder))
        self.assertTrue(torch.equal(checkpoint["model_state"]["core.embedding.weight"], initial["embedding.weight"]))
        exported = torch.load(result["model"], map_location="cpu", weights_only=True)
        self.assertEqual(set(exported["model_state"]), set(initial))
        self.assertEqual(exported["weights"], "ema")
        context = load_checkpoint(result["model"], device="cpu")
        self.assert_state_equal(context["model"].state_dict(), checkpoint["ema"])

    def test_inference_export_cannot_resume_training(self):
        config = training_fixture(self.root)
        result = run_training(config, max_updates=1)
        config["train"]["resume"] = result["model"]
        with self.assertRaisesRegex(ValueError, "inference export"):
            run_training(config, max_updates=1)

    def test_resume_rejects_changed_data_and_optimizer_or_schedule(self):
        config = training_fixture(self.root)
        result = run_training(config, max_updates=1)
        config["train"]["resume"] = result["checkpoint"]
        other_data = prepare_tiny_corpus(self.root, name="other", shard_sizes=(13, 20, 28))
        modifications = (("optimizer", "lr", 0.002), ("train", "max_steps", 5),
                         ("model", "dropout", 0.2), ("data", "output_dir", other_data["output_dir"]))
        for group, key, value in modifications:
            with self.subTest(group=group, key=key):
                altered = copy.deepcopy(config)
                altered[group][key] = value
                with self.assertRaisesRegex(ValueError, "Resume config, data, tokenizer or world size changed"):
                    run_training(altered, max_updates=1)

    def test_fully_masked_auxiliary_targets_keep_code_training_finite(self):
        data = prepare_tiny_corpus(self.root)
        for path in data["source"]["input_paths"]:
            rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
            for row in rows:
                row["code"] = "invalid Python !\n"
            Path(path).write_text("".join(json.dumps(row) + "\n" for row in rows))
        data["output_dir"] = str(self.root / "masked")
        prepare_data(data)
        config = tiny_training_config(self.root, data)
        result = run_training(config, max_updates=1)
        self.assertEqual(result["optimizer_steps"], 1)
        self.assertEqual(result["metrics"]["tasks/code_sampled"], 1)
        self.assertEqual(result["metrics"]["tasks/structure_sampled"], 0)
        self.assertEqual(result["metrics"]["tasks/grounding_sampled"], 0)
        self.assertTrue(np.isfinite(result["metrics"]["loss/code"]))
        self.assertNotIn("loss/structure", result["metrics"])
        self.assertNotIn("loss/grounding", result["metrics"])


if __name__ == "__main__":
    unittest.main()
