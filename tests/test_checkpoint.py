import copy
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from nlp_codegen.checkpoint import (
    atomic_save, capture_rng_state, export_checkpoint, load_training_checkpoint,
    restore_optimizer_device, restore_rng_state, training_signature,
)
from nlp_codegen.inference import load_checkpoint
from nlp_codegen.model import CodeTransformer


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.initial_rng = capture_rng_state("cpu")

    def tearDown(self):
        restore_rng_state(self.initial_rng)
        self.temporary.cleanup()

    def test_rng_restores_python_numpy_and_torch(self):
        random.seed(13)
        np.random.seed(14)
        torch.manual_seed(15)
        state = capture_rng_state("cpu")
        expected = (random.random(), np.random.random(5), torch.rand(5))
        random.random()
        np.random.random(30)
        torch.rand(30)
        restore_rng_state(state)
        self.assertEqual(random.random(), expected[0])
        np.testing.assert_array_equal(np.random.random(5), expected[1])
        self.assertTrue(torch.equal(torch.rand(5), expected[2]))

    def test_active_cuda_rng_uses_saved_device(self):
        fake_state = torch.arange(8, dtype=torch.uint8)
        with patch("torch.cuda.get_rng_state", return_value=fake_state) as get_state:
            state = capture_rng_state("cuda:2")
        get_state.assert_called_once_with(2)
        self.assertEqual(state["cuda_device"], 2)
        with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.set_rng_state") as set_state:
            restore_rng_state(state)
        self.assertTrue(torch.equal(set_state.call_args.args[0], fake_state))
        self.assertEqual(set_state.call_args.kwargs, {"device": 2})

    def test_default_rng_capture_does_not_initialize_cuda(self):
        with patch("torch.cuda.is_initialized", return_value=False), patch("torch.cuda.get_rng_state") as get_state:
            state = capture_rng_state()
        get_state.assert_not_called()
        self.assertIsNone(state["cuda"])
        self.assertIsNone(state["mps"])

    def test_mps_rng_round_trip_calls_only_requested_backend(self):
        fake_state = torch.arange(8, dtype=torch.uint8)
        with patch("torch.mps.get_rng_state", return_value=fake_state) as get_state:
            state = capture_rng_state("mps")
        get_state.assert_called_once_with()
        with patch("torch.backends.mps.is_available", return_value=True), patch("torch.mps.set_rng_state") as set_state:
            restore_rng_state(state)
        self.assertTrue(torch.equal(set_state.call_args.args[0], fake_state))

    def test_atomic_save_preserves_previous_file_on_failure(self):
        path = self.root / "checkpoints/training.pt"
        atomic_save(path, {"value": torch.tensor([1, 2])})
        previous = path.read_bytes()
        with patch("nlp_codegen.checkpoint.torch.save", side_effect=RuntimeError("save failed")):
            with self.assertRaisesRegex(RuntimeError, "save failed"):
                atomic_save(path, {"value": torch.tensor([3, 4])})
        self.assertEqual(path.read_bytes(), previous)
        self.assertEqual(list(path.parent.iterdir()), [path])

    def test_restored_training_matches_next_uninterrupted_step(self):
        random.seed(3)
        np.random.seed(4)
        torch.manual_seed(5)
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

        def update(current_model, current_optimizer):
            inputs = torch.randn(4, 3) * random.random()
            targets = torch.from_numpy(np.random.randn(4, 2)).float()
            current_optimizer.zero_grad()
            loss = (current_model(inputs) - targets).square().mean()
            loss.backward()
            current_optimizer.step()
            return loss.detach()

        update(model, optimizer)
        path = self.root / "training.pt"
        atomic_save(path, {
            "format": "nlp_codegen_training_v1",
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "rng_state": capture_rng_state("cpu"),
        })
        expected_loss = update(model, optimizer)
        expected_state = copy.deepcopy(model.state_dict())
        checkpoint = load_training_checkpoint(path)
        resumed = torch.nn.Linear(3, 2)
        resumed.load_state_dict(checkpoint["model_state"])
        resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=0.01)
        resumed_optimizer.load_state_dict(checkpoint["optimizer_state"])
        restore_optimizer_device(resumed_optimizer, "cpu")
        restore_rng_state(checkpoint["rng_state"])
        actual_loss = update(resumed, resumed_optimizer)
        self.assertTrue(torch.equal(actual_loss, expected_loss))
        for key, tensor in resumed.state_dict().items():
            self.assertTrue(torch.equal(tensor, expected_state[key]))

    def test_signature_ignores_resume_output_initialization_and_comet(self):
        config = {
            "model": {"d_model": 16},
            "train": {"resume": None, "output_dir": "outputs/run", "max_steps": 10, "amp": "fp32", "seed": 42},
            "optimizer": {"lr": 0.01},
            "tasks": {"auxiliary_weight": 0.2},
            "data": {"max_source_length": 32},
            "comet": {"enabled": False},
        }
        before = copy.deepcopy(config)
        expected = training_signature(config, "corpus", "tokenizer", 1)
        equivalent = copy.deepcopy(config)
        equivalent["train"]["resume"] = "checkpoints/training.pt"
        equivalent["train"]["output_dir"] = "outputs/another"
        equivalent["train"]["init_checkpoint"] = "checkpoints/model.pt"
        equivalent["comet"] = {"enabled": True, "offline": True, "experiment_key": "run-key"}
        self.assertEqual(training_signature(equivalent, "corpus", "tokenizer", 1), expected)
        self.assertEqual(config, before)
        for group, key, value in (
            ("model", "d_model", 32),
            ("train", "max_steps", 11),
            ("train", "amp", "bf16"),
            ("train", "seed", 43),
            ("optimizer", "lr", 0.02),
            ("tasks", "auxiliary_weight", 0.3),
            ("data", "max_source_length", 64),
        ):
            with self.subTest(group=group, key=key):
                changed = copy.deepcopy(config)
                changed[group][key] = value
                self.assertNotEqual(training_signature(changed, "corpus", "tokenizer", 1), expected)
        self.assertNotEqual(training_signature(config, "other-corpus", "tokenizer", 1), expected)
        self.assertNotEqual(training_signature(config, "corpus", "other-tokenizer", 1), expected)
        self.assertNotEqual(training_signature(config, "corpus", "tokenizer", 2), expected)

    def test_signature_is_stable_for_mapping_order_and_path_values(self):
        first = {"train": {"max_steps": 3}, "data": {"path": Path("data/train.jsonl")}}
        second = {"data": {"path": "data/train.jsonl"}, "train": {"max_steps": 3}}
        self.assertEqual(training_signature(first, "corpus", "tokenizer", 1), training_signature(second, "corpus", "tokenizer", 1))

    def test_export_is_core_only_and_loads_for_inference(self):
        config = {
            "vocab_size": 5, "pad_id": 0, "bos_id": 1, "eos_id": 2,
            "d_model": 8, "num_heads": 2, "d_ff": 16,
            "encoder_layers": 1, "decoder_layers": 1,
        }
        model = torch.nn.ModuleDict({"core": CodeTransformer(config), "head": torch.nn.Linear(8, 3)})
        tokenizer = Tokenizer(WordLevel({"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3, "word": 4}, unk_token="[UNK]"))
        path = self.root / "model.pt"
        metadata = {"rows": np.int64(2), "path": Path("data/train.jsonl")}
        export_checkpoint(path, model["core"], config, tokenizer.to_str(), metadata, 1)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        self.assertEqual(set(payload), {"format", "model_config", "model_state", "tokenizer_json", "metadata", "step", "weights"})
        self.assertEqual(set(payload["model_state"]), set(model["core"].state_dict()))
        self.assertEqual(payload["metadata"], {"rows": 2, "path": "data/train.jsonl"})
        self.assertFalse(any(key.startswith(("core.", "head.")) for key in payload["model_state"]))
        context = load_checkpoint(path, device="cpu")
        for key, tensor in context["model"].state_dict().items():
            self.assertTrue(torch.equal(tensor, model["core"].state_dict()[key]))
        with self.assertRaisesRegex(ValueError, "inference export"):
            load_training_checkpoint(path)
        with self.assertRaisesRegex(TypeError, "CodeTransformer core"):
            export_checkpoint(path, model, config, tokenizer.to_str(), {}, 1)

    def test_resume_rejects_unrecognized_format(self):
        path = self.root / "unsupported.pt"
        atomic_save(path, {"format": "other"})
        with self.assertRaisesRegex(ValueError, "nlp_codegen_training_v1"):
            load_training_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
