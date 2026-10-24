import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from nlp_codegen.inference import load_checkpoint, load_questions, run_inference
from nlp_codegen.model import CodeTransformer


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.questions = ["  βéta third  \nKeep punctuation!?", "short", "middle raw"]
        cls.expected_codes = ["gamma", "alpha", "beta"]
        vocabulary = {
            "[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3,
            "short": 4, "raw": 5, "middle": 6, "βéta": 7,
            "third": 8, "Keep": 9, "punctuation": 10, "!?": 11,
            "alpha": 12, "beta": 13, "gamma": 14,
        }
        tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        cls.config = {
            "vocab_size": len(vocabulary), "pad_id": 0, "bos_id": 1,
            "eos_id": 2, "d_model": 16, "num_heads": 2, "d_ff": 32,
            "encoder_layers": 1, "decoder_layers": 1, "dropout": 0.0,
        }
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(29)
            model = CodeTransformer(cls.config)
            encodings = tokenizer.encode_batch(cls.questions, add_special_tokens=False)
            sources = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(encoding.ids + [2]) for encoding in encodings],
                batch_first=True, padding_value=0,
            )
            targets = torch.tensor([[14, 2], [12, 2], [13, 2]])
            decoder = torch.cat((torch.ones((3, 1), dtype=torch.long), targets[:, :-1]), dim=1)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
            for _ in range(100):
                optimizer.zero_grad()
                logits = model(sources, decoder)
                loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten())
                loss.backward()
                optimizer.step()
        cls.payload = {
            "model_config": cls.config,
            "model_state": model.state_dict(),
            "tokenizer_json": tokenizer.to_str(),
            "metadata": {"pad_id": 0, "bos_id": 1, "eos_id": 2},
            "step": 1,
            "weights": "live",
        }
        cls.checkpoint = cls.root / "model.pt"
        torch.save(cls.payload, cls.checkpoint)
        cls.test_path = cls.root / "questions.jsonl"
        with cls.test_path.open("w", encoding="utf-8") as file:
            for question in cls.questions:
                file.write(json.dumps({"question": question, "unused": "ignored"}, ensure_ascii=False) + "\n\n")

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)
        cls.temporary.cleanup()

    def test_new_checkpoint_loads_explicit_model(self):
        context = load_checkpoint(self.checkpoint, device="cpu")
        self.assertIsInstance(context["model"], CodeTransformer)
        self.assertFalse(context["model"].training)
        self.assertEqual(next(context["model"].parameters()).dtype, torch.float32)
        self.assertEqual(context["architecture"]["encoder_layers"], 1)
        self.assertEqual(context["architecture"]["norm_type"], "rmsnorm")
        self.assertEqual(context["checkpoint_step"], 1)
        self.assertEqual(context["metadata"]["eos_id"], 2)
        self.assertEqual(context["checkpoint_sha256"], hashlib.sha256(self.checkpoint.read_bytes()).hexdigest())

    def test_actual_beam_inference_preserves_raw_question_order(self):
        output = self.root / "predictions.jsonl"
        metrics = run_inference(
            self.checkpoint, self.test_path, output, device="cpu",
            batch_size=2, num_beams=2, max_new_tokens=4,
            started_at=time.perf_counter() - 1.0,
        )
        rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(rows, [
            {"question": question, "code": code}
            for question, code in zip(self.questions, self.expected_codes, strict=True)
        ])
        self.assertEqual(metrics["output_rows"], len(self.questions))
        self.assertEqual(metrics["output_sha256"], hashlib.sha256(output.read_bytes()).hexdigest())
        self.assertEqual(metrics["output_bytes"], output.stat().st_size)
        self.assertEqual(metrics["unfinished_eos"], 0)
        self.assertEqual(metrics["generated_tokens"], len(self.questions))
        self.assertGreaterEqual(metrics["total_seconds"], 1.0)
        self.assertTrue(metrics["external_started_at"])
        self.assertEqual(json.loads(output.with_suffix(".metrics.json").read_text()), metrics)

    def test_legacy_checkpoint_matches_new_checkpoint_predictions(self):
        for target in ("src.model.CodeTransformer", "model.CodeTransformer"):
            with self.subTest(target=target):
                payload = dict(self.payload)
                del payload["model_config"]
                payload["model_spec"] = {"_target_": target, "config": self.config}
                checkpoint = self.root / "legacy.pt"
                torch.save(payload, checkpoint)
                output = self.root / "legacy_predictions.jsonl"
                metrics = run_inference(
                    checkpoint, self.test_path, output, device="cpu",
                    batch_size=2, num_beams=2, max_new_tokens=4,
                )
                rows = [json.loads(line) for line in output.read_text().splitlines()]
                self.assertEqual([row["code"] for row in rows], self.expected_codes)
                self.assertEqual([row["question"] for row in rows], self.questions)
                self.assertEqual(metrics["precision"], "fp32")

    def test_missing_local_input_fails(self):
        missing = self.root / "missing.jsonl"
        with self.assertRaises(FileNotFoundError):
            run_inference(self.checkpoint, missing, self.root / "missing_output.jsonl", device="cpu")

    def test_input_is_not_overwritten(self):
        before = self.test_path.read_bytes()
        with self.assertRaises(ValueError):
            run_inference(self.checkpoint, self.test_path, self.test_path, device="cpu")
        self.assertEqual(self.test_path.read_bytes(), before)

    def test_invalid_question_type_fails(self):
        path = self.root / "invalid.jsonl"
        path.write_text('{"question": 17}\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Line 1"):
            load_questions(path)

    def test_unsupported_model_target_fails(self):
        payload = dict(self.payload)
        del payload["model_config"]
        payload["model_spec"] = {"_target_": "other.Model", "config": self.config}
        path = self.root / "unsupported.pt"
        torch.save(payload, path)
        with self.assertRaisesRegex(ValueError, "Only CodeTransformer"):
            load_checkpoint(path, device="cpu")

    def test_cpu_fp16_fails(self):
        with self.assertRaisesRegex(ValueError, "CPU inference uses fp32"):
            load_checkpoint(self.checkpoint, device="cpu", precision="fp16")

    def test_fp32_load_resets_reduced_precision_global_settings(self):
        previous = (
            torch.get_float32_matmul_precision(),
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
        )
        try:
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            load_checkpoint(self.checkpoint, device="cpu", precision="fp32")
            self.assertEqual(torch.get_float32_matmul_precision(), "highest")
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            self.assertFalse(torch.backends.cudnn.allow_tf32)
        finally:
            torch.set_float32_matmul_precision(previous[0])
            torch.backends.cuda.matmul.allow_tf32 = previous[1]
            torch.backends.cudnn.allow_tf32 = previous[2]


if __name__ == "__main__":
    unittest.main()
