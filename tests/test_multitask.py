import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from nlp_codegen.model import CodeTransformer
from nlp_codegen.multitask import MultitaskModel, TaskSampler, task_loss_sums
from nlp_codegen.targets import OPS, SIZES, derive_auxiliary_labels


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return SimpleNamespace(ids=[ord(character) + 3 for character in text],
                               offsets=[(index, index + 1) for index in range(len(text))])

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(value - 3) for value in ids)


def encoded(text, tokenizer):
    return tokenizer.encode(text, add_special_tokens=False).ids + [2]


def tiny_config():
    return dict(d_model=16, num_heads=2, num_kv_heads=2, d_ff=32, vocab_size=64,
                encoder_layers=1, decoder_layers=2, dropout=0., norm_type="layernorm", norm_eps=1e-5,
                rope_theta=10000., pad_id=0, bos_id=1, eos_id=2)


def tiny_batch():
    return dict(source=torch.tensor([[3, 4, 2], [5, 2, 0]]),
                decoder=torch.tensor([[1, 6, 7, 8], [1, 9, 2, 0]]),
                target=torch.tensor([[6, 7, 8, 2], [9, 2, 0, 0]]),
                structure=torch.tensor([[0] * len(SIZES), [-100] * len(SIZES)]),
                grounding=torch.tensor([[1., -100., 0., -100.], [-100., -100., -100., -100.]]))


class TargetTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharacterTokenizer()

    def derive(self, question, code):
        return derive_auxiliary_labels(encoded(question, self.tokenizer), encoded(code, self.tokenizer), self.tokenizer)

    def test_dag_sharing_and_dead_operations(self):
        code = "def solve():\n    unused = 1 / 0\n    x = 4\n    y = x + x\n    z = y * y\n    return z\n"
        structure, _ = self.derive("Use 4", code)
        self.assertEqual(structure[OPS.index("Add")], 1)
        self.assertEqual(structure[OPS.index("Mult")], 1)
        self.assertEqual(structure[OPS.index("Div")], 0)
        self.assertEqual(structure[-3:], [1 + OPS.index("Mult"), 2, 1])

    def test_definition_uses_previous_binding(self):
        structure, _ = self.derive("Use 2 and 3", "def solve():\n    x = 2\n    x = x + 3\n    return x\n")
        self.assertEqual(structure[0], 1)
        self.assertEqual(structure[-1], 2)

    def test_decoder_numeric_offsets_and_eos(self):
        code = "def solve():\n    total = 50 * 4\n    return total\n"
        structure, grounding = self.derive("A total of 50 objects", code)
        self.assertNotEqual(structure, [-100] * len(SIZES))
        self.assertEqual(len(grounding), len(code) + 1)
        for index, character in enumerate(code):
            if character in "50":
                self.assertEqual(grounding[index], 1.)
            elif character == "4":
                self.assertEqual(grounding[index], 0.)
            else:
                self.assertEqual(grounding[index], -100.)
        self.assertEqual(grounding[-1], -100.)

    def test_unicode_offsets_and_signed_values(self):
        code = "def solve():\n    число = -3\n    return число + 2\n"
        _, grounding = self.derive("Values -3 and 2", code)
        self.assertEqual(grounding[code.index("3")], 1.)
        self.assertEqual(grounding[code.index("2")], 1.)
        self.assertEqual(grounding[code.index("-")], -100.)

    def test_unsupported_program_never_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "side_effect"
            code = f"def solve():\n    open({str(path)!r}, 'w').write('changed')\n    return 1\n"
            structure, grounding = self.derive("Use 1", code)
            self.assertEqual(structure, [-100] * len(SIZES))
            self.assertTrue(all(value == -100. for value in grounding))
            self.assertFalse(path.exists())

    def test_roundtrip_failure_masks_all_targets(self):
        tokenizer = CharacterTokenizer()
        ids = encoded("def solve():\n    return 2\n", tokenizer)
        question = encoded("Use 2", tokenizer)
        tokenizer.encode = lambda text, add_special_tokens=False: SimpleNamespace(ids=[4], offsets=[(0, 1)])
        structure, grounding = derive_auxiliary_labels(question, ids, tokenizer)
        self.assertEqual(structure, [-100] * len(SIZES))
        self.assertTrue(all(value == -100. for value in grounding))


class MultitaskTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_core_export_and_logits_remain_identical(self):
        core = CodeTransformer(tiny_config()).eval()
        state = copy.deepcopy(core.state_dict())
        wrapper = MultitaskModel(core).eval()
        batch = tiny_batch()
        expected = core(batch["source"], batch["decoder"])
        output = wrapper(batch["source"], batch["decoder"])
        torch.testing.assert_close(output["logits"], expected, rtol=0, atol=0)
        self.assertEqual(set(wrapper.core.state_dict()), set(state))
        restored = CodeTransformer(tiny_config()).eval()
        restored.load_state_dict(wrapper.core.state_dict(), strict=True)
        torch.testing.assert_close(restored(batch["source"], batch["decoder"]), expected, rtol=0, atol=0)
        self.assertFalse(any(name.startswith(("structure.", "grounding.")) for name in wrapper.core.state_dict()))

    def test_auxiliary_losses_reach_decoder_when_encoder_is_frozen(self):
        for name in ("structure", "grounding"):
            with self.subTest(task=name):
                core = CodeTransformer(tiny_config())
                for parameter_name, parameter in core.named_parameters():
                    parameter.requires_grad_(parameter_name.startswith(("decoder.", "decoder_norm.")))
                wrapper = MultitaskModel(core).eval()
                batch = tiny_batch()
                losses = task_loss_sums(wrapper(batch["source"], batch["decoder"]), batch)
                summed, count = losses[name]
                self.assertGreater(count, 0)
                (summed / count).backward()
                decoder_norm = sum(float(parameter.grad.abs().sum()) for parameter_name, parameter in core.named_parameters()
                                   if parameter_name.startswith(("decoder.", "decoder_norm.")) and parameter.grad is not None)
                self.assertGreater(decoder_norm, 0)
                self.assertTrue(all(parameter.grad is None for parameter_name, parameter in core.named_parameters()
                                    if not parameter_name.startswith(("decoder.", "decoder_norm."))))

    def test_disabled_heads_and_loss_counts(self):
        wrapper = MultitaskModel(CodeTransformer(tiny_config()), structure=False, grounding=False).eval()
        batch = tiny_batch()
        output = wrapper(batch["source"], batch["decoder"])
        self.assertIsNone(wrapper.structure)
        self.assertIsNone(wrapper.grounding)
        self.assertEqual(set(output), {"logits"})
        self.assertEqual(wrapper.active_tasks, ("code",))
        losses = task_loss_sums(output, batch)
        self.assertEqual(set(losses), {"code"})
        self.assertEqual(losses["code"][1], 6)
        wrapper = MultitaskModel(CodeTransformer(tiny_config())).eval()
        losses = task_loss_sums(wrapper(batch["source"], batch["decoder"]), batch)
        self.assertEqual({name: count for name, (_, count) in losses.items()}, dict(code=6, structure=1, grounding=2))

    def test_masked_auxiliary_batch_is_finite(self):
        wrapper = MultitaskModel(CodeTransformer(tiny_config())).eval()
        batch = tiny_batch()
        batch["structure"].fill_(-100)
        batch["grounding"].fill_(-100)
        losses = task_loss_sums(wrapper(batch["source"], batch["decoder"]), batch)
        for name in ("structure", "grounding"):
            self.assertEqual(losses[name][1], 0)
            self.assertEqual(float(losses[name][0].detach()), 0.)
            self.assertTrue(bool(torch.isfinite(losses[name][0])))

    def test_sampler_restore_and_disabled_head_exclusion(self):
        sampler = TaskSampler(enabled=("code", "grounding"), weights={"code": 1., "grounding": 2., "structure": 4.}, seed=17)
        for _ in range(4):
            self.assertEqual(sampler.sample(), ("code", "grounding"))
        sampler = TaskSampler(weights={"code": 1., "structure": .2, "grounding": .8}, seed=19)
        for _ in range(7):
            sampler.sample()
        state = sampler.state_dict()
        expected = [sampler.sample() for _ in range(30)]
        restored = TaskSampler(weights={"code": 1., "structure": .2, "grounding": .8}, seed=999)
        restored.load_state_dict(state)
        self.assertEqual([restored.sample() for _ in range(30)], expected)
        self.assertTrue(all(value[0] == "code" and len(value) == 2 for value in expected))
        self.assertEqual(restored.steps, sampler.steps)


if __name__ == "__main__":
    unittest.main()
