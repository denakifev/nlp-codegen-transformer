import unittest

import torch

from nlp_codegen.gradients import project_gradients


class GradientProjectionTests(unittest.TestCase):
    def test_strict_orthogonality_and_input_immutability(self):
        gradients = [torch.tensor([1., 2., 3.]), torch.tensor([2., -1., 1.]), torch.tensor([1., 1., -1.])]
        original = [value.clone() for value in gradients]
        result, metrics = project_gradients(gradients, "orthogonal", torch.Generator().manual_seed(13))
        for first in range(len(result)):
            for second in range(first):
                self.assertAlmostEqual(float(torch.dot(result[first], result[second])), 0., delta=2e-6)
        self.assertLess(metrics["max_abs_cosine_after"], 1e-6)
        for before, after in zip(original, gradients):
            torch.testing.assert_close(before, after, atol=0, rtol=0)

    def test_pcgrad_removes_analytic_conflict(self):
        gradients = [torch.tensor([1., 0.]), torch.tensor([-1., 1.])]
        result, metrics = project_gradients(gradients, "pcgrad", torch.Generator().manual_seed(2))
        torch.testing.assert_close(result[0], torch.tensor([.5, .5]))
        torch.testing.assert_close(result[1], torch.tensor([0., 1.]))
        self.assertEqual(metrics["conflicts_before"], 1)
        self.assertEqual(metrics["conflicts_after"], 0)

    def test_aligned_pcgrad_and_none_preserve_values(self):
        gradients = [torch.tensor([1., 1.]), torch.tensor([2., 0.])]
        for mode in ("none", "pcgrad"):
            result, _ = project_gradients(gradients, mode, torch.Generator().manual_seed(8))
            for expected, observed in zip(gradients, result):
                torch.testing.assert_close(expected, observed, atol=0, rtol=0)
                self.assertIsNot(expected, observed)

    def test_collinear_and_zero_vectors_are_stable(self):
        gradients = [torch.tensor([1., 0.]), torch.tensor([2., 0.]), torch.zeros(2)]
        result, metrics = project_gradients(gradients, "orthogonal", torch.Generator().manual_seed(7))
        self.assertEqual(metrics["zero_outputs"], 2)
        self.assertTrue(all(bool(torch.isfinite(value).all()) for value in result))
        opposite, _ = project_gradients([torch.tensor([1., 0.]), torch.tensor([-2., 0.])], "pcgrad")
        self.assertTrue(all(bool(value.eq(0).all()) for value in opposite))

    def test_generator_restoration_and_validation(self):
        gradients = [torch.tensor([1., 2.]), torch.tensor([2., -1.]), torch.tensor([1., 1.])]
        generator = torch.Generator().manual_seed(11)
        state = generator.get_state()
        first, first_metrics = project_gradients(gradients, "orthogonal", generator)
        generator.set_state(state)
        second, second_metrics = project_gradients(gradients, "orthogonal", generator)
        self.assertEqual(first_metrics, second_metrics)
        for left, right in zip(first, second):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        with self.assertRaises(ValueError):
            project_gradients([torch.ones(2, dtype=torch.float64)])
        with self.assertRaises(ValueError):
            project_gradients([torch.tensor([float("nan")])])
        with self.assertRaises(ValueError):
            project_gradients(gradients, "unknown")

    def test_empty_tasks_and_empty_parameter_vector(self):
        result, metrics = project_gradients([], "orthogonal")
        self.assertEqual(result, [])
        self.assertEqual(metrics["tasks"], 0)
        result, metrics = project_gradients([torch.empty(0), torch.empty(0)], "orthogonal")
        self.assertEqual(metrics["zero_outputs"], 2)
        self.assertTrue(all(value.numel() == 0 for value in result))


if __name__ == "__main__":
    unittest.main()
