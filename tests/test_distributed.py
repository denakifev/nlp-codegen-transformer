import copy
import json
import multiprocessing
import os
from pathlib import Path
import queue
import socket
import tempfile
import time
import traceback
import unittest
from unittest import mock

import torch
import torch.distributed as dist

from nlp_codegen.checkpoint import capture_rng_state, restore_rng_state
from nlp_codegen.data import PreparedCorpus, ShardBatchSampler


def _capture_training(config, destination):
    from nlp_codegen import trainer
    captured = {}
    original_projection = trainer.project_gradients
    original_update = trainer.update_step
    def projection(gradients, mode, generator):
        captured["task_gradients"] = [value.detach().cpu().clone() for value in gradients]
        result, metrics = original_projection(gradients, mode=mode, generator=generator)
        captured["projected_gradients"] = [value.detach().cpu().clone() for value in result]
        captured["projection_metrics"] = metrics
        return result, metrics
    def update(model, batches, optimizer, scaler, selected, denominators, configuration, projection_rng, device):
        captured["selected"] = selected
        captured["denominators"] = denominators
        result = original_update(model, batches, optimizer, scaler, selected, denominators, configuration, projection_rng, device)
        captured["model_state"] = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        captured["optimizer"] = copy.deepcopy(optimizer.state_dict())
        return result
    with mock.patch.object(trainer, "project_gradients", side_effect=projection), mock.patch.object(trainer, "update_step", side_effect=update):
        result = trainer.run_training(config)
    captured["result"] = result
    torch.save(captured, destination)


def _distributed_worker(rank, config, port, interface, destination, errors):
    try:
        os.environ.update({
            "RANK": str(rank), "LOCAL_RANK": str(rank), "WORLD_SIZE": "2",
            "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port),
            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "GLOO_SOCKET_IFNAME": interface,
        })
        torch.set_num_threads(1)
        _capture_training(config, destination)
    except BaseException:
        errors.put((rank, traceback.format_exc()))
        raise


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "CPU distributed Gloo is unavailable")
class DistributedTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.rng = capture_rng_state("cpu")
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        restore_rng_state(self.rng)
        torch.set_num_threads(self.threads)
        self.temporary.cleanup()

    def assert_tensors_close(self, first, second, rtol=3e-4, atol=2e-7):
        self.assertEqual(set(first), set(second))
        for name in first:
            with self.subTest(tensor=name):
                torch.testing.assert_close(first[name], second[name], rtol=rtol, atol=atol)

    def test_two_rank_global_gradient_projection_matches_one_global_batch(self):
        from helpers import prepare_tiny_corpus, tiny_training_config
        data = prepare_tiny_corpus(self.root)
        config = tiny_training_config(self.root, data, dropout=0.0, max_steps=1)
        config["model"]["dropout"] = 0.0
        config["train"].update(
            max_steps=1, batch_size=4, microbatch_size=2, gradient_accumulation=1,
            device="cpu", precision="fp32", num_workers=0, pin_memory=False,
            output_dir=str(self.root / "single"), resume=None, init_checkpoint=None,
            checkpoint_every=1, log_every=1,
        )
        config["evaluation"]["every"] = 0
        config["comet"]["enabled"] = False
        config["optimizer"]["eps"] = 1e-5
        config["heads"] = {"structure": True, "grounding": True}
        config["task_sampling"].update(always_code=True, auxiliary_tasks_per_step=1)
        config["gradients"]["mode"] = "orthogonal"
        corpus = PreparedCorpus(config["data"]["output_dir"])
        self.assertGreaterEqual(sum(count > 0 for count in corpus.train.shard_counts), 3)
        from nlp_codegen.targets import derive_auxiliary_labels
        global_indices = next(iter(ShardBatchSampler(corpus.train, 4, seed=config["seed"], mode=config["data"]["sampling_mode"])))
        for index in global_indices:
            record = corpus.train[index]
            structure, grounding = derive_auxiliary_labels(record["source_ids"], record["target_ids"], corpus.tokenizer)
            self.assertTrue(all(value != -100 for value in structure))
            self.assertTrue(any(value != -100 for value in grounding))
        single_path = self.root / "single_capture.pt"
        with mock.patch.dict(os.environ, {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1"}):
            _capture_training(config, single_path)
        baseline = torch.load(single_path, map_location="cpu", weights_only=True)
        self.assertEqual(len(baseline["selected"]), 2)
        self.assertEqual(baseline["projection_metrics"]["mode"], "orthogonal")
        self.assertGreater(baseline["projection_metrics"]["projections"], 0)
        self.assertTrue(all(bool(value.norm() > 0) for value in baseline["task_gradients"]))
        distributed = copy.deepcopy(config)
        distributed["train"].update(batch_size=2, microbatch_size=1, output_dir=str(self.root / "distributed"))
        interfaces = [name for _, name in socket.if_nameindex() if name in ("lo", "lo0")]
        self.assertTrue(interfaces, "A loopback interface is required for the local Gloo test")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.bind(("127.0.0.1", 0))
            port = server.getsockname()[1]
        context = multiprocessing.get_context("spawn")
        errors = context.Queue()
        captures = [self.root / f"rank_{rank}.pt" for rank in range(2)]
        processes = [context.Process(target=_distributed_worker, args=(rank, distributed, port, interfaces[0], str(captures[rank]), errors)) for rank in range(2)]
        try:
            for process in processes:
                process.start()
            deadline = time.monotonic() + 45
            for process in processes:
                process.join(max(0, deadline - time.monotonic()))
            alive = [process for process in processes if process.is_alive()]
            messages = []
            while True:
                try:
                    messages.append(errors.get_nowait())
                except queue.Empty:
                    break
            self.assertFalse(alive, f"Gloo workers timed out: {messages}")
            self.assertEqual([process.exitcode for process in processes], [0, 0], f"Gloo workers failed: {messages}")
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                if process.pid is not None:
                    process.join(5)
            errors.close()
            errors.join_thread()
        rank_outputs = [torch.load(path, map_location="cpu", weights_only=True) for path in captures]
        for rank, output in enumerate(rank_outputs):
            self.assertEqual(output["selected"], baseline["selected"])
            self.assertEqual(output["denominators"], baseline["denominators"])
            self.assertEqual(output["projection_metrics"]["order"], baseline["projection_metrics"]["order"])
            self.assert_tensors_close(dict(enumerate(output["task_gradients"])), dict(enumerate(baseline["task_gradients"])))
            self.assert_tensors_close(dict(enumerate(output["projected_gradients"])), dict(enumerate(baseline["projected_gradients"])))
            self.assert_tensors_close(output["model_state"], baseline["model_state"], rtol=2e-5, atol=4e-6)
            self.assertEqual(output["optimizer"]["param_groups"], baseline["optimizer"]["param_groups"])
            self.assertEqual(set(output["optimizer"]["state"]), set(baseline["optimizer"]["state"]))
            for parameter, state in output["optimizer"]["state"].items():
                self.assert_tensors_close(state, baseline["optimizer"]["state"][parameter], rtol=3e-4, atol=2e-8)
            self.assertEqual(output["result"]["rank"], rank)
            self.assertEqual(output["result"]["world_size"], 2)
        self.assert_tensors_close(rank_outputs[0]["model_state"], rank_outputs[1]["model_state"], rtol=0, atol=0)
        self.assert_tensors_close(dict(enumerate(rank_outputs[0]["task_gradients"])), dict(enumerate(rank_outputs[1]["task_gradients"])), rtol=0, atol=0)
        checkpoint = torch.load(self.root / "distributed" / "last_training.pt", map_location="cpu", weights_only=False)
        self.assertEqual(checkpoint["step"], 1)
        self.assertEqual(checkpoint["optimizer_steps"], 1)
        self.assertEqual(len(checkpoint["rank_states"]), 2)
        expected_batches = [next(iter(ShardBatchSampler(corpus.train, 2, rank=rank, world_size=2, seed=config["seed"], mode=config["data"]["sampling_mode"]))) for rank in range(2)]
        self.assertFalse(set(expected_batches[0]) & set(expected_batches[1]))
        total_seen = [0] * len(corpus.train.shard_counts)
        for rank, state in enumerate(checkpoint["rank_states"]):
            sampler = state["sampler"]
            self.assertEqual(sampler["config"]["world_size"], 2)
            self.assertEqual(sampler["config"]["rank"], rank)
            self.assertEqual(sampler["consumed_batches"], 1)
            self.assertEqual(state["tasks"]["steps"], 1)
            expected_seen = [0] * len(corpus.train.shard_counts)
            for index in expected_batches[rank]:
                expected_seen[corpus.train[index]["shard_id"]] += 1
            self.assertEqual(state["shard_seen"], expected_seen)
            total_seen = [first + second for first, second in zip(total_seen, expected_seen)]
        metrics = json.loads((self.root / "distributed" / "metrics.jsonl").read_text().splitlines()[-1])
        self.assertEqual(metrics["data/total_examples"], 4)
        for shard_id, count in enumerate(total_seen):
            self.assertEqual(metrics[f"data/shard_{shard_id}_seen"], count)
        for task in ("code", "structure", "grounding"):
            self.assertEqual(metrics[f"tasks/{task}_sampled"], int(task in baseline["selected"]))


if __name__ == "__main__":
    unittest.main()
