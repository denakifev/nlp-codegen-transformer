from pathlib import Path
import os
import tempfile
import types
import unittest
from unittest.mock import patch

from nlp_codegen.checkpoint import atomic_save, load_training_checkpoint
from nlp_codegen.tracking import CometLogger


class FakeExperiment:
    instances = []
    kind = "Experiment"

    def __init__(self, **arguments):
        self.arguments = arguments
        self.key = arguments.get("experiment_key", arguments.get("previous_experiment", "new-experiment"))
        self.url = f"https://tracking.invalid/{self.key}"
        self.parameters = []
        self.metrics = []
        self.assets = []
        self.ended = 0
        self.instances.append(self)

    def get_key(self):
        return self.key

    def log_parameters(self, parameters):
        self.parameters.append(parameters)

    def log_metrics(self, metrics, step):
        self.metrics.append((metrics, step))

    def log_asset(self, path):
        self.assets.append(path)

    def end(self):
        self.ended += 1


def fake_sdk():
    return types.SimpleNamespace(**{
        name: type(name, (FakeExperiment,), {"kind": name})
        for name in ("Experiment", "ExistingExperiment", "OfflineExperiment", "ExistingOfflineExperiment")
    })


class TrackingTests(unittest.TestCase):
    def setUp(self):
        FakeExperiment.instances.clear()
        self.sdk = fake_sdk()
        self.environment = patch.dict(os.environ, {"COMET_API_KEY": "placeholder-api-value"}, clear=True)
        self.environment.start()
        self.importer = patch("nlp_codegen.tracking.importlib.import_module", return_value=self.sdk)
        self.import_mock = self.importer.start()

    def tearDown(self):
        self.importer.stop()
        self.environment.stop()

    def test_disabled_and_nonzero_rank_do_not_import_sdk(self):
        for config, rank in (({}, 0), ({"enabled": False}, 0), ({"enabled": True}, 1)):
            with self.subTest(config=config, rank=rank):
                logger = CometLogger(config, rank=rank)
                logger.log_parameters({"value": 1})
                logger.log_metrics({"loss": 1.0}, step=1)
                logger.log_asset("model.pt")
                logger.close()
                self.assertIsNone(logger.key)
                self.assertIsNone(logger.url)
        self.import_mock.assert_not_called()

    def test_missing_online_key_fails_before_sdk_import(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "COMET_API_KEY"):
                CometLogger({"enabled": True})
        self.import_mock.assert_not_called()

    def test_online_uses_environment_and_disables_automatic_collection(self):
        logger = CometLogger({"comet": {"enabled": True, "api_key": "ignored-config-value", "workspace": "team"}})
        experiment = FakeExperiment.instances[-1]
        self.assertEqual(experiment.kind, "Experiment")
        self.assertEqual(experiment.arguments["api_key"], "placeholder-api-value")
        self.assertEqual(experiment.arguments["project_name"], "nlp-codegen")
        self.assertEqual(experiment.arguments["workspace"], "team")
        self.assertNotIn("ignored-config-value", experiment.arguments.values())
        for key in (
            "log_code", "log_graph", "auto_param_logging", "auto_metric_logging",
            "parse_args", "auto_output_logging", "log_env_details", "log_git_metadata",
            "log_git_patch", "log_env_gpu", "log_env_host", "log_env_cpu",
            "log_env_network", "log_env_disk", "auto_weight_logging", "auto_log_co2",
        ):
            self.assertIs(experiment.arguments[key], False)
        self.assertEqual(logger.key, "new-experiment")
        self.assertEqual(logger.url, "https://tracking.invalid/new-experiment")
        logger.close()
        logger.close()
        self.assertEqual(experiment.ended, 1)

    def test_recursive_redaction_does_not_change_nonsecret_configuration(self):
        logger = CometLogger({"enabled": True})
        logger.log_parameters({
            "train": {"max_new_tokens": 32, "learning_rate": 0.01},
            "nested": [{"api_key": "a", "token": "b", "password": "c", "secret": "d", "accessToken": "e"}],
            "note": "contains placeholder-api-value here",
        })
        logger.log_metrics({"loss": 0.25, "nested": {"secret": "hidden"}}, step=3)
        experiment = FakeExperiment.instances[-1]
        recorded = experiment.parameters[0]
        self.assertEqual(recorded["train"], {"max_new_tokens": 32, "learning_rate": 0.01})
        self.assertTrue(all(value == "[REDACTED]" for value in recorded["nested"][0].values()))
        self.assertEqual(recorded["note"], "contains [REDACTED] here")
        self.assertEqual(experiment.metrics, [({"loss": 0.25, "nested": {"secret": "[REDACTED]"}}, 3)])

    def test_experiment_key_persists_through_training_checkpoint_and_resume(self):
        logger = CometLogger({"enabled": True})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.pt"
            atomic_save(path, {"format": "nlp_codegen_training_v1", "comet": {"key": logger.key, "url": logger.url}})
            logger.close()
            saved = load_training_checkpoint(path)
            resumed = CometLogger({"enabled": True}, experiment_key=saved["comet"]["key"])
        self.assertEqual(FakeExperiment.instances[-1].kind, "ExistingExperiment")
        self.assertEqual(resumed.key, saved["comet"]["key"])
        self.assertEqual(resumed.url, saved["comet"]["url"])

    def test_offline_creation_and_resume_require_no_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            logger = CometLogger({"enabled": True, "offline": True, "offline_directory": Path("outputs/offline")})
            self.assertEqual(FakeExperiment.instances[-1].kind, "OfflineExperiment")
            self.assertEqual(FakeExperiment.instances[-1].arguments["offline_directory"], "outputs/offline")
            self.assertNotIn("api_key", FakeExperiment.instances[-1].arguments)
            resumed = CometLogger({"enabled": True, "offline": True}, experiment_key=logger.key)
        self.assertEqual(FakeExperiment.instances[-1].kind, "ExistingOfflineExperiment")
        self.assertEqual(resumed.key, logger.key)

    def test_unsupported_resume_does_not_create_a_new_experiment(self):
        del self.sdk.ExistingOfflineExperiment
        with self.assertRaisesRegex(RuntimeError, "ExistingOfflineExperiment"):
            CometLogger({"enabled": True, "offline": True}, experiment_key="saved-experiment")
        self.assertEqual(FakeExperiment.instances, [])

    def test_resumed_key_mismatch_fails_and_closes_experiment(self):
        class WrongKey(FakeExperiment):
            def get_key(self):
                return "different-experiment"

        self.sdk.ExistingExperiment = WrongKey
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            CometLogger({"enabled": True}, experiment_key="saved-experiment")
        self.assertEqual(FakeExperiment.instances[-1].ended, 1)

    def test_checkpoint_assets_are_opt_in(self):
        logger = CometLogger({"enabled": True})
        logger.log_asset("checkpoints/model.pt")
        logger.log_asset("outputs/metrics.json")
        self.assertEqual(FakeExperiment.instances[-1].assets, ["outputs/metrics.json"])
        logger = CometLogger({"enabled": True, "log_checkpoints": True})
        logger.log_asset("checkpoints/model.pt")
        self.assertEqual(FakeExperiment.instances[-1].assets, ["checkpoints/model.pt"])


if __name__ == "__main__":
    unittest.main()
