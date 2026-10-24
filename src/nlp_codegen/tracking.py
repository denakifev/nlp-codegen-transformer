from collections.abc import Mapping
import importlib
import inspect
import os
from pathlib import Path
import re


def _secret_key(key):
    key = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(key)).lower()
    pieces = set(re.split(r"[^a-z0-9]+", key))
    return bool(pieces & {"token", "password", "secret", "apikey"}) or "api_key" in key


def redact(value, secrets=()):
    if isinstance(value, Mapping):
        return {
            str(key): "[REDACTED]" if _secret_key(key) else redact(item, secrets)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(item, secrets) for item in value]
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[REDACTED]")
    return value


def _constructor_arguments(constructor, values):
    parameters = inspect.signature(constructor).parameters
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return values
    required = {
        "log_code", "log_graph", "auto_param_logging", "auto_metric_logging",
        "parse_args", "auto_output_logging", "log_env_details",
        "log_git_metadata", "log_git_patch",
    }
    if not required <= parameters.keys():
        raise RuntimeError("Installed Comet SDK cannot disable automatic data collection")
    return {key: value for key, value in values.items() if key in parameters}


class CometLogger:
    def __init__(self, config, experiment_key=None, rank=0):
        if not isinstance(config, Mapping):
            raise ValueError("Comet configuration must be a mapping")
        config = config.get("comet", config)
        if not isinstance(config, Mapping):
            raise ValueError("Comet configuration must be a mapping")
        self.enabled = bool(config.get("enabled", False)) and rank == 0
        self.key = None
        self.url = None
        self._experiment = None
        self._secrets = ()
        self._log_checkpoints = bool(config.get("log_checkpoints", False))
        if not self.enabled:
            return
        offline = bool(config.get("offline", False))
        api_key = None if offline else os.environ.get("COMET_API_KEY")
        if not offline and not api_key:
            raise RuntimeError("COMET_API_KEY is required for online Comet tracking")
        self._secrets = (api_key,) if api_key else ()
        try:
            comet = importlib.import_module("comet_ml")
        except ImportError as error:
            raise RuntimeError("Install comet-ml to enable Comet tracking") from error
        constructor_name = (
            "ExistingOfflineExperiment" if offline else "ExistingExperiment"
        ) if experiment_key else ("OfflineExperiment" if offline else "Experiment")
        constructor = getattr(comet, constructor_name, None)
        if constructor is None:
            raise RuntimeError(f"Installed Comet SDK does not support {constructor_name}")
        arguments = {
            "project_name": config.get("project_name", "nlp-codegen"),
            "workspace": config.get("workspace"),
            "log_code": False,
            "log_graph": False,
            "auto_param_logging": False,
            "auto_metric_logging": False,
            "parse_args": False,
            "auto_output_logging": False,
            "log_env_details": False,
            "log_git_metadata": False,
            "log_git_patch": False,
            "log_env_gpu": False,
            "log_env_host": False,
            "log_env_cpu": False,
            "log_env_network": False,
            "log_env_disk": False,
            "auto_weight_logging": False,
            "auto_log_co2": False,
            "auto_histogram_tensorboard_logging": False,
            "auto_histogram_weight_logging": False,
            "auto_histogram_gradient_logging": False,
            "auto_histogram_activation_logging": False,
            "display_summary_level": 0,
        }
        if offline:
            arguments["offline_directory"] = str(config.get("offline_directory", "outputs/comet"))
        else:
            arguments["api_key"] = api_key
        if experiment_key:
            parameters = inspect.signature(constructor).parameters
            if "experiment_key" in parameters or any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
            ):
                arguments["experiment_key"] = experiment_key
            elif "previous_experiment" in parameters:
                arguments["previous_experiment"] = experiment_key
            else:
                raise RuntimeError("Installed Comet SDK cannot resume an existing experiment")
        self._experiment = constructor(**_constructor_arguments(constructor, arguments))
        self.key = self._experiment.get_key()
        if not self.key or (experiment_key is not None and self.key != experiment_key):
            self._experiment.end()
            self._experiment = None
            raise RuntimeError("Comet experiment key does not match the requested experiment")
        self.url = getattr(self._experiment, "url", None)

    def log_parameters(self, parameters):
        if self._experiment is not None:
            self._experiment.log_parameters(redact(parameters, self._secrets))

    def log_metrics(self, metrics, step):
        if self._experiment is not None:
            self._experiment.log_metrics(redact(metrics, self._secrets), step=step)

    def log_asset(self, path):
        if self._experiment is None:
            return
        path = Path(path)
        if path.suffix.lower() in {".pt", ".pth", ".ckpt"} and not self._log_checkpoints:
            return
        self._experiment.log_asset(str(path))

    def close(self):
        if self._experiment is not None:
            self._experiment.end()
            self._experiment = None
