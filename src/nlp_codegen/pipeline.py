import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

from omegaconf import OmegaConf

from nlp_codegen.config import load_config, validate_config


def load_plan(path, overrides=()):
    recipe = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(recipe, dict) or not isinstance(recipe.get("stages"), list) or not recipe["stages"]:
        raise ValueError("Pipeline recipe requires a nonempty stages list")
    base = load_config(recipe.get("base_config", "configs/online.yaml"))
    common = OmegaConf.merge(base, recipe.get("overrides", {}), OmegaConf.from_dotlist(list(overrides)))
    common = OmegaConf.to_container(common, resolve=True)
    if common["train"].get("resume"):
        raise ValueError("Pipeline manages train.resume from its stage output directories")
    output = Path(recipe.get("output_dir", "runs/pipeline"))
    seed = int(recipe.get("seed", common["seed"]))
    stages, names = [], set()
    for index, stage in enumerate(recipe["stages"]):
        if not isinstance(stage, dict):
            raise ValueError("Each pipeline stage must be a mapping")
        name, steps = stage.get("name"), stage.get("steps")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name) or name in names:
            raise ValueError("Stage names must be unique directory names")
        if type(steps) is not int or steps < 1:
            raise ValueError("Stage steps must be positive integers")
        handoff = stage.get("handoff", "model")
        if handoff not in {"best", "model"}:
            raise ValueError("Stage handoff must be best or model")
        config = OmegaConf.to_container(OmegaConf.merge(common, stage.get("overrides", {})), resolve=True)
        config["seed"] = seed + index
        config["train"].update(output_dir=str(output / name), max_steps=steps, resume=None)
        config["train"]["init_checkpoint"] = common["train"].get("init_checkpoint") if index == 0 else str(
            Path(stages[-1]["config"]["train"]["output_dir"]) / ("best.pt" if stages[-1]["handoff"] == "best" else "model.pt")
        )
        validate_config(config)
        stages.append({"name": name, "handoff": handoff, "config": config})
        names.add(name)
    return {"recipe": recipe, "output_dir": str(output), "stages": stages}


def describe_plan(plan):
    stages = [{
        "name": stage["name"], "steps": stage["config"]["train"]["max_steps"],
        "max_total_seconds": stage["config"]["train"].get("max_total_seconds"),
        "seed": stage["config"]["seed"], "scope": stage["config"]["train"]["scope"],
        "heads": {key: stage["config"]["heads"][key] for key in ("structure", "grounding")},
        "task_sampling": {key: stage["config"]["task_sampling"][key]
                          for key in ("always_code", "auxiliary_tasks_per_step", "weights", "drop_probabilities")
                          if key in stage["config"]["task_sampling"]},
        "learning_rate": stage["config"]["optimizer"]["lr"],
        "data": {"mode": stage["config"]["data"].get("mode", "prepared"),
                 "source": stage["config"]["data"]["source"]["kind"],
                 "output_dir": stage["config"]["data"]["output_dir"]},
        "handoff": stage["handoff"],
    } for stage in plan["stages"]]
    limits = [stage["max_total_seconds"] for stage in stages]
    return {"output_dir": plan["output_dir"], "stages": stages,
            "total_max_steps": sum(stage["steps"] for stage in stages),
            "total_max_seconds": sum(limits) if all(value is not None for value in limits) else None}


def _atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as file:
            temporary = Path(file.name)
            json.dump(payload, file, ensure_ascii=False, indent=2, allow_nan=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _rank_zero(function, rank, distributed):
    failure = [None]
    if rank == 0:
        try:
            function()
        except Exception as error:
            failure[0] = f"{type(error).__name__}: {error}"
    if distributed.is_initialized():
        distributed.broadcast_object_list(failure, src=0)
    if failure[0] is not None:
        raise RuntimeError(failure[0])


def _publish_export(source, output):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=output.parent, delete=False) as file:
            temporary = Path(file.name)
            with Path(source).open("rb") as origin:
                shutil.copyfileobj(origin, file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _handoff(stage):
    directory = Path(stage["config"]["train"]["output_dir"])
    path = directory / "best.pt" if stage["handoff"] == "best" and (directory / "best.pt").is_file() else directory / "model.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Completed stage has no inference export: {path}")
    return str(path)


def run_pipeline(path, overrides=(), max_updates=None, plan_only=False):
    if max_updates is not None and (type(max_updates) is not int or max_updates < 1):
        raise ValueError("max_updates must be a positive integer")
    overrides = tuple(overrides)
    plan = load_plan(path, overrides)
    if plan_only:
        return describe_plan(plan)
    import torch.distributed as distributed
    from nlp_codegen.checkpoint import load_training_checkpoint
    from nlp_codegen.prepare import prepare_data
    from nlp_codegen.trainer import choose_device, run_training

    rank = distributed.get_rank() if distributed.is_initialized() else int(os.getenv("RANK", "0"))
    world_size = distributed.get_world_size() if distributed.is_initialized() else int(os.getenv("WORLD_SIZE", "1"))
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("Invalid pipeline rank or world size")
    initialized_here = world_size > 1 and not distributed.is_initialized()
    if initialized_here:
        device = choose_device(plan["stages"][0]["config"]["train"]["device"], int(os.getenv("LOCAL_RANK", "0")))
        distributed.init_process_group("nccl" if device.type == "cuda" else "gloo")
    output = Path(plan["output_dir"])
    identity_stages = copy.deepcopy(plan["stages"])
    for stage in identity_stages:
        stage["config"].pop("comet", None)
    identity = hashlib.sha256(json.dumps(
        {"stages": identity_stages, "output_dir": plan["output_dir"],
         "prepare_data": plan["recipe"].get("prepare_data", True), "world_size": world_size},
        sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()
    summary = {"format": "nlp_codegen_pipeline_v1", "identity": identity,
               "status": "paused", "session_updates": 0, "stages": []}
    try:
        def manifest():
            path = output / "pipeline.json"
            if path.exists():
                existing = json.loads(path.read_text(encoding="utf-8"))
                if existing.get("format") != summary["format"] or existing.get("identity") != identity:
                    raise ValueError("Pipeline recipe, training config, overrides or world size changed")
            else:
                if output.exists() and any(output.iterdir()):
                    raise ValueError("Pipeline output is not empty and has no pipeline.json")
                _atomic_json(path, {"format": summary["format"], "identity": identity,
                                    "world_size": world_size, "plan": describe_plan(plan)})
        _rank_zero(manifest, rank, distributed)
        remaining = max_updates
        for index, stage in enumerate(plan["stages"]):
            config = copy.deepcopy(stage["config"])
            train = config["train"]
            checkpoint_path = Path(train["output_dir"]) / "last_training.pt"
            before = 0
            before_elapsed = 0.0
            if checkpoint_path.is_file():
                checkpoint = load_training_checkpoint(checkpoint_path)
                before = int(checkpoint["step"])
                before_elapsed = float(checkpoint.get("elapsed_seconds", 0.0))
                del checkpoint
                train.update(resume=str(checkpoint_path), init_checkpoint=None)
            elif index:
                train["init_checkpoint"] = _handoff(plan["stages"][index - 1])
            time_limit = train.get("max_total_seconds")
            before_complete = before >= train["max_steps"] or (
                time_limit is not None and before_elapsed >= float(time_limit)
            )
            if remaining == 0 and not before_complete:
                break
            if config["data"].get("mode", "prepared") == "prepared" and plan["recipe"].get("prepare_data", True):
                def prepare():
                    if not (Path(config["data"]["output_dir"]) / "manifest.json").is_file():
                        prepare_data(config["data"])
                _rank_zero(prepare, rank, distributed)
            result = run_training(config, max_updates=remaining if remaining else None)
            if distributed.is_initialized():
                distributed.barrier()
            after = int(result["step"])
            if after < before:
                raise RuntimeError("Stage training step moved backwards")
            used = after - before
            summary["session_updates"] += used
            if remaining is not None:
                remaining -= used
                if remaining < 0:
                    raise RuntimeError("Training exceeded the pipeline session budget")
            complete = bool(result.get("complete", after >= train["max_steps"]))
            completion_reason = result.get("completion_reason", "steps" if complete else None)
            summary["stages"].append({"name": stage["name"], "step": after,
                                      "target_steps": train["max_steps"], "complete": complete,
                                      "elapsed_seconds": float(result.get("elapsed_seconds", before_elapsed)),
                                      "completion_reason": completion_reason,
                                      "handoff": _handoff(stage) if complete else None})
            if not complete:
                break
        if len(summary["stages"]) == len(plan["stages"]) and all(stage["complete"] for stage in summary["stages"]):
            summary["status"] = "complete"
            summary["model"] = str(output / "model.pt")
            _rank_zero(lambda: _publish_export(summary["stages"][-1]["handoff"], output / "model.pt"), rank, distributed)
        _rank_zero(lambda: _atomic_json(output / "pipeline_summary.json", summary), rank, distributed)
        return summary
    finally:
        if initialized_here and distributed.is_initialized():
            distributed.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "--recipe", default="configs/pipeline.yaml")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    if not args.plan:
        from dotenv import load_dotenv
        load_dotenv(Path.cwd() / ".env", override=False)
    result = run_pipeline(args.config, args.overrides, args.max_updates, args.plan)
    if int(os.getenv("RANK", "0")) == 0:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
