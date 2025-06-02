from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.state import PartialState
from accelerate.utils import ProjectConfiguration
from accelerate.utils.modeling import load_checkpoint_in_model
from dataclasses import asdict
from importlib import import_module
from omegaconf import DictConfig, OmegaConf
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    CheckpointImpl,
    checkpoint_wrapper,
)
from torch.distributed.fsdp.wrap import _module_wrap_policy
from starflow.dataset.metric.vl_metric import VLMetric
from starflow.dataset.vl_dataset import VLDataset, VLExample
from starflow.model.vl_model import VLModel
import fnmatch
import functools
import hashlib
import json
import os
import re


class LossMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.sum = 0.0
        self.average = 0.0

    def update(self, loss: float, count: int):
        self.count += count
        self.sum += loss * count
        self.average = self.sum / self.count


def get_vl_object(class_path: str, **init_kwargs):
    module_path, class_name = class_path.rsplit(".", 1)
    return getattr(import_module(module_path), class_name)(**init_kwargs)


def get_config():
    cli_config = OmegaConf.from_cli()
    dataset_config_file = cli_config.pop("dataset_config_file")
    dataset_config = OmegaConf.load(dataset_config_file)
    model_config_file = cli_config.pop("model_config_file")
    model_config = OmegaConf.load(model_config_file)
    pipeline_config_file = cli_config.pop("pipeline_config_file")
    pipeline_config = OmegaConf.load(pipeline_config_file)
    return OmegaConf.merge(dataset_config, model_config, pipeline_config, cli_config)


def get_logging_dir(config: DictConfig):
    config_dict = OmegaConf.to_container(config)
    config_str = json.dumps(config_dict, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()
    output_dir = os.environ.get("OUTPUT_DIR", os.path.join(os.getcwd(), "output"))
    logging_dir = os.path.join(
        output_dir,
        config.dataset.name,
        config.model.name,
        config.pipeline.name,
        config_hash,
    )
    if PartialState().is_main_process:
        os.makedirs(logging_dir, exist_ok=True)
        config_file = os.path.join(logging_dir, "config.yaml")
        OmegaConf.save(config, config_file)
    PartialState().wait_for_everyone()
    return logging_dir


def get_accelerator(config: DictConfig, logging_dir: str, vl_model: VLModel):
    fsdp_plugin = None
    if hasattr(config.pipeline, "fsdp_plugin_kwargs"):
        auto_wrap_policy = functools.partial(
            _module_wrap_policy, module_classes=vl_model.layer_classes
        )
        if config.pipeline.fsdp_plugin_kwargs.activation_checkpointing:
            apply_activation_checkpointing(
                vl_model,
                checkpoint_wrapper_fn=functools.partial(
                    checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
                ),
                auto_wrap_policy=auto_wrap_policy,
            )
        if PartialState().use_distributed:
            fsdp_plugin_kwargs = OmegaConf.to_container(
                config.pipeline.fsdp_plugin_kwargs
            )
            fsdp_plugin_kwargs.update(
                {
                    "auto_wrap_policy": auto_wrap_policy,
                    "activation_checkpointing": False,
                }
            )
            fsdp_plugin = FullyShardedDataParallelPlugin(**fsdp_plugin_kwargs)
    project_config = ProjectConfiguration(logging_dir=logging_dir)
    accelerator_kwargs = OmegaConf.to_container(config.pipeline.accelerator_kwargs)
    accelerator_kwargs.update(
        {"fsdp_plugin": fsdp_plugin, "project_config": project_config}
    )
    accelerator = Accelerator(**accelerator_kwargs)
    if len(accelerator.log_with) != 0:
        accelerator.init_trackers(
            project_name=f"{config.dataset.name}_{config.model.name}_{config.pipeline.name}",
            config=OmegaConf.to_container(config),
            init_kwargs={
                "wandb": {"dir": logging_dir, "name": os.path.basename(logging_dir)}
            },
        )
    return accelerator


def get_latest_checkpoint_dir(logging_dir: str):
    pattern = re.compile(r"checkpoint_(\d+)")
    latest_overall_step = 0
    latest_checkpoint = None
    for checkpoint in os.listdir(logging_dir):
        match = pattern.match(checkpoint)
        if match is not None:
            overall_step = int(match.group(1))
            if overall_step > latest_overall_step:
                latest_overall_step = overall_step
                latest_checkpoint = checkpoint
    if latest_checkpoint is None:
        latest_checkpoint_dir = None
    else:
        latest_checkpoint_dir = os.path.join(logging_dir, latest_checkpoint)
    return latest_checkpoint_dir


def load_model_checkpoint(config: DictConfig, vl_model: VLModel):
    model_checkpoint_file = None
    if config.model.source_logging_dir is not None:
        checkpoint_dir = get_latest_checkpoint_dir(config.model.source_logging_dir)
        if checkpoint_dir is not None:
            for item in os.listdir(checkpoint_dir):
                if fnmatch.fnmatch(item, "pytorch_model*.bin") or fnmatch.fnmatch(
                    item, "model*.safetensors"
                ):
                    model_checkpoint_file = os.path.join(checkpoint_dir, item)
                    break
    if model_checkpoint_file is not None:
        load_checkpoint_in_model(vl_model, model_checkpoint_file)
    return model_checkpoint_file


def load_checkpoint(config: DictConfig, accelerator: Accelerator):
    checkpoint_dir = get_latest_checkpoint_dir(accelerator.logging_dir)
    if checkpoint_dir is None:
        overall_step = 0
        if config.pipeline.source_logging_dir is not None:
            checkpoint_dir = get_latest_checkpoint_dir(
                config.pipeline.source_logging_dir
            )
            if checkpoint_dir is not None and config.pipeline.resume_overall_step:
                overall_step = int(os.path.basename(checkpoint_dir).split("_")[-1])
    else:
        overall_step = int(os.path.basename(checkpoint_dir).split("_")[-1])
    if checkpoint_dir is not None:
        accelerator.load_state(checkpoint_dir)
    return checkpoint_dir, overall_step


def save_checkpoint(accelerator: Accelerator, overall_step: int):
    checkpoint_dir = os.path.join(accelerator.logging_dir, f"checkpoint_{overall_step}")
    accelerator.save_state(checkpoint_dir)
    return checkpoint_dir


def evaluate_prediction_on_example(
    vl_example: VLExample, vl_metrics: list[VLMetric], prediction: str
):
    candidates = [prediction]
    references = [[vl_example.annotations[-1]]]
    evaluation = {}
    for vl_metric in vl_metrics:
        score = vl_metric.compute(candidates, references)
        evaluation.update(score)
    return evaluation


def save_example_with_prediction_and_evaluation(
    vl_example: VLExample, prediction: str, evaluation: dict, examples_dir: str
):
    vl_example_dict = asdict(vl_example)
    example_dir = os.path.join(examples_dir, vl_example_dict["identifier"])
    os.makedirs(example_dir, exist_ok=True)
    image_files = []
    for index, image in enumerate(vl_example_dict.pop("images")):
        image_file = os.path.join(example_dir, f"image_{index}.png")
        image.save(image_file)
        image_files.append(image_file)
    vl_example_dict.update(
        {"prediction": prediction, "evaluation": evaluation, "image_files": image_files}
    )
    example_file = os.path.join(example_dir, "example.json")
    with open(example_file, "wt") as stream:
        json.dump(vl_example_dict, stream, indent=4)
    return example_dir


def load_prediction_of_example(vl_example: VLExample, examples_dir: str):
    example_file = os.path.join(examples_dir, vl_example.identifier, "example.json")
    try:
        with open(example_file, "rt") as stream:
            vl_example_dict = json.load(stream)
        prediction = vl_example_dict["prediction"]
    except:
        prediction = None
    return prediction


def evaluate_generation_on_dataset(
    vl_dataset: VLDataset, vl_metrics: list[VLMetric], generation: dict
):
    candidates = []
    references = []
    for index in range(len(vl_dataset)):
        vl_example = vl_dataset[index]
        candidates.append(generation[vl_example.identifier])
        references.append([vl_example.annotations[-1]])
    evaluation = {}
    for vl_metric in vl_metrics:
        score = vl_metric.compute(candidates, references)
        evaluation.update(score)
    return evaluation
