from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.state import PartialState
from accelerate.utils import ProjectConfiguration
from dataclasses import asdict
from functools import partial
from hashlib import md5
from omegaconf import DictConfig, OmegaConf
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    CheckpointImpl,
    checkpoint_wrapper,
)
from torch.distributed.fsdp.wrap import _module_wrap_policy
from typing import Optional
from starflow.dataset.base import VLDataset, VLExample
from starflow.dataset.metric.base import VLMetric
from starflow.model.base import VLModel
from starflow.utils import get_dataset_config, get_model_config, get_pipeline_config
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


def get_config() -> DictConfig:
    configs = []
    cli_config = OmegaConf.from_cli()
    dataset_name = cli_config.pop("dataset_name", None)
    if dataset_name is not None:
        configs.append(get_dataset_config(dataset_name))
    model_name = cli_config.pop("model_name", None)
    if model_name is not None:
        configs.append(get_model_config(model_name))
    pipeline_name = cli_config.pop("pipeline_name", None)
    if pipeline_name is not None:
        configs.append(get_pipeline_config(pipeline_name))
    configs.append(cli_config)
    return OmegaConf.merge(*configs)


def get_logging_dir(config: DictConfig) -> str:
    config_dict = OmegaConf.to_container(config)
    config_str = json.dumps(config_dict, sort_keys=True)
    config_hash = md5(config_str.encode()).hexdigest()
    output_dir = os.environ.get("OUTPUT_DIR", os.path.join(os.getcwd(), "output"))
    logging_dir = output_dir
    if config.get("dataset") is not None:
        logging_dir = os.path.join(logging_dir, config.dataset.name)
    if config.get("model") is not None:
        logging_dir = os.path.join(logging_dir, config.model.name)
    if config.get("pipeline") is not None:
        logging_dir = os.path.join(logging_dir, config.pipeline.name)
    logging_dir = os.path.join(logging_dir, config_hash)
    if PartialState().is_main_process:
        os.makedirs(logging_dir, exist_ok=True)
        config_file = os.path.join(logging_dir, "config.yaml")
        OmegaConf.save(config, config_file)
    PartialState().wait_for_everyone()
    return logging_dir


def get_accelerator(
    config: DictConfig, logging_dir: str, vl_model: VLModel
) -> Accelerator:
    fsdp_plugin = None
    if hasattr(config.pipeline, "fsdp_kwargs"):
        auto_wrap_policy = partial(
            _module_wrap_policy, module_classes=vl_model.get_layer_classes()
        )
        if config.pipeline.fsdp_kwargs.activation_checkpointing:
            apply_activation_checkpointing(
                vl_model,
                checkpoint_wrapper_fn=partial(
                    checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
                ),
                auto_wrap_policy=auto_wrap_policy,
            )
        if PartialState().use_distributed:
            fsdp_kwargs = OmegaConf.to_container(config.pipeline.fsdp_kwargs)
            fsdp_kwargs["auto_wrap_policy"] = auto_wrap_policy
            fsdp_kwargs["activation_checkpointing"] = False
            fsdp_plugin = FullyShardedDataParallelPlugin(**fsdp_kwargs)
    project_config = ProjectConfiguration(logging_dir=logging_dir)
    accelerator_kwargs = OmegaConf.to_container(config.pipeline.accelerator_kwargs)
    accelerator_kwargs["fsdp_plugin"] = fsdp_plugin
    accelerator_kwargs["project_config"] = project_config
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


def get_latest_checkpoint_dir(logging_dir: str) -> Optional[str]:
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


def load_checkpoint(
    config: DictConfig, accelerator: Accelerator
) -> tuple[Optional[str], int]:
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
) -> dict:
    candidates = [prediction]
    references = [[vl_example.annotations[-1]]]
    evaluation = {}
    for vl_metric in vl_metrics:
        score = vl_metric(candidates, references)
        evaluation.update(score)
    return evaluation


def save_example_with_prediction_and_evaluation(
    vl_example: VLExample, prediction: str, evaluation: dict, examples_dir: str
) -> str:
    vl_example_dict = asdict(vl_example)
    example_dir = os.path.join(examples_dir, vl_example_dict["identifier"])
    os.makedirs(example_dir, exist_ok=True)
    image_files = []
    for index, image in enumerate(vl_example_dict.pop("images")):
        image_file = os.path.join(example_dir, f"image_{index}.png")
        image.save(image_file)
        image_files.append(image_file)
    vl_example_dict["prediction"] = prediction
    vl_example_dict["evaluation"] = evaluation
    vl_example_dict["image_files"] = image_files
    example_file = os.path.join(example_dir, "example.json")
    with open(example_file, "wt") as stream:
        json.dump(vl_example_dict, stream, indent=4)
    return example_dir


def load_prediction_of_example(
    vl_example: VLExample, examples_dir: str
) -> Optional[str]:
    example_file = os.path.join(examples_dir, vl_example.identifier, "example.json")
    try:
        with open(example_file, "rt") as stream:
            vl_example_dict = json.load(stream)
        prediction = vl_example_dict["prediction"]
    except:
        prediction = None
    return prediction


def evaluate_generation_on_dataset(
    test_vl_dataset: VLDataset, vl_metrics: list[VLMetric], generation: dict
) -> dict:
    candidates = []
    references = []
    for index in range(len(test_vl_dataset)):
        vl_example = test_vl_dataset[index]
        candidates.append(generation[vl_example.identifier])
        references.append([vl_example.annotations[-1]])
    evaluation = {}
    for vl_metric in vl_metrics:
        score = vl_metric(candidates, references)
        evaluation.update(score)
    return evaluation
