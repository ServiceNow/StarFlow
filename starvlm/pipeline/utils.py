from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.state import PartialState
from accelerate.utils import LoggerType
from dataclasses import asdict, dataclass
from hashlib import md5
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from pathlib import Path
from typing import Any, Optional, Union
from uuid import uuid4
from starvlm.dataset.base import VLDataset, VLExample
from starvlm.dataset.metric.base import VLMetric
from starvlm.model.base import VLLocalModel
from starvlm.utils import divide_path
import json
import os
import re
import warnings


@dataclass
class LossMeter:
    sum: float = 0.0
    count: int = 0
    average: float = 0.0

    def update(self, loss: float, count: int) -> None:
        if count <= 0:
            raise ValueError(f"count must be positive, got {count}")
        self.sum += loss * count
        self.count += count
        self.average = self.sum / self.count

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0
        self.average = 0.0


def get_config_by_composition(
    composition_spec: dict[str, Union[str, list[str]]],
) -> dict[str, Any]:
    tmp_field = f"tmp_{uuid4().hex}"
    overrides = []
    for mount_path, config_path in composition_spec.items():
        mount_path = mount_path.strip()
        if not mount_path:
            raise ValueError("mount_path cannot be empty")
        if isinstance(config_path, str):
            config_path = config_path.strip()
            if not config_path:
                raise ValueError("config_path cannot be empty")
            try:
                group_path, config_name = divide_path(config_path, "/")
            except ValueError as e:
                raise ValueError(
                    f"config_path '{config_path}' cannot be divided into group_path and config_name"
                ) from e
            overrides.append(f"+{group_path}@{mount_path}={config_name}")
        elif isinstance(config_path, list):
            tmp_mount_paths = []
            for index, item in enumerate(config_path):
                item = item.strip()
                if not item:
                    raise ValueError(
                        f"item '{item}' at index '{index}' in config_path '{config_path}' cannot be empty"
                    ) from e
                try:
                    group_path, config_name = divide_path(item, "/")
                except ValueError as e:
                    raise ValueError(
                        f"item '{item}' at index '{index}' in config_path '{config_path}' cannot be divided into group_path and config_name"
                    ) from e
                tmp_mount_path = f"{tmp_field}.{mount_path}.{index}"
                overrides.append(f"+{group_path}@{tmp_mount_path}={config_name}")
                tmp_mount_paths.append(tmp_mount_path)
            mount_reference = ",".join(
                f"${{{tmp_mount_path}}}" for tmp_mount_path in tmp_mount_paths
            )
            overrides.append(f"+{mount_path}=[{mount_reference}]")
        else:
            raise TypeError(
                f"type of config_path must be str or list, got {type(config_path).__name__}"
            )
    config_root = Path(__file__).resolve().parent.parent.joinpath("config")
    if not config_root.is_dir():
        raise ValueError(f"config_root '{config_root}' is not valid directory")
    try:
        with initialize_config_dir(str(config_root), version_base=None):
            config = compose(overrides=overrides)
    except Exception as e:
        raise RuntimeError(
            f"failed to compose config from '{config_root}' with overrides '{overrides}'"
        ) from e
    try:
        config = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    except Exception as e:
        raise RuntimeError("failed to convert config to container") from e
    config.pop(tmp_field, None)
    return config


def get_logging_dir(logging_path: str, config: dict[str, Any]) -> Path:
    logging_path = logging_path.strip()
    if not logging_path:
        raise ValueError("logging_path cannot be empty")
    try:
        config_str = json.dumps(config, sort_keys=True)
    except Exception as e:
        raise RuntimeError("failed to convert config to str") from e
    config_hash = md5(config_str.encode()).hexdigest()
    logging_root = os.environ.get("LOGGING_ROOT", "").strip()
    if logging_root:
        logging_root = Path(logging_root).expanduser()
    else:
        logging_root = Path.cwd().joinpath("logging")
        warnings.warn(
            "environment variable 'LOGGING_ROOT' is unavailable or empty, fell back to './logging'",
            category=RuntimeWarning,
            stacklevel=2,
        )
    logging_dir = logging_root.joinpath(logging_path, config_hash)
    state = PartialState()
    if state.is_main_process:
        try:
            logging_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise RuntimeError(
                f"failed to create logging directory at '{logging_dir}'"
            ) from e
        config_file = logging_dir.joinpath("config.yaml")
        try:
            OmegaConf.save(OmegaConf.create(config), str(config_file))
        except Exception as e:
            raise RuntimeError(f"failed to save config to '{config_file}'") from e
    state.wait_for_everyone()
    return logging_dir


def get_accelerator(
    accelerator_kwargs: dict[str, Any],
    fsdp_kwargs: Optional[dict[str, Any]] = None,
    vl_local_model: Optional[VLLocalModel] = None,
    use_wandb: bool = False,
    pipeline_name: Optional[str] = None,
    logging_dir: Optional[Path] = None,
) -> Accelerator:
    accelerator_kwargs = accelerator_kwargs.copy()
    if fsdp_kwargs is None:
        accelerator_kwargs["fsdp_plugin"] = None
    else:
        fsdp_kwargs = fsdp_kwargs.copy()
        if vl_local_model is None:
            raise ValueError("vl_local_model cannot be None when using FSDP")
        fsdp_kwargs["auto_wrap_policy"] = "transformer_based_wrap"
        fsdp_kwargs["transformer_cls_names_to_wrap"] = [
            layer_class.__name__ for layer_class in vl_local_model.get_layer_classes()
        ]
        try:
            accelerator_kwargs["fsdp_plugin"] = FullyShardedDataParallelPlugin(
                **fsdp_kwargs
            )
        except Exception as e:
            raise RuntimeError(
                f"failed to create fsdp_plugin with fsdp_kwargs '{fsdp_kwargs}'"
            ) from e
    if use_wandb:
        accelerator_kwargs["log_with"] = [LoggerType.WANDB]
    else:
        accelerator_kwargs["log_with"] = None
    try:
        accelerator = Accelerator(**accelerator_kwargs)
    except Exception as e:
        raise RuntimeError(
            f"failed to create accelerator with accelerator_kwargs '{accelerator_kwargs}'"
        ) from e
    if use_wandb:
        if pipeline_name is None:
            raise ValueError("pipeline_name cannot be None when using Wandb")
        pipeline_name = pipeline_name.strip()
        if not pipeline_name:
            raise ValueError("pipeline_name cannot be empty")
        if logging_dir is None:
            raise ValueError("logging_dir cannot be None when using Wandb")
        if not logging_dir.is_dir():
            raise ValueError(f"logging_dir '{logging_dir}' is not valid directory")
        try:
            accelerator.init_trackers(
                pipeline_name,
                config=accelerator_kwargs,
                init_kwargs={str(LoggerType.WANDB): {"dir": str(logging_dir)}},
            )
        except Exception as e:
            raise RuntimeError("failed to start Wandb tracker") from e
    return accelerator


def load_checkpoint(
    accelerator: Accelerator,
    logging_dir: Path,
    source_logging_dir: Optional[Path] = None,
) -> tuple[Optional[Path], int]:
    def get_checkpoint_dir(logging_dir: Path) -> Optional[Path]:
        checkpoint_dir = None
        latest_step = 0
        for child in logging_dir.iterdir():
            if child.is_dir():
                match = re.compile(r"checkpoint_(\d+)").match(child.name)
                if match is not None:
                    step = int(match.group(1))
                    if step > latest_step:
                        checkpoint_dir = child
                        latest_step = step
        return checkpoint_dir

    if not logging_dir.is_dir():
        raise ValueError(f"logging_dir '{logging_dir}' is not valid directory")
    checkpoint_dir = get_checkpoint_dir(logging_dir)
    if checkpoint_dir is None and source_logging_dir is not None:
        if not source_logging_dir.is_dir():
            raise ValueError(
                f"source_logging_dir '{source_logging_dir}' is not valid directory"
            )
        checkpoint_dir = get_checkpoint_dir(source_logging_dir)
    if checkpoint_dir is None:
        step = 0
    else:
        try:
            accelerator.load_state(str(checkpoint_dir))
        except Exception as e:
            raise RuntimeError(
                f"failed to load checkpoint from '{checkpoint_dir}'"
            ) from e
        step = int(checkpoint_dir.name.split("_")[-1])
    return checkpoint_dir, step


def save_checkpoint_and_export_model(
    accelerator: Accelerator, vl_local_model: VLLocalModel, logging_dir: Path, step: int
) -> tuple[Path, Path]:
    if not logging_dir.is_dir():
        raise ValueError(f"logging_dir '{logging_dir}' is not valid directory")
    if not isinstance(step, int):
        raise TypeError(f"type of step must be int, got {type(step).__name__}")
    checkpoint_dir = logging_dir.joinpath(f"checkpoint_{step}")
    try:
        accelerator.save_state(str(checkpoint_dir))
    except Exception as e:
        raise RuntimeError(f"failed to save checkpoint to '{checkpoint_dir}'") from e
    export_dir = checkpoint_dir.joinpath("export")
    try:
        state_dict = accelerator.get_state_dict(vl_local_model.model)
    except Exception as e:
        raise RuntimeError(
            f"failed to get state dict of vl_local_model.model ({type(vl_local_model).__name__}.model)"
        ) from e
    if accelerator.is_main_process:
        try:
            unwrapped_model = accelerator.unwrap_model(vl_local_model.model)
        except Exception as e:
            raise RuntimeError(
                f"failed to unwrap vl_local_model.model ({type(vl_local_model).__name__}.model)"
            ) from e
        try:
            unwrapped_model.save_pretrained(str(export_dir), state_dict=state_dict)
        except Exception as e:
            raise RuntimeError(
                f"failed to save vl_local_model.model ({type(vl_local_model).__name__}.model) to '{export_dir}'"
            ) from e
        try:
            vl_local_model.processor.save_pretrained(str(export_dir))
        except Exception as e:
            raise RuntimeError(
                f"failed to save vl_local_model.processor ({type(vl_local_model).__name__}.processor) to '{export_dir}'"
            ) from e
    accelerator.wait_for_everyone()
    return checkpoint_dir, export_dir


def evaluate_output_on_example(
    vl_example: VLExample, vl_metrics: list[VLMetric], output: str
) -> dict[str, Any]:
    if not vl_metrics:
        raise ValueError("vl_metrics cannot be empty")
    candidates = [output]
    references = [[vl_example.annotations[-1]]]
    evaluation = {}
    for vl_metric in vl_metrics:
        score = vl_metric(candidates, references)
        evaluation.update(score)
    return evaluation


def save_example_with_output_and_evaluation(
    vl_example: VLExample, output: str, evaluation: dict[str, Any], examples_dir: Path
) -> None:
    if not examples_dir.is_dir():
        raise ValueError(f"examples_dir '{examples_dir}' is not valid directory")
    try:
        vl_example = asdict(vl_example)
    except Exception as e:
        raise RuntimeError("failed to convert vl_example to dict") from e
    example_dir = examples_dir.joinpath(vl_example["identifier"])
    try:
        example_dir.mkdir(exist_ok=True)
    except Exception as e:
        raise RuntimeError(
            f"failed to create example directory at '{example_dir}'"
        ) from e
    image_files = []
    for index, image in enumerate(vl_example.pop("images")):
        image_file = example_dir.joinpath(f"image_{index}.png")
        try:
            image.save(str(image_file))
        except Exception as e:
            raise RuntimeError(
                f"failed to save image at index '{index}' in vl_example['images'] to '{image_file}'"
            ) from e
        image_files.append(str(image_file))
    vl_example["images"] = image_files
    vl_example["output"] = output
    vl_example["evaluation"] = evaluation
    example_file = example_dir.joinpath("example.json")
    try:
        example_file.write_text(json.dumps(vl_example, indent=4))
    except Exception as e:
        raise RuntimeError(f"failed to save vl_example to '{example_file}'") from e


def load_output_on_example(vl_example: VLExample, examples_dir: Path) -> Optional[str]:
    if not examples_dir.is_dir():
        raise ValueError(f"examples_dir '{examples_dir}' is not valid directory")
    example_file = examples_dir.joinpath(vl_example.identifier, "example.json")
    try:
        vl_example = json.loads(example_file.read_text())
        output = vl_example["output"]
    except:
        output = None
    return output


def evaluate_generation_on_dataset(
    vl_dataset: VLDataset, vl_metrics: list[VLMetric], generation: dict[str, Any]
) -> dict[str, Any]:
    if not vl_metrics:
        raise ValueError("vl_metrics cannot be empty")
    candidates = []
    references = []
    for vl_example in vl_dataset:
        if vl_example.identifier not in generation:
            raise KeyError(
                f"generation does not contain output on vl_example '{vl_example.identifier}'"
            )
        candidates.append(generation[vl_example.identifier])
        references.append([vl_example.annotations[-1]])
    evaluation = {}
    for vl_metric in vl_metrics:
        score = vl_metric(candidates, references)
        evaluation.update(score)
    return evaluation
