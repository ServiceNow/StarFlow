from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import gather_object, tqdm
from accelerate.utils.modeling import get_balanced_memory, load_checkpoint_in_model
from collections import ChainMap
from omegaconf import OmegaConf
from starflow.pipeline.utils import (
    evaluate_generation_on_dataset,
    evaluate_prediction_on_example,
    get_accelerator,
    get_config,
    get_logging_dir,
    load_prediction_of_example,
    save_example_with_prediction_and_evaluation,
)
from starflow.utils import get_vl_object
import json
import logging
import os
import warnings


def evaluate():
    config = get_config()
    logging_dir = get_logging_dir(config)
    vl_model = get_vl_object(
        config.model.class_path, **OmegaConf.to_container(config.model.init_kwargs)
    )
    accelerator = get_accelerator(config, logging_dir, vl_model)
    accelerator.print(f"Logging dir: {logging_dir}")
    accelerator.print(f"Dataset: {config.dataset.name}")
    accelerator.print(f"Model: {config.model.name}")
    accelerator.print(f"Pipeline: {config.pipeline.name}")
    if config.pipeline.model_checkpoint_file is not None:
        load_checkpoint_in_model(vl_model, config.pipeline.model_checkpoint_file)
        accelerator.print(
            f"Loaded model checkpoint from {config.pipeline.model_checkpoint_file}"
        )
    accelerator.print(f"Num processes: {accelerator.num_processes}")
    test_vl_dataset = get_vl_object(
        config.dataset.class_path,
        **OmegaConf.to_container(config.dataset.test.init_kwargs),
    )
    accelerator.print(f"Test dataset size: {len(test_vl_dataset)}")
    examples_dir = os.path.join(logging_dir, "examples")
    if accelerator.is_main_process:
        os.makedirs(examples_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    if accelerator.use_distributed:
        vl_model = vl_model.to(accelerator.device)
    else:
        no_split_module_classes = [
            layer_class.__name__ for layer_class in vl_model.get_layer_classes()
        ]
        max_memory = get_balanced_memory(
            vl_model, no_split_module_classes=no_split_module_classes
        )
        device_map = infer_auto_device_map(
            vl_model,
            max_memory=max_memory,
            no_split_module_classes=no_split_module_classes,
        )
        vl_model = dispatch_model(
            vl_model, device_map=device_map, main_device=accelerator.device
        )
    vl_model.eval()
    vl_metrics = [
        get_vl_object(metric.class_path, **OmegaConf.to_container(metric.init_kwargs))
        for metric in config.dataset.metrics
    ]
    with accelerator.split_between_processes(
        list(range(len(test_vl_dataset))), apply_padding=True
    ) as indices:
        generation = {}
        progress_bar = tqdm(desc="Generate", total=len(indices))
        for index in indices:
            accelerator.wait_for_everyone()
            vl_example = test_vl_dataset[index]
            prediction = load_prediction_of_example(vl_example, examples_dir)
            if prediction is None:
                with accelerator.autocast():
                    prediction = vl_model.generate(vl_example)
            generation[vl_example.identifier] = prediction
            evaluation = evaluate_prediction_on_example(
                vl_example, vl_metrics, prediction
            )
            save_example_with_prediction_and_evaluation(
                vl_example, prediction, evaluation, examples_dir
            )
            progress_bar.update()
        progress_bar.close()
    generation = [generation]
    generation = gather_object(generation)
    generation = dict(ChainMap(*generation))
    accelerator.print(f"Num predictions: {len(generation)}")
    evaluation = evaluate_generation_on_dataset(test_vl_dataset, vl_metrics, generation)
    evaluation_file = os.path.join(logging_dir, "evaluation.json")
    if accelerator.is_main_process:
        with open(evaluation_file, "wt") as stream:
            json.dump(evaluation, stream, indent=4)
    accelerator.wait_for_everyone()
    accelerator.print("Evaluation:", evaluation)
    accelerator.state.destroy_process_group()


if __name__ == "__main__":
    logging.disable(logging.WARNING)
    warnings.filterwarnings("ignore")
    evaluate()
