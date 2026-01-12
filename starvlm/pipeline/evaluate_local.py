from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import gather_object, tqdm
from accelerate.utils.modeling import get_balanced_memory, load_checkpoint_in_model
from collections import ChainMap
from starvlm.pipeline.utils import (
    evaluate_generation_on_dataset,
    evaluate_output_on_example,
    get_accelerator,
    get_config_by_composition,
    get_logging_dir,
    load_output_on_example,
    save_example_with_output_and_evaluation,
)
from starvlm.utils import get_class_instance
import argparse
import json


def evaluate(pipeline_name: str, model_name: str, dataset_name: str) -> None:
    config = get_config_by_composition(
        {
            "pipeline": f"pipeline/{pipeline_name}",
            "model": f"model/{model_name}",
            "dataset": f"dataset/{dataset_name}",
        }
    )
    logging_dir = get_logging_dir(
        f"{pipeline_name}/{model_name}/{dataset_name}", config
    )
    vl_local_model = get_class_instance(
        config["model"]["class_path"], **config["model"]["init_kwargs"]
    )
    accelerator = get_accelerator(config["pipeline"]["accelerator_kwargs"])
    accelerator.print(f"logging directory: '{logging_dir}'")
    accelerator.print(f"number of processes: {accelerator.num_processes}")
    if config["pipeline"]["model_checkpoint"] is not None:
        load_checkpoint_in_model(
            vl_local_model.model, checkpoint=config["pipeline"]["model_checkpoint"]
        )
        accelerator.print(
            f"loaded model checkpoint from '{config['pipeline']['model_checkpoint']}'"
        )
    if config["model"]["coordinates_adapter"] is None:
        coordinates_adapter = None
    else:
        coordinates_adapter = get_class_instance(
            config["model"]["coordinates_adapter"]["class_path"],
            **config["model"]["coordinates_adapter"]["init_kwargs"],
        )
    test_vl_dataset = get_class_instance(
        config["dataset"]["class_path"], **config["dataset"]["test"]["init_kwargs"]
    )
    test_vl_dataset.coordinates_adapter = coordinates_adapter
    accelerator.print(f"test dataset size: {len(test_vl_dataset)}")
    vl_metrics = [
        get_class_instance(metric_config["class_path"], **metric_config["init_kwargs"])
        for metric_config in config["dataset"]["test"]["metrics"]
    ]
    if accelerator.use_distributed:
        vl_local_model.model = vl_local_model.model.to(accelerator.device)
    else:
        no_split_module_classes = [
            layer_class.__name__ for layer_class in vl_local_model.get_layer_classes()
        ]
        max_memory = get_balanced_memory(
            vl_local_model.model, no_split_module_classes=no_split_module_classes
        )
        device_map = infer_auto_device_map(
            vl_local_model.model,
            max_memory=max_memory,
            no_split_module_classes=no_split_module_classes,
        )
        vl_local_model.model = dispatch_model(
            vl_local_model.model, device_map, main_device=accelerator.device
        )
    examples_dir = logging_dir.joinpath("examples")
    if accelerator.is_main_process:
        examples_dir.mkdir(exist_ok=True)
    accelerator.wait_for_everyone()
    generation = {}
    vl_local_model.model.eval()
    with accelerator.split_between_processes(
        list(range(len(test_vl_dataset))), apply_padding=True
    ) as indices:
        progress_bar = tqdm(desc="generate", total=len(indices))
        accelerator.wait_for_everyone()
        for index in indices:
            vl_example = test_vl_dataset[index]
            output = load_output_on_example(vl_example, examples_dir)
            if output is None:
                with accelerator.autocast():
                    output = vl_local_model.generate(vl_example)
            generation[vl_example.identifier] = output
            evaluation = evaluate_output_on_example(vl_example, vl_metrics, output)
            save_example_with_output_and_evaluation(
                vl_example, output, evaluation, examples_dir
            )
            progress_bar.update()
            accelerator.wait_for_everyone()
        progress_bar.close()
    generation = [generation]
    generation = gather_object(generation)
    generation = dict(ChainMap(*generation))
    accelerator.print(f"number of predictions: {len(generation)}")
    evaluation = evaluate_generation_on_dataset(test_vl_dataset, vl_metrics, generation)
    evaluation_file = logging_dir.joinpath("evaluation.json")
    if accelerator.is_main_process:
        evaluation_file.write_text(json.dumps(evaluation, indent=4))
    accelerator.wait_for_everyone()
    accelerator.print(f"evaluation: {evaluation}")
    accelerator.state.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline_name", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    args = parser.parse_args()
    evaluate(args.pipeline_name, args.model_name, args.dataset_name)
