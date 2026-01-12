from collections import ChainMap
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm
from typing import Any
from starvlm.pipeline.utils import (
    evaluate_generation_on_dataset,
    evaluate_output_on_example,
    get_config_by_composition,
    get_logging_dir,
    load_output_on_example,
    save_example_with_output_and_evaluation,
)
from starvlm.utils import get_class_instance
import argparse
import json
import numpy as np


def generate(
    indices: list[int], config: dict[str, Any], examples_dir: Path
) -> dict[str, Any]:
    vl_api_model = get_class_instance(
        config["model"]["class_path"], **config["model"]["init_kwargs"]
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
    vl_metrics = [
        get_class_instance(metric_config["class_path"], **metric_config["init_kwargs"])
        for metric_config in config["dataset"]["test"]["metrics"]
    ]
    generation = {}
    progress_bar = tqdm(desc="generate", total=len(indices))
    for index in indices:
        vl_example = test_vl_dataset[index]
        output = load_output_on_example(vl_example, examples_dir)
        if output is None:
            output = vl_api_model.generate(vl_example)
        generation[vl_example.identifier] = output
        evaluation = evaluate_output_on_example(vl_example, vl_metrics, output)
        save_example_with_output_and_evaluation(
            vl_example, output, evaluation, examples_dir
        )
        progress_bar.update()
    progress_bar.close()
    return generation


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
    print(f"logging directory: '{logging_dir}'")
    print(f"number of processes: {config['pipeline']['num_processes']}")
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
    print(f"test dataset size: {len(test_vl_dataset)}")
    vl_metrics = [
        get_class_instance(metric_config["class_path"], **metric_config["init_kwargs"])
        for metric_config in config["dataset"]["test"]["metrics"]
    ]
    examples_dir = logging_dir.joinpath("examples")
    examples_dir.mkdir(exist_ok=True)
    with Pool(config["pipeline"]["num_processes"]) as pool:
        generation = pool.map(
            partial(generate, config=config, examples_dir=examples_dir),
            [
                indices.tolist()
                for indices in np.array_split(
                    np.arange(len(test_vl_dataset)), config["pipeline"]["num_processes"]
                )
            ],
        )
    generation = dict(ChainMap(*generation))
    print(f"number of predictions: {len(generation)}")
    evaluation = evaluate_generation_on_dataset(test_vl_dataset, vl_metrics, generation)
    evaluation_file = logging_dir.joinpath("evaluation.json")
    evaluation_file.write_text(json.dumps(evaluation, indent=4))
    print(f"evaluation: {evaluation}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline_name", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    args = parser.parse_args()
    evaluate(args.pipeline_name, args.model_name, args.dataset_name)
