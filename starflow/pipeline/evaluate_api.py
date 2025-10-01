from collections import ChainMap
from functools import partial
from multiprocessing import Pool
from omegaconf import OmegaConf
from tqdm import tqdm
from starflow.pipeline.utils import (
    evaluate_generation_on_dataset,
    evaluate_prediction_on_example,
    get_config,
    get_logging_dir,
    load_prediction_of_example,
    save_example_with_prediction_and_evaluation,
)
from starflow.utils import get_vl_object
import json
import logging
import numpy as np
import os
import warnings


def generate(indices, config, examples_dir):
    vl_api_model = get_vl_object(
        config.model.class_path, **OmegaConf.to_container(config.model.init_kwargs)
    )
    test_vl_dataset = get_vl_object(
        config.dataset.class_path,
        **OmegaConf.to_container(config.dataset.test.init_kwargs),
    )
    vl_metrics = [
        get_vl_object(metric.class_path, **OmegaConf.to_container(metric.init_kwargs))
        for metric in config.dataset.metrics
    ]
    generation = {}
    progress_bar = tqdm(desc="Generate", total=len(indices))
    for index in indices:
        vl_example = test_vl_dataset[index]
        prediction = load_prediction_of_example(vl_example, examples_dir)
        if prediction is None:
            prediction = vl_api_model.generate(vl_example)
        generation[vl_example.identifier] = prediction
        evaluation = evaluate_prediction_on_example(vl_example, vl_metrics, prediction)
        save_example_with_prediction_and_evaluation(
            vl_example, prediction, evaluation, examples_dir
        )
        progress_bar.update()
    progress_bar.close()
    return generation


def evaluate():
    config = get_config()
    logging_dir = get_logging_dir(config)
    print(f"Logging dir: {logging_dir}")
    print(f"Dataset: {config.dataset.name}")
    print(f"Model: {config.model.name}")
    print(f"Pipeline: {config.pipeline.name}")
    print(f"Num processes: {config.pipeline.num_processes}")
    test_vl_dataset = get_vl_object(
        config.dataset.class_path,
        **OmegaConf.to_container(config.dataset.test.init_kwargs),
    )
    print(f"Test dataset size: {len(test_vl_dataset)}")
    examples_dir = os.path.join(logging_dir, "examples")
    os.makedirs(examples_dir, exist_ok=True)
    vl_metrics = [
        get_vl_object(metric.class_path, **OmegaConf.to_container(metric.init_kwargs))
        for metric in config.dataset.metrics
    ]
    with Pool(config.pipeline.num_processes) as pool:
        generation = pool.map(
            partial(generate, config=config, examples_dir=examples_dir),
            [
                indices.tolist()
                for indices in np.array_split(
                    np.arange(len(test_vl_dataset)), config.pipeline.num_processes
                )
            ],
        )
    generation = dict(ChainMap(*generation))
    print(f"Num predictions: {len(generation)}")
    evaluation = evaluate_generation_on_dataset(test_vl_dataset, vl_metrics, generation)
    evaluation_file = os.path.join(logging_dir, "evaluation.json")
    with open(evaluation_file, "wt") as stream:
        json.dump(evaluation, stream, indent=4)
    print("Evaluation:", evaluation)


if __name__ == "__main__":
    logging.disable(logging.WARNING)
    warnings.filterwarnings("ignore")
    evaluate()
