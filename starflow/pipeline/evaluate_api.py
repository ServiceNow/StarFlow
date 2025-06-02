from omegaconf import OmegaConf
from tqdm import tqdm
from starflow.pipeline.utils import (
    evaluate_generation_on_dataset,
    evaluate_prediction_on_example,
    get_config,
    get_logging_dir,
    get_vl_object,
    load_prediction_of_example,
    save_example_with_prediction_and_evaluation,
)
import json
import logging
import os
import warnings


def evaluate():
    config = get_config()
    logging_dir = get_logging_dir(config)
    vl_api_model = get_vl_object(
        config.model.class_path, **OmegaConf.to_container(config.model.init_kwargs)
    )
    print(f"Logging dir: {logging_dir}")
    print(f"Dataset: {config.dataset.name}")
    print(f"Model: {config.model.name}")
    print(f"Pipeline: {config.pipeline.name}")
    print(f"Model chunk size: {config.model.chunk_size}")
    test_vl_dataset = get_vl_object(
        config.dataset.class_path,
        **OmegaConf.to_container(config.dataset.test.init_kwargs),
    )
    print(f"Test dataset size: {len(test_vl_dataset)}")
    examples_dir = os.path.join(logging_dir, "examples")
    os.makedirs(examples_dir, exist_ok=True)
    vl_metrics = []
    for metric in config.dataset.test.metrics:
        vl_metric = get_vl_object(
            metric.class_path, **OmegaConf.to_container(metric.init_kwargs)
        )
        vl_metrics.append(vl_metric)
    generation = {}
    progress_bar = tqdm(desc="Generate", total=len(test_vl_dataset))
    for chunk_start in range(0, len(test_vl_dataset), config.model.chunk_size):
        chunk_end = min(chunk_start + config.model.chunk_size, len(test_vl_dataset))
        vl_examples = []
        for index in range(chunk_start, chunk_end):
            vl_example = test_vl_dataset[index]
            prediction = load_prediction_of_example(vl_example, examples_dir)
            if prediction is None:
                vl_examples.append(vl_example)
            else:
                generation.update({vl_example.identifier: prediction})
        predictions = vl_api_model.generate(
            vl_examples, **OmegaConf.to_container(config.model.generate_kwargs)
        )
        for vl_example, prediction in zip(vl_examples, predictions):
            generation.update({vl_example.identifier: prediction})
            evaluation = evaluate_prediction_on_example(
                vl_example, vl_metrics, prediction
            )
            save_example_with_prediction_and_evaluation(
                vl_example, prediction, evaluation, examples_dir
            )
        progress_bar.update(chunk_end - chunk_start)
    progress_bar.close()
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
