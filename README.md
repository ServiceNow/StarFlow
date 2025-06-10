# StarFlow: Generating Structured Workflow Outputs From Sketch Images

## Setup

1. Clone the repository

```shell
git clone https://github.com/ServiceNow/StarFlow.git
cd StarFlow
```

2. Edit `~/.secret` (create a new file if it does not exist)

```shell
export HF_TOKEN=<HF_TOKEN>
export WANDB_API_KEY=<WANDB_API_KEY>
export OPENROUTER_API_KEY=<OPENROUTER_API_KEY>
export OUTPUT_DIR=<OUTPUT_DIR>
...
```

3. Edit `~/.bashrc` (create a new file if it does not exist)

```shell
source ~/.secret
...
```

4. Install packages

```shell
# for Llama, Qwen, Pixtral, and API models
bash installer/default/install.sh
# for Phi-3.5 model
bash installer/phi35/install.sh
# for Phi-4 model
bash installer/phi4/install.sh
# for DeepSeek models
bash installer/deepseek/install.sh
```

## Training and Evaluation Guide

### Commands

1. Training

```shell
torchrun \
    --nproc-per-node 2 \
    starflow/pipeline/train.py \
    dataset_config_file=starflow/config/dataset/bigdocs_sketch2flow.yaml \
    model_config_file=starflow/config/model/llama_32_11b.yaml \
    pipeline_config_file=starflow/config/pipeline/train.yaml
```

2. Evaluation

```shell
torchrun \
    --nproc-per-node 2 \
    starflow/pipeline/evaluate.py \
    dataset_config_file=starflow/config/dataset/bigdocs_sketch2flow.yaml \
    model_config_file=starflow/config/model/llama_32_11b.yaml \
    pipeline_config_file=starflow/config/pipeline/evaluate.yaml
```

3. Evaluation for very large models (e.g. Llama-3.2-90B-Vision-Instruct)

```shell
python \
    starflow/pipeline/evaluate.py \
    dataset_config_file=starflow/config/dataset/bigdocs_sketch2flow.yaml \
    model_config_file=starflow/config/model/llama_32_90b.yaml \
    pipeline_config_file=starflow/config/pipeline/evaluate.yaml
```

4. Evaluation for API models (e.g. GPT-4o)

```shell
python \
    starflow/pipeline/evaluate_api.py \
    dataset_config_file=starflow/config/dataset/bigdocs_sketch2flow.yaml \
    model_config_file=starflow/config/model/gpt_4o.yaml \
    pipeline_config_file=starflow/config/pipeline/evaluate_api.yaml
```

### Notes

1. Other models can be trained and evaluated by setting their config file path as the value of `model_config_file`.

2. The values in the involved config files should be set properly before running training and evaluation.

## Concept Introduction

StarFlow consists of four types of components: datasets, metrics, models, and pipelines.

### Datasets

Datasets provide vision-language data for training and evaluation. They are encapsulated as sub-classes of [`VLDataset`](starflow/dataset/vl_dataset.py). For example, the `BigDocs` datasets are encapsulated as [`BigDocsDataset`](starflow/dataset/vl_datasets/bigdocs.py).

When instantiating a dataset, its data examples are first loaded from either Hugging Face or local storage, and then encapsulated as [`VLExample`](starflow/dataset/vl_dataset.py).

Each dataset comes with a config file, which specifies the settings for instantiating and using the dataset. For example, the config file for `ServiceNow/BigDocs-Sketch2Flow` is [`starflow/config/dataset/bigdocs_sketch2flow.yaml`](starflow/config/dataset/bigdocs_sketch2flow.yaml).

### Metrics

Metrics compute performance numbers of models on datasets. They are encapsulated as sub-classes of [`VLMetric`](starflow/dataset/metric/vl_metric.py). For example, the `Flow Similarity` metric is encapsulated as [`FlowSimilarityMetric`](starflow/dataset/metric/vl_metrics/flow_similarity.py).

When using a metric to evaluate a model on a dataset, the metric compares the outputs of the model with the corresponding ground truths in the dataset and thereby obtains the performance numbers.

Each metric is applied to one or more datasets, and the settings for instantiating and using the metric are specifed in the config files of the target datasets. For example, the settings for `FlowSimilarityMetric` are specified in the config file of `ServiceNow/BigDocs-Sketch2Flow` ([`starflow/config/dataset/bigdocs_sketch2flow.yaml`](starflow/config/dataset/bigdocs_sketch2flow.yaml)).

### Models

Models generate textual outputs given vision-language inputs from datasets. They are encapsulated as sub-classes of [`VLModel`](starflow/model/vl_model.py), and their inputs are encapsulated as sub-classes of [`VLInput`](starflow/model/vl_model.py). For example, the `Llama-3.2-Vision-Instruct` models are encapsulated as [`LlamaModel`](starflow/model/vl_models/llama32.py), and their inputs are encapsulated as [`LlamaInput`](starflow/model/vl_models/llama32.py).

When training a model, a cross-entropy loss is obtained from the forward pass of the model, which is then optimized in the backward pass through gradient decent. When evaluating a model, the textual outputs of the model are processed by the applied metrics to compute performance numbers.

Each model comes with a config file, which specifies the settings for instantiating and using the model. For example, the config file for `Llama-3.2-11B-Vision-Instruct` is [`starflow/config/model/llama_32_11b.yaml`](starflow/config/model/llama_32_11b.yaml).

A special category of models is API models, which can only be used through API calls. They are encapsulated as sub-classes of [`VLAPIModel`](starflow/model/vl_api_model.py), and each of them comes with a config file. For example, the OpenRouter-routed `GPT-4o` model is encapsulated as [`OpenRouterAPIModel`](starflow/model/vl_api_models/open_router.py), and its config file is [`starflow/config/model/gpt_4o.yaml`](starflow/config/model/gpt_4o.yaml). API models cannot be trained, but can still be evaluated.

### Pipelines

Pipelines are Python scripts that execute complete processes with datasets, metrics, and models. There are three pipelines, each of with comes with a config file:

- Training pipeline: the pipeline for training a model on a dataset. It is implemented as [`starflow/pipeline/train.py`](starflow/pipeline/train.py), and its config file is [`starflow/config/pipeline/train.yaml`](starflow/config/pipeline/train.yaml).

- Evaluation pipeline: the pipeline for evaluating a model on a dataset with the applied metrics. It is implemented as [`starflow/pipeline/evaluate.py`](starflow/pipeline/evaluate.py), and its config file is [`starflow/config/pipeline/evaluate.yaml`](starflow/config/pipeline/evaluate.yaml).

- API model evaluation pipeline: the pipeline for evaluating an API model on a dataset with the applied metrics. It is implemented as [`starflow/pipeline/evaluate_api.py`](starflow/pipeline/evaluate_api.py), and its config file is [`starflow/config/pipeline/evaluate_api.yaml`](starflow/config/pipeline/evaluate_api.yaml).
