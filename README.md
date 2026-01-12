# StarFlow: Generating Structured Workflow Outputs From Sketch Images

## Concept Introduction

StarFlow is based on StarVLM, a framework for training and evaluating vision-language models. StarVLM consists of three categories of components: models, datasets, and pipelines.

### Models

Models are divided into local models and API models:

- Local models are encapsulated as sub-classes of [`VLLocalModel`](starvlm/model/base.py), and their inputs are encapsulated as sub-classes of [`VLLocalInput`](starvlm/model/base.py). For example, the Qwen3-VL models (`Qwen/Qwen3-VL-8B-Instruct`, `Qwen/Qwen3-VL-32B-Instruct`, etc.) are implemented as [`QwenModel`](starvlm/model/qwen_3.py), and their inputs are implemented as [`QwenInput`](starvlm/model/qwen_3.py).

- API models are encapsulated as sub-classes of [`VLAPIModel`](starvlm/model/base.py). For example, the OpenAI-compatible API models (`openai/gpt-4o`, `anthropic/claude-3.7-sonnet`, etc.) are implemented as [`OpenAIModel`](starvlm/model/openai.py). The inputs of API models are [`VLAPIConversation`](starvlm/model/base.py) instances, each of which is a sequence of [`VLAPIMessage`](starvlm/model/base.py) instances.

Each local and API model is bound to a config file. For example, local model `Qwen/Qwen3-VL-8B-Instruct` is bound to config file [`starvlm/config/model/qwen_3_vl_8b.yaml`](starvlm/config/model/qwen_3_vl_8b.yaml), and API model `openai/gpt-4o` is bound to config file [`starvlm/config/model/gpt_4o.yaml`](starvlm/config/model/gpt_4o.yaml).

### Datasets

Datasets are encapsulated as sub-classes of [`VLDataset`](starvlm/dataset/base.py). For example, dataset `ServiceNow/BigDocs-Sketch2Flow` is implemented as [`BigDocsDataset`](starvlm/dataset/bigdocs.py). Data examples in a dataset are [`VLExample`](starvlm/dataset/base.py) instances. Each dataset is bound to a config file. For example, dataset `ServiceNow/BigDocs-Sketch2Flow` is bound to config file [`starvlm/config/dataset/bigdocs_sketch2flow.yaml`](starvlm/config/dataset/bigdocs_sketch2flow.yaml).

### Pipelines

Three pipelines are implemented:

- FSDP training pipeline: the pipeline for training a local model on one or more datasets using FSDP. It is implemented as script [`starvlm/pipeline/train_fsdp.py`](starvlm/pipeline/train_fsdp.py), and bound to two config files: [`starvlm/config/pipeline/train_fsdp_1.yaml`](starvlm/config/pipeline/train_fsdp_1.yaml) for using FSDP1, and [`starvlm/config/pipeline/train_fsdp_2.yaml`](starvlm/config/pipeline/train_fsdp_2.yaml) for using FSDP2.

- Local model evaluation pipeline: the pipeline for evaluating a local model on a dataset. It is implemented as script [`starvlm/pipeline/evaluate_local.py`](starvlm/pipeline/evaluate_local.py), and bound to config file [`starvlm/config/pipeline/evaluate_local.yaml`](starvlm/config/pipeline/evaluate_local.yaml).

- API model evaluation pipeline: the pipeline for evaluating an API model on a dataset. It is implemented as script [`starvlm/pipeline/evaluate_api.py`](starvlm/pipeline/evaluate_api.py), and bound to config file [`starvlm/config/pipeline/evaluate_api.yaml`](starvlm/config/pipeline/evaluate_api.yaml).

## Environment Setup

### Commands

1. Edit `~/.secret` (create it if missing)

```shell
export HF_TOKEN=<HF_TOKEN>
export WANDB_API_KEY=<WANDB_API_KEY>
export OPENAI_API_KEY=<OPENAI_API_KEY>
...
```

2. Edit `~/.bashrc` (create it if missing)

```shell
source ~/.secret
# home directory of venvs
export UV_HOME=<UV_HOME>
# root directory of logs
export LOGGING_ROOT=<LOGGING_ROOT>
...
```

3. Clone repository and run installers to create venvs

```shell
git clone https://github.com/ServiceNow/StarFlow.git
cd StarFlow
# default installer (for API models and most local models)
bash installer/default/install.sh
# phi35 installer (for Phi-3.5 local model)
bash installer/phi35/install.sh
# phi4 installer (for Phi-4 local model)
bash installer/phi4/install.sh
# deepseek installer (for DeepSeek-VL2 local models)
bash installer/deepseek/install.sh
# vllm installer (for vLLM-served API models)
bash installer/vllm/install.sh
```

### Notes

Before conducting experiments, activate the proper venv:

```shell
source ${UV_HOME}/starvlm_<installer_name>/bin/activate
```

## Experiment Guide

### Commands

1. Train a local model

```shell
torchrun --nproc-per-node 2 starvlm/pipeline/train_fsdp.py --pipeline_name train_fsdp_2 --model_name qwen_3_vl_8b --dataset_names bigdocs_sketch2flow
```

2. Evaluate a local model

```shell
torchrun --nproc-per-node 2 starvlm/pipeline/evaluate_local.py --pipeline_name evaluate_local --model_name qwen_3_vl_8b --dataset_name bigdocs_sketch2flow
```

3. Evaluate a large local model (e.g. `Qwen/Qwen3-VL-32B-Instruct`)

```shell
python starvlm/pipeline/evaluate_local.py --pipeline_name evaluate_local --model_name qwen_3_vl_32b --dataset_name bigdocs_sketch2flow
```

4. Evaluate an API model (e.g. `openai/gpt-4o`)

```shell
python starvlm/pipeline/evaluate_api.py --pipeline_name evaluate_api --model_name gpt_4o --dataset_name bigdocs_sketch2flow
```

5. Evaluate a vLLM-served API model (e.g. `Qwen/Qwen3-VL-8B-Instruct`)

```shell
vllm serve Qwen/Qwen3-VL-8B-Instruct --max-num-seqs 4 --tensor-parallel-size 2 --dtype bfloat16 --host 0.0.0.0 --port 8000
python starvlm/pipeline/evaluate_api.py --pipeline_name evaluate_api --model_name vllm_qwen_3_vl_8b --dataset_name bigdocs_sketch2flow
```

### Notes

Before running the above commands, properly set the values in the involved config files:

- pipeline config file: `starvlm/config/pipeline/<pipeline_name>.yaml`

- model config file: `starvlm/config/model/<model_name>.yaml`

- dataset config file: `starvlm/config/dataset/<dataset_name>.yaml`

## Citation

```BibTeX
@article{bechard2025starflow,
  title={StarFlow: Generating Structured Workflow Outputs From Sketch Images},
  author={Bechard, Patrice and Wang, Chao and Abaskohi, Amirhossein and Rodriguez, Juan and Pal, Christopher and Vazquez, David and Gella, Spandana and Rajeswar, Sai and Taslakian, Perouz},
  journal={arXiv preprint arXiv:2503.21889},
  year={2025}
}
```
