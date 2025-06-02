# StarFlow: Generating Structured Workflow Outputs From Sketch Images

## Setup

1. Clone the repository

```shell
git clone https://github.com/ServiceNow/StarFlow.git
cd StarFlow
```

2. Edit `~/.secret`

```shell
export HF_TOKEN=<HF_TOKEN>
export WANDB_API_KEY=<WANDB_API_KEY>
export OPENROUTER_API_KEY=<OPENROUTER_API_KEY>
export OUTPUT_DIR=<OUTPUT_DIR>
...
```

3. Edit `~/.bashrc`

```shell
source ~/.secret
...
```

4. Install packages

```shell
conda init
source ~/.bashrc
conda create --name starflow --yes python=3.11
conda activate starflow
pip install --upgrade pip
pip install --editable .
pip install --no-build-isolation flash-attn==2.7.4.post1
python -m nltk.downloader punkt_tab
```

## Training and Evaluation Guide

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

3. Evaluation for Very Large Models (e.g. Llama-3.2-90B-Vision-Instruct)

```shell
python \
    starflow/pipeline/evaluate.py \
    dataset_config_file=starflow/config/dataset/bigdocs_sketch2flow.yaml \
    model_config_file=starflow/config/model/llama_32_90b.yaml \
    pipeline_config_file=starflow/config/pipeline/evaluate.yaml
```

4. Evaluation for API Models (e.g. GPT-4o)

```shell
python \
    starflow/pipeline/evaluate_api.py \
    dataset_config_file=starflow/config/dataset/bigdocs_sketch2flow.yaml \
    model_config_file=starflow/config/model/gpt_4o.yaml \
    pipeline_config_file=starflow/config/pipeline/evaluate_api.yaml
```
