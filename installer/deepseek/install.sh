[ -d ${UV_HOME}/starvlm_deepseek ] || uv venv ${UV_HOME}/starvlm_deepseek --python 3.11 --seed
source ${UV_HOME}/starvlm_deepseek/bin/activate
uv pip install torch==2.7.1 torchvision==0.22.1 xformers==0.0.31.post1 --torch-backend=cu128
uv pip install flash-attn==2.8.0.post2 --no-build-isolation
uv pip install --no-deps git+https://github.com/deepseek-ai/DeepSeek-VL2.git
uv pip install -r installer/deepseek/requirements.txt
uv pip install -e .
python -m nltk.downloader punkt_tab
