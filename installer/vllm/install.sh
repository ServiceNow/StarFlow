[ -d ${UV_HOME}/starvlm_vllm ] || uv venv ${UV_HOME}/starvlm_vllm --python 3.11 --seed
source ${UV_HOME}/starvlm_vllm/bin/activate
uv pip install vllm==0.12.0 --torch-backend=cu129
uv pip install -r installer/vllm/requirements.txt
uv pip install -e .
python -m nltk.downloader punkt_tab
