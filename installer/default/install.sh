[ -d ${UV_HOME}/starvlm_default ] || uv venv ${UV_HOME}/starvlm_default --python 3.11 --seed
source ${UV_HOME}/starvlm_default/bin/activate
uv pip install torch==2.8.0 torchvision==0.23.0 --torch-backend=cu129
uv pip install flash-attn==2.8.3 --no-build-isolation
uv pip install -r installer/default/requirements.txt
uv pip install -e .
python -m nltk.downloader punkt_tab
