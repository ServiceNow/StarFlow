[ -d ${UV_HOME}/starvlm_phi35 ] || uv venv ${UV_HOME}/starvlm_phi35 --python 3.11 --seed
source ${UV_HOME}/starvlm_phi35/bin/activate
uv pip install torch==2.8.0 torchvision==0.23.0 --torch-backend=cu129
uv pip install flash-attn==2.8.3 --no-build-isolation
uv pip install -r installer/phi35/requirements.txt
uv pip install -e .
python -m nltk.downloader punkt_tab
