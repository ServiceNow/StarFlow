[ -d ~/uv_home/starflow_vllm ] || uv venv ~/uv_home/starflow_vllm --python 3.11 --seed
source ~/uv_home/starflow_vllm/bin/activate
pip install pip setuptools --upgrade
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu128
pip install flash-attn --no-build-isolation
pip install -r installer/vllm/requirements.txt
pip install --editable .
python -m nltk.downloader punkt_tab
