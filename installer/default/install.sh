[ -d ~/uv_home/starflow_default ] || uv venv ~/uv_home/starflow_default --python 3.11 --seed
source ~/uv_home/starflow_default/bin/activate
pip install pip setuptools --upgrade
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install flash-attn --no-build-isolation
pip install -r installer/default/requirements.txt
pip install --editable .
python -m nltk.downloader punkt_tab
