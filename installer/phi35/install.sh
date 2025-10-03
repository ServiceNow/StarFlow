[ -d ~/uv_home/starflow_phi35 ] || uv venv ~/uv_home/starflow_phi35 --python 3.11 --seed
source ~/uv_home/starflow_phi35/bin/activate
pip install pip setuptools --upgrade
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install flash-attn --no-build-isolation
pip install -r installer/phi35/requirements.txt
pip install --editable .
python -m nltk.downloader punkt_tab
