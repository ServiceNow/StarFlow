[ -d ~/starflow_deepseek ] || uv venv ~/starflow_deepseek --python 3.11 --seed
source ~/starflow_deepseek/bin/activate
pip install pip setuptools --upgrade
pip install torch torchvision xformers --index-url https://download.pytorch.org/whl/cu128
pip install flash-attn --no-build-isolation
pip install git+https://github.com/deepseek-ai/DeepSeek-VL2.git --no-deps
pip install -r installer/deepseek/requirements.txt
pip install --editable .
python -m nltk.downloader punkt_tab
