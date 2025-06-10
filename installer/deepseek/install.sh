conda init
source ~/.bashrc
conda create --name starflow_deepseek --yes python=3.11
conda activate starflow_deepseek
pip install --upgrade pip
pip install torch==2.6.0 torchvision==0.21.0 xformers==0.0.29.post3 --index-url https://download.pytorch.org/whl/cu124
pip install --no-build-isolation flash-attn==2.7.4.post1
pip install attrdict==2.0.1 timm==1.0.15 transformers==4.38.2
git clone https://github.com/deepseek-ai/DeepSeek-VL2.git ~/DeepSeek-VL2
pip install --no-deps --editable ~/DeepSeek-VL2
pip install --editable .
python -m nltk.downloader punkt_tab
