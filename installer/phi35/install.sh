conda init
source ~/.bashrc
conda create --name starflow_phi35 --yes python=3.11
conda activate starflow_phi35
pip install --upgrade pip
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install --no-build-isolation flash-attn==2.7.4.post1
pip install transformers==4.43.0
pip install --editable .
