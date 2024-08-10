# batt

```bash
python -m venv .venv   # 创建环境
.venv\Scripts\activate.bat  # cmd 激活环境
# .venv\Scripts\Activate.ps1 # powershell 激活环境
pip install numpy==1.26.4 # 安装 numpy
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118    # 安装 pytorch
pip install pytorch-lightning tensorboard six scipy pyDOE matplotlib # 安装其他包

# 训练：
python train.py

# 测试：
python test.py --ckpt lightning_logs/version_0/checkpoints/epoch=0-step=250.ckpt
```