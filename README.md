# batt

```bash
conda create -n batt python=3.9
conda activate batt
pip install numpy==1.26.4 # 安装 numpy
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge    # 安装 pytorch
pip install pytorch-lightning=2.2 tensorboard six scipy pyDOE matplotlib # 安装其他包
pip install optuna optuna-integration lightning

# 训练：
python train.py

# 测试：注意选择最好的 version
python test.py --ckpt lightning_logs/version_0/checkpoints/epoch=0-step=10.ckpt --hparams lightning_logs/version_0/hparams.yaml
```