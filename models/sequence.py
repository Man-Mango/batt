import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
import torch.autograd as autograd  # 导入自动求导工具，用于计算梯度

# 定义一个继承自pytorch_lightning的LightningModule的类Sequentialmodel
class Sequentialmodel(pl.LightningModule):
    def __init__(self, layers, ub, lb):
        super().__init__()  # 调用父类的__init__方法，进行初始化
        self.layers = layers  # 保存神经网络的层结构
        self.u_b = torch.from_numpy(ub).float()  # 将上界ub从numpy数组转为torch张量，并转换为浮点型
        self.l_b = torch.from_numpy(lb).float()  # 将下界lb从numpy数组转为torch张量，并转换为浮点型
        self.activation = nn.Tanh()  # 定义激活函数为Tanh
        self.loss_function = nn.MSELoss(reduction='mean')  # 定义损失函数为均方误差损失函数
        # 使用nn.ModuleList初始化神经网络层，创建一个包含线性层的列表
        self.linears = nn.ModuleList(
            [nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])

        # 使用Xavier Normal初始化方法初始化权重
        for i in range(len(layers)-1):
            # 使用Xavier Normal初始化权重，并设置增益系数为1.0
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linears[i].bias.data)  # 将偏置项初始化为零

    # 定义前向传播方法
    def forward(self, x):
        if not torch.is_tensor(x):  # 如果输入不是torch张量
            x = torch.from_numpy(x)  # 将输入从numpy数组转换为torch张量

        # 对输入进行预处理，进行特征缩放
        x = (x - self.l_b.to(x.device)) / (self.u_b.to(x.device) - self.l_b.to(x.device))

        a = x.float()  # 将输入转换为浮点型

        # 逐层进行前向传播
        for i in range(len(self.layers) - 2):
            z = self.linears[i](a)  # 进行线性变换
            a = self.activation(z)  # 通过激活函数

        a = self.linears[-1](a)  # 最后一层不经过激活函数，直接输出

        return a  # 返回最终输出

    # 定义边界条件损失函数
    def loss_BC(self, x, y):
        loss_u = self.loss_function(self.forward(x), y)  # 计算模型预测值与真实值之间的损失
        return loss_u

    # 定义偏微分方程(PDE)损失函数
    def loss_PDE(self, x_to_train_f, f_hat):
        nu = 0.01 / np.pi  # 定义粘性系数
        x_1_f = x_to_train_f[:, [0]]  # 提取第一个输入变量
        x_2_f = x_to_train_f[:, [1]]  # 提取第二个输入变量
        g = x_to_train_f.clone()  # 克隆输入数据
        g.requires_grad = True  # 设置需要计算梯度
        u = self.forward(g)  # 通过模型进行前向传播，得到预测值
        u_x_t = autograd.grad(u, g, torch.ones([*x_to_train_f.shape[0:-1], 1]).to(
            x_to_train_f.device), retain_graph=True, create_graph=True)[0]  # 计算u对g的梯度

        u_xx_tt = autograd.grad(u_x_t, g, torch.ones(
            x_to_train_f.shape).to(x_to_train_f.device), create_graph=True)[0]  # 计算二阶导数

        u_x = u_x_t[..., [0]]  # 提取对x的导数
        u_t = u_x_t[..., [1]]  # 提取对t的导数
        u_xx = u_xx_tt[..., [0]]  # 提取对x的二阶导数
        f = u_t + (self.forward(g)) * u_x - (nu) * u_xx  # 定义PDE方程
        loss_f = self.loss_function(f, f_hat)  # 计算PDE方程的损失

        return loss_f

    # 定义总损失函数，结合边界条件损失和PDE损失
    def loss(self, x, y, x_to_train_f, f_hat):
        loss_u = self.loss_BC(x, y)  # 计算边界条件损失
        loss_f = self.loss_PDE(x_to_train_f, f_hat)  # 计算PDE损失

        self.log("loss_u", loss_u)  # 记录边界条件损失
        self.log("loss_f", loss_f)  # 记录PDE损失
        loss_val = loss_u + loss_f  # 总损失为两者之和

        return loss_val

    # 定义训练步骤
    def training_step(self, batch, batch_idx):
        X_u_train, u_train, X_f_train, f_hat = batch  # 解包batch数据
        loss = self.loss(X_u_train, u_train, X_f_train, f_hat)  # 计算总损失
        return loss  # 返回损失

    # 定义测试步骤
    def test_step(self, batch, batch_idx):
        X_u_test_tensor, u = batch  # 解包测试数据
        u_pred = self.forward(X_u_test_tensor)  # 进行前向传播，得到预测值
        test_loss = (torch.norm((u - u_pred), 2) / torch.norm(u, 2)).mean()  # 计算测试损失
        self.log("test_loss", test_loss)  # 记录测试损失

    # 定义优化器配置
    def configure_optimizers(self):
        optimizer = torch.optim.LBFGS(self.parameters(), lr=0.1, 
                              max_iter=250, 
                              tolerance_grad=1e-05, 
                              tolerance_change=1e-09, 
                              history_size=100, 
                              line_search_fn='strong_wolfe')  # 使用LBFGS优化器
        return optimizer  # 返回优化器