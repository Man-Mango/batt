import numpy as np
import scipy
import torch
from torch.utils.data import Dataset
from pyDOE import lhs  # 导入拉丁超立方采样方法

# 定义一个继承自PyTorch Dataset类的数据集类，用于PINN模型的训练和测试
class PINN_dataset(Dataset):
    def __init__(self, filepath, test=False, N_u=100, N_f=10000):
        self.test = test  # 判断是训练集还是测试集
        data = scipy.io.loadmat(filepath)  # 从.mat文件中加载数据
        self.x = data['x']  # 从文件中加载x数据，表示空间点
        self.t = data['t']  # 从文件中加载t数据，表示时间点
        self.usol = data['usol']  # 从文件中加载解usol，对应于x和t网格点上的解

        # 创建网格，使得X[i]和T[j]对应u(X[i],T[j])=usol[i][j]
        self.X, self.T = np.meshgrid(self.x, self.t)  # 创建空间和时间的网格

        # 生成测试数据集X_u_test，将X和T展开为一维数组并水平堆叠
        self.X_u_test = np.hstack((self.X.flatten()[:, None], self.T.flatten()[:, None]))

        # 定义域的边界
        self.lb = self.X_u_test[0]  # 下边界
        self.ub = self.X_u_test[-1]  # 上边界

        # 将usol展平成列向量，列优先（Fortran风格）
        self.u_true = self.usol.flatten('F')[:, None]

        # 定义用于训练的数据点数量
        self.N_u = N_u  # 用于训练u的数据点数量
        self.N_f = N_f  # 用于训练的配点数量

        # 调用函数生成训练数据，并转换为PyTorch张量
        X_f_train_np_array, X_u_train_np_array, u_train_np_array = self.training_data()
        self.X_f_train = torch.from_numpy(X_f_train_np_array).float()  # 配点数据
        self.X_u_train = torch.from_numpy(X_u_train_np_array).float()  # 训练数据中的x和t
        self.u_train = torch.from_numpy(u_train_np_array).float()  # 训练数据中的u
        self.X_u_test_tensor = torch.from_numpy(self.X_u_test).float()  # 测试数据中的x和t
        self.u = torch.from_numpy(self.u_true).float()  # 测试数据中的真实解u
        self.f_hat = torch.zeros(self.X_f_train.shape[0], 1)  # 初始化f_hat为零向量

    def training_data(self):
        '''生成用于训练的数据'''
        # 边界条件数据（初始条件）-1 <= x <= 1 且 t = 0
        leftedge_x = np.hstack((self.X[0, :][:, None], self.T[0, :][:, None]))  # 左边界数据L1
        leftedge_u = self.usol[:, 0][:, None]  # 左边界对应的u值

        # 边界条件数据x = -1 且 0 <= t <= 1
        bottomedge_x = np.hstack((self.X[:, 0][:, None], self.T[:, 0][:, None]))  # 下边界数据L2
        bottomedge_u = self.usol[-1, :][:, None]  # 下边界对应的u值

        # 边界条件数据x = 1 且 0 <= t <= 1
        topedge_x = np.hstack((self.X[:, -1][:, None], self.T[:, 0][:, None]))  # 上边界数据L3
        topedge_u = self.usol[0, :][:, None]  # 上边界对应的u值

        # 合并所有边界数据
        all_X_u_train = np.vstack([leftedge_x, bottomedge_x, topedge_x])  # 合并后的X_u_train [456, 2]
        all_u_train = np.vstack([leftedge_u, bottomedge_u, topedge_u])  # 对应的u值 [456, 1]

        # 随机选取N_u个点作为训练数据
        idx = np.random.choice(all_X_u_train.shape[0], self.N_u, replace=False)  # 随机选取的索引
        X_u_train = all_X_u_train[idx, :]  # 选取的训练点(x, t)
        u_train = all_u_train[idx, :]  # 对应的u值

        '''配点采样'''
        # 使用拉丁超立方采样法生成N_f个配点
        X_f_train = self.lb + (self.ub - self.lb) * lhs(2, self.N_f)
        X_f_train = np.vstack((X_f_train, X_u_train))  # 将训练点添加到配点中

        return X_f_train, X_u_train, u_train  # 返回配点，训练点及其对应的u值

    def __getitem__(self, index):
        if self.test:
            return self.X_u_test_tensor, self.u  # 如果是测试集，返回测试数据
        return self.X_u_train, self.u_train, self.X_f_train, self.f_hat  # 返回训练数据

    def __len__(self):
        return 250  # 返回数据集的大小，这里设定为250