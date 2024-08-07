import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
import torch.autograd as autograd         # computation graph


class Sequentialmodel(pl.LightningModule):
    def __init__(self, layers, ub, lb):
        super().__init__()  # call __init__ from parent class
        self.layers = layers
        self.u_b = torch.from_numpy(ub).float()
        self.l_b = torch.from_numpy(lb).float()
        # 'activation function'
        self.activation = nn.Tanh()
        # 'loss function'
        self.loss_function = nn.MSELoss(reduction='mean')
        # 'Initialise neural network as a list using nn.Modulelist'
        self.linears = nn.ModuleList(
            [nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])

        # 'Xavier Normal Initialization'
        for i in range(len(layers)-1):
            # weights from a normal distribution with
            # Recommended gain value for tanh = 5/3?
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            # set biases to zero
            nn.init.zeros_(self.linears[i].bias.data)

    #'foward pass'
    def forward(self, x):
        if torch.is_tensor(x) != True:
            x = torch.from_numpy(x)


        # preprocessing input
        x = (x - self.l_b.to(x.device))/(self.u_b.to(x.device) - self.l_b.to(x.device))  # feature scaling

        # convert to float
        a = x.float()

        '''     
        Alternatively:
        
        a = self.activation(self.fc1(a))
        a = self.activation(self.fc2(a))
        a = self.activation(self.fc3(a))
        a = self.fc4(a)
        
        '''

        for i in range(len(self.layers)-2):
            z = self.linears[i](a)
            a = self.activation(z)

        a = self.linears[-1](a)

        return a

    def loss_BC(self, x, y):
        loss_u = self.loss_function(self.forward(x), y)
        return loss_u

    def loss_PDE(self, x_to_train_f, f_hat):
        nu = 0.01/np.pi
        x_1_f = x_to_train_f[:, [0]]
        x_2_f = x_to_train_f[:, [1]]
        g = x_to_train_f.clone()
        g.requires_grad = True
        u = self.forward(g)
        u_x_t = autograd.grad(u, g, torch.ones([*x_to_train_f.shape[0:-1], 1]).to(
            x_to_train_f.device), retain_graph=True, create_graph=True)[0]

        u_xx_tt = autograd.grad(u_x_t, g, torch.ones(
            x_to_train_f.shape).to(x_to_train_f.device), create_graph=True)[0]

        u_x = u_x_t[..., [0]]
        u_t = u_x_t[..., [1]]
        u_xx = u_xx_tt[..., [0]]
        f = u_t + (self.forward(g))*(u_x) - (nu)*u_xx
        loss_f = self.loss_function(f, f_hat)

        return loss_f

    def loss(self, x, y, x_to_train_f, f_hat):

        loss_u = self.loss_BC(x, y)
        loss_f = self.loss_PDE(x_to_train_f, f_hat)

        self.log("loss_u", loss_u)
        self.log("loss_f", loss_f)
        loss_val = loss_u + loss_f

        return loss_val

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        X_u_train, u_train, X_f_train, f_hat = batch
        loss = self.loss(X_u_train, u_train, X_f_train, f_hat)
        return loss
    
    def test_step(self, batch, batch_idx):
        # this is the test loop
        X_u_test_tensor, u = batch
        u_pred = self.forward(X_u_test_tensor)
        test_loss = (torch.norm((u-u_pred),2)/torch.norm(u,2)).mean()
        self.log("test_loss", test_loss)
    
    def configure_optimizers(self):
        optimizer = torch.optim.LBFGS(self.parameters(), lr=0.1, 
                              max_iter = 250, 
                              max_eval = None, 
                              tolerance_grad = 1e-05, 
                              tolerance_change = 1e-09, 
                              history_size = 100, 
                              line_search_fn = 'strong_wolfe')
        return optimizer
    
    # 'callable for optimizer'
    # def closure(self):
    #     optimizer.zero_grad()
    #     loss = self.loss(X_u_train, u_train, X_f_train)
    #     loss.backward()
    #     self.iter += 1
    #     if self.iter % 100 == 0:
    #         error_vec, _ = PINN.test()
    #         print(loss, error_vec)
    #     return loss

    # 'test neural network'
    # def test(self):
    #     u_pred = self.forward(X_u_test_tensor)
    #     # Relative L2 Norm of the error (Vector)
    #     error_vec = torch.linalg.norm((u-u_pred), 2)/torch.linalg.norm(u, 2)
    #     u_pred = u_pred.cpu().detach().numpy()
    #     u_pred = np.reshape(u_pred, (256, 100), order='F')
    #     return error_vec, u_pred
