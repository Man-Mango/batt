import os
import numpy as np
from torch import optim, nn, utils, Tensor
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import pytorch_lightning as pl
from models.sequence import Sequentialmodel
from datasets.demo import PINN_dataset

if __name__ == '__main__':
    # setup data
    dataset = PINN_dataset('Data/burgers_shock_mu_01_pi.mat')
    train_loader = utils.data.DataLoader(dataset, num_workers=8)
    # init the model
    layers = np.array([2,20,20,20,20,20,20,20,20,1]) #8 hidden layers
    model = Sequentialmodel(layers, dataset.ub, dataset.lb)

    # train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(model=model, train_dataloaders=train_loader)

