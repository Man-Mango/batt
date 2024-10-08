import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from torch import utils
import lightning.pytorch as pl
import yaml

from datasets.demo import PINN_dataset
from models.sequence import Sequentialmodel

def solutionplot(u_pred,X_u_train,u_train, T, X, t, x, usol):
    fig, ax = plt.subplots()
    ax.axis('off')

    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])

    h = ax.imshow(u_pred, interpolation='nearest', cmap='rainbow', 
                extent=[T.min(), T.max(), X.min(), X.max()], 
                origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    
    ax.plot(X_u_train[:,1], X_u_train[:,0], 'kx', label = 'Data (%d points)' % (u_train.shape[0]), markersize = 4, clip_on = False)

    line = np.linspace(x.min(), x.max(), 2)[:,None]
    ax.plot(t[25]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax.plot(t[50]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax.plot(t[75]*np.ones((2,1)), line, 'w-', linewidth = 1)    

    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.legend(frameon=False, loc = 'best')
    ax.set_title('$u(x,t)$', fontsize = 10)
    
    ''' 
    Slices of the solution at points t = 0.25, t = 0.50 and t = 0.75
    '''
    
    ####### Row 1: u(t,x) slices ##################
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1-1/3, bottom=0, left=0.1, right=0.9, wspace=0.5)

    ax = plt.subplot(gs1[0, 0])
    ax.plot(x,usol.T[25,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,u_pred.T[25,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(x,t)$')    
    ax.set_title('$t = 0.25s$', fontsize = 10)
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])

    ax = plt.subplot(gs1[0, 1])
    ax.plot(x,usol.T[50,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,u_pred.T[50,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(x,t)$')
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])
    ax.set_title('$t = 0.50s$', fontsize = 10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)

    ax = plt.subplot(gs1[0, 2])
    ax.plot(x,usol.T[75,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,u_pred.T[75,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(x,t)$')
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])    
    ax.set_title('$t = 0.75s$', fontsize = 10)
    
    plt.savefig('Burgers.png',dpi = 500)   


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--hparams',
                           type=str, 
                           default='lightning_logs/version_4/hparams.yaml',
                           help='hparams path')

    argparser.add_argument('--ckpt',
                           type=str,
                           default='lightning_logs/version_4/checkpoints/epoch=0-step=10.ckpt',
                           help='checkpoint path')
    args = argparser.parse_args()

    # 打开并读取 YAML 文件
    with open(args.hparams, "r") as file:
        config = yaml.safe_load(file)

    layers = config['layer_dims']
    lr = config['lr']

    # setup data
    dataset = PINN_dataset('Data/burgers_shock_mu_01_pi.mat', test=True)
    test_loader = utils.data.DataLoader(dataset)
    # init the model
    model = Sequentialmodel.load_from_checkpoint(args.ckpt,
                                                lr=lr,
                                                layers=layers, 
                                                ub=dataset.ub, 
                                                lb=dataset.lb)
    # initialize the Trainer
    trainer = pl.Trainer()

    # test the model
    trainer.test(model, dataloaders=test_loader)
    
    u_pred = model.forward(dataset.X_u_test_tensor).cpu().detach().numpy()
    u_pred = np.reshape(u_pred,(256,100),order='F')
    #Solution Plot
    solutionplot(u_pred, 
                dataset.X_u_train.cpu().detach().numpy(), 
                dataset.u_train.cpu().detach().numpy(),
                dataset.T, 
                dataset.X, 
                dataset.t, 
                dataset.x, 
                dataset.usol)
