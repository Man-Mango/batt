import argparse
import os
import numpy as np
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from torch import optim, nn, utils, Tensor
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning.pytorch as pl
from models.sequence import Sequentialmodel
from datasets.demo import PINN_DataModule

def objective(trial: optuna.trial.Trial) -> float:
    pl.seed_everything(42, workers=True) # 固定所有随机数种子
    # We optimize the number of layers, hidden units in each layer.
    lr = n_layers = trial.suggest_float("lr", 1e-6, 1e-2, log=True)
    n_layers = trial.suggest_int("n_middle_layers", 1, 16)
    layers = [
        trial.suggest_int("n_units_l{}".format(i), 4, 32, log=True) for i in range(n_layers)
    ]
    layers = [2] + layers + [1]

    # setup data
    datamodule = PINN_DataModule('Data/burgers_shock_mu_01_pi.mat')
    # init the model
    model = Sequentialmodel(lr=lr,
                            layers=layers,
                            ub=datamodule.test_dataset.ub,
                            lb=datamodule.test_dataset.lb)

    trainer = pl.Trainer(
        logger=True,
        max_epochs=1,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss")],
    )
    hyperparameters = dict(lr=lr, n_layers=n_layers, layer_dims=layers)
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, datamodule=datamodule)

    return trainer.callback_metrics["val_loss"].item()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch Lightning example.")
    parser.add_argument(
        "--pruning",
        "-p",
        action="store_true",
        help="Activate the pruning feature. `MedianPruner` stops unpromising "
             "trials at the early stages of training.",
    )
    args = parser.parse_args()

    pruner = optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()

    study = optuna.create_study(direction="minimize", pruner=pruner, sampler=optuna.samplers.GPSampler())
    study.optimize(objective, n_trials=5, timeout=600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
