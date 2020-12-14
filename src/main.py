#!/usr/bin/env python3
###########################################################
# Authors: Joel Anyanti, Jui-Chieh Chang, Alex Condotti
# Carnegie Mellon Univerity
# 11-785 (Introduction to Deep Learning)
#
# main.py
###########################################################
# Imports
###########################################################
import sys, os, time
import imageio
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from IPython.display import Image

from config import *
from model import Generator, Discriminator, train
from loader import POPDataset, BarDataset, parse_data
from util import *

###########################################################
# Model Run
###########################################################
def main():
    # Setup CUDA device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    has_gpu = torch.cuda.is_available()

    # Training hyper parameters
    lr = 2e-4
    lr_scale_g = 0.25
    betas = (0.5, 0.99)
    epochs = 20
    nz = 30
    is_chord = False
    chord_dims = (13,1)

    # Load data
    #train_dataset = POPDataset("encodings.npz")
    train_data = parse_data("../dataset")
    train_dataset = BarDataset(train_data)

    train_loader = DataLoader(train_dataset, batch_size=N_BATCH, shuffle=False,
                            num_workers=0, drop_last=True)

    # Model instantiation
    torch.cuda.empty_cache()
    modelG = Generator(nz=nz, is_chord=is_chord, chord_dims=chord_dims)
    modelD = Discriminator(is_chord=is_chord, chord_dims=chord_dims)
    modelG.to(device)
    modelD.to(device)

    # Model optimizers
    optG = torch.optim.Adam(modelG.parameters(), lr=lr, betas=betas)
    optD = torch.optim.Adam(modelD.parameters(), lr=lr*lr_scale_g, betas=betas)

    # Model Criterion
    criterion = nn.BCELoss()

    # Model Train
    training_run = train(model=(modelG, modelD), train_loader=train_loader,
                    opt=(optG, optD), criterion=criterion, nz=nz,
                    device=device, is_chord=is_chord, epochs=epochs)

    plot_loss(*extract_loss(training_run))

if __name__ == '__main__':
    main()
