#!/usr/bin/env python3
###########################################################
# Authors: Joel Anyanti, Jui-Chieh Chang, Alex Condotti
# Carnegie Mellon Univerity
# 11-785 (Introduction to Deep Learning)
#
# util.py
###########################################################
# Imports
###########################################################
import sys, os, re
from datetime import datetime
from matplotlib import pyplot as plt

from config import *
###########################################################
# Plot Run
###########################################################
def extract_loss(text):
    d_loss = re.findall("average_lossD: (\d+.\d+)", text)
    d_loss = [float(val) for val in d_loss]
    g_loss = re.findall("average_lossG: (\d+.\d+)", text)
    g_loss = [float(val) for val in g_loss]

    return d_loss, g_loss

def plot_loss(d_loss, g_loss, plot_name=None):
    plt.plot(d_loss, label="D Loss")
    plt.plot(g_loss, label="G Loss")
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')

    if not os.path.exists("training_plots"):
        os.mkdir("training_plots")

    save_str = "plt-{}".format(datetime.now().isoformat(timespec='minutes')) if plot_name is None else plot_name
    save_path = os.path.join("training_plots", save_str + ".png")

    plt.savefig(save_path)
    plt.show()

###########################################################
# Display Images
###########################################################
def bars_to_img(bar_sequence, path, transpose=False):
    num_bars = len(bar_sequence)
    data = [torch.unsqueeze(bar_sequence[i][0], dim=0).detach().cpu().numpy() for i in range(num_bars)]

    # Arrange data into H X W numpy array
    data = np.concatenate(data, axis=3)[0][0]

    # Fix output between [0, 255] for grayscale conversion
    data = np.clip((data - np.min(data))*(255.0/(np.max(data) - np.min(data))), 0, 255).astype(np.uint8)

    imageio.imwrite(path, data)
    Image(filename=path)
