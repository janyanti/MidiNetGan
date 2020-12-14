#!/usr/bin/env python3
###########################################################
# Authors: Joel Anyanti, Jui-Chieh Chang, Alex Condotti
# Carnegie Mellon Univerity
# 11-785 (Introduction to Deep Learning)
#
# loader.py
###########################################################
# Imports
###########################################################
import os
import numpy as np
import torch
import pypianoroll
from torch.utils.data import Dataset, DataLoader

from config import *
###########################################################
# Bar Collection
###########################################################
"""
    Bar Collection:
        A collection of multiple training points on the granularity of
        a music bar.

        For simplicity we assume that all Midi input files share the
        following properties

        Resolution: 24 (per beat)
        Tempo: 120 bpm
        Time Signature: 4/4
        Note Pitch: [0-127] (128 possibilities)
        # Tracks: 1 (Single-Track Midi)
            - If Midi contains multiple tracks, use only 1st track

"""
def midi_to_array(path):
    """
        midi_to_array: Returns binarized midi represention of input file
            as numpy.ndarry

        Args:
            path(str): Path to target midi file

        Returns:
            data (np.ndarry): Matrix representation of midi file
    """
    # Import midi data to pypianoroll Multitrack
    data = pypianoroll.read(path)

    # Export Multitrack to numpy ndarray
    data = data.stack() #(N x T x P)

    # Select 1st track if other tracks present
    data = np.expand_dims(data[0,:,:], axis=0)

    # Set all velocity values to zero to binarize data
    data[data >= 1] = 1

    data_len = data.shape[1] // RESOLUTION

    return data, data_len

def parse_data(datadir):
    """
        parse_data: Reads all midi files from a directory to produce a
            bar collection

        Args:
            datadir(str): Directory to import data from

        Return:
            bar_collection (np.ndarry: N x T x P): Resulting collection
                from file directory

    """
    def append_files(datadir):
        for root, dirs, filenames in os.walk(datadir):
            #for dir_ in dirs:
            #    append_files(dir_)
            for filename in filenames:
                if filename.endswith(".mid") or filename.endswith(".midi"):
                    path = os.path.join(root, filename)
                    midi_data, midi_len = midi_to_array(path)
                    midi_list.append(midi_data)

    midi_list = []

    append_files(datadir)

    print("Loaded {} files from directory: {}".format(len(midi_list), datadir))

    # Concatenate arrays along time (T) dimension
    bar_concat = np.concatenate(midi_list, axis=1) # (1 x T x H)
    num_timesteps = bar_concat.shape[1]
    num_bars = num_timesteps // BAR
    bar_subset = num_timesteps - (num_timesteps % BAR) # Account for uneven sequence lengths

    print("Resulting Collection has a total of {} bars".format(int(num_bars)))

    # Process collection for training
    bar_concat = np.transpose(bar_concat, axes=(0,2,1)) # (1 x H x T)
    bars = np.array_split(bar_concat[:,:,:bar_subset], num_bars, axis=2) # List (N): (H x W)
    bars = [np.expand_dims(bar, axis=0) for bar in bars] # List (N): (1 x H x W)
    bar_collection = np.concatenate(bars, axis=0) # (N, 1, H, W)

    return bar_collection

def parse_stream_data(datadir):
  """
      parse_data: Reads all midi files from a directory to produce a
          bar collection

      Args:
          datadir(str): Directory to import data from

      Return:
          bar_collection (np.ndarry: N x T x P): Resulting collection
              from file directory

  """
  def append_files(datadir):
      midi_list = []
      midi_lens = []
      for root, dirs, filenames in os.walk(datadir):
          for dir_ in dirs:
              append_files(dir_)
          for filename in filenames:
              if filename.endswith(".mid") or filename.endswith(".midi"):
                  path = os.path.join(root, filename)
                  midi_data, midi_len = midi_to_array(path)
                  midi_list.append(midi_data)
                  midi_lens.append(midi_len)

      return midi_list, midi_lens

  return append_files(datadir)

###########################################################
# Dataset
###########################################################
class BarDataset(Dataset):
    def __init__(self, collection, step_size=BAR):
        self.data, self.data_len = self.extract_data(collection)
        self.step_size = step_size

    def __len__(self):
        return self.data_len

    @property
    def shape(self):
        return self.data.shape

    def __getitem__(self, idx):
        X = self.data[idx]
        X_prev = torch.zeros(X.shape).cuda() if idx == 0 else self.data[idx-1]

        return X, X_prev

    def extract_data(self, collection):
        data_len = collection.shape[0]

        # Transform data to appropriate format
        data = torch.Tensor(collection)
        data = data.cuda()

        return data, data_len

class BarStreamDataset(Dataset):
    def __init__(self, stream_collection, stream_lens, step_size=BAR, stream_limit=100):
        self.files = stream_collection
        self.files_lens = stream_lens
        self.files_len = sum(stream_lens)
        self.stream_idx = 0
        self.stream_limit = stream_limit
        self.step_size = step_size
        self.data_len = 0
        self.files_idx = 0

    def __len__(self):
        return self.files_len

    @property
    def shape(self):
        return self.data.shape

    def __getitem__(self, idx):
        idx_ = idx

        if idx_ == 0:
          self.extract_data(reset=True)

        idx = idx - self.files_idx
        X = self.data[idx]
        X_prev = torch.zeros(X.shape).cuda() if idx == 0 else self.data[idx-1]

        if idx == self.data_len - 1:
          self.files_idx += self.data_len
          self.extract_data()
          print("Loading more data...")

        return X, X_prev

    def extract_data(self, reset=False):

        if reset:
          self.stream_idx = 0
          self.files_idx = 0

        # Concatenate arrays allong time (T) dimension
        lower_idx = self.stream_idx
        upper_idx = min(self.stream_idx + self.stream_limit, len(self.files))

        midi_list = self.files[lower_idx:upper_idx]
        bar_concat = np.concatenate(midi_list, axis=1) # (1 x T x H)
        num_bars = bar_concat.shape[1] / RESOLUTION


        # Process collection for training
        bar_concat = np.transpose(bar_concat, axes=(0,2,1)) # (1 x H x T)
        bars = np.array_split(bar_concat, num_bars, axis=2) # List (N): (H x W)
        bars = [np.expand_dims(bar, axis=0) for bar in bars] # List (N): (1 x H x W)
        bar_collection = np.concatenate(bars, axis=0) # (N, 1, H, W)

        self.data_len = bar_collection.shape[0]

        # Transform data to appropriate format
        self.data = torch.Tensor(bar_collection)
        self.data = self.data.cuda()

        self.stream_idx += self.stream_limit


class POPDataset(Dataset):
    def __init__(self, data_path, per_bar=True):
        data = np.load(data_path, allow_pickle=True)
        self.per_bar = per_bar
        self.midis = torch.Tensor(np.expand_dims(data['midis'], axis=1))
        self.chords = torch.Tensor(data['chords'])
        self.chords_per_bar = torch.Tensor(data['chords_per_bar'])
        self.margin_indices = data['margin_indices']
        print("POP909 dataset loaded, midi = {}, chords = {}, chords_per_bar = {}"
              .format(self.midis.size(), self.chords.size(), self.chords_per_bar.size()))

    def __len__(self):
        return self.midis.size(0)

    def __getitem__(self, index):
        if not self.per_bar:
            # (prev_midi (1x128x96), cur_midi (1x128x96), cur_chord (13x4))
            if index == 0:
                return torch.zeros(self.midis[index].size()), self.midis[index], self.chords[index]
            else:
                return self.midis[index-1], self.midis[index], self.chords[index]
        else:
            # (prev_midi (1x128x96), cur_midi (1x128x96), cur_chord (13x1))
            if index == 0:
                return torch.zeros(self.midis[index].size()), self.midis[index], self.chords_per_bar[index]
            else:
                return self.midis[index-1], self.midis[index], self.chords_per_bar[index]
