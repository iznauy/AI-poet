import numpy as np
import torch
from torch.utils import data

class PoetryDataSet(data.Dataset):

    def __init__(self, path):
        data = np.load(path)
        self.poems = torch.from_numpy(data['data'])
        self.word2ix = data['word2ix'].item()
        self.ix2word = data['ix2word'].item()


    def __len__(self):
        return self.poems.shape[0]


    def __getitem__(self, item):
        return self.poems[item]
