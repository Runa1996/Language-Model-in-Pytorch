"""
    @Time    : 2020/1/5 19:41
    @Author  : Runa
"""

from Input_data import Data
from SkipGram import Word2Vec
from torch.utils.data import DataLoader

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

class Config:
    def __init__(self,
                 emb_dim=300,
                 windows_size=5,
                 num_negs=5,
                 learning_rate=0.001,
                 epoch=10,
                 batch_size=64,
                 min_count=1):
        self.emb_dim = emb_dim
        self.windows_size = windows_size
        self.num_negs = num_negs
        self.lr = learning_rate
        self.epoch = epoch
        self.batch_size = batch_size
        self.min_count = min_count


class TrainProcess:
    def __init__(self, config, input_file, output_file=None):
        self.config = config
        self.input_file = input_file
        self.lr = config.lr
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if not output_file:
            root_path = os.path.dirname(input_file)
            self.output_file = os.path.join(root_path, 'Word2Vec_'+str(self.config.emb_dim)+'.txt')
        else:
            self.output_file = output_file

    def train(self):
        train_data = Data(self.input_file, self.config.min_count, self.config.windows_size, self.config.num_negs)
        train_loader = DataLoader(train_data, batch_size=self.config.batch_size, shuffle=True)
        vocab_size = train_data.get_vocab_size()
        model = Word2Vec(vocab_size, self.config.emb_dim, self.output_file).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        print('Start to Train...')
        for batch in range(self.config.batch_size):
            iter_times = 0
            for i, (pos_u, pos_v, neg_v) in (enumerate(train_loader)):
                optimizer.zero_grad()
                loss = model(pos_u, pos_v, neg_v)
                loss.backward()
                optimizer.step()

                if iter_times % 100 == 0:
                    print("Batch:{:>2d}\tIter:{:>2d}\tLoss:{:>5.4%}".format(batch, iter_times, loss))
                if batch >= 3 and iter_times % 20000 == 0:
                    lr = self.lr * (1.0 - batch / 20)
                    for params in optimizer.param_groups:
                        params['lr'] = lr
                iter_times += 1

        model.save_embedding(train_data.get_id2word())


if __name__ == '__main__':
    input_file = 'train.txt'
    config = Config()
    Train = TrainProcess(config, input_file)
    Train.train()
