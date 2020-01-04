"""
    @Time    : 2020/1/5 0:08
    @Author  : Runa
"""

import numpy as np
import torch
import pickle
import jieba
from collections import deque
from torch.utils.data import Dataset, DataLoader

class Data(Dataset):
    def __init__(self, filename, min_count, windows_size):
        self.filename = filename
        self.windows_size = windows_size
        self.get_words()
        self.word_pair_catch = deque()
        self.init_sample_table()

    def get_words(self, min_count):
        self.word_frequency = {}
        self.sentence_count = 0
        self.sentence_length = 0
        with open(self.filename, 'r') as file:
            for line in file.readlines():
                l = list(jieba.cut(line.strip()))
                self.sentence_length += 1
                self.sentence_length += len(l)
                for word in l:
                    self.word_frequency[word] = self.word_frequency.get(word, 0) + 1
            self.word2id = {}
            self.id2word = {}
            for i, (word, count) in enumerate(self.word_frequency.items()):
                if count < min_count:
                    self.sentence_length -= count
                    self.word_frequency.pop(word)
                    continue
                self.word2id[word] = i
                self.id2word[i] = word
            self.num_words = len(self.word2id)
            assert len(self.word2id) == len(self.id2word)

    def init_sample_maps(self):
        self.sample_map = []
        sample_map_size = 1e8
        pow_frequency = np.array(list(self.word_frequency.values())) ** 0.75
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = np.round(ratio * sample_map_size)
        for i, c in enumerate(count):
            self.sample_map = np.append(self.sample_map, [i] * c)

    def get_batch_pairs(self):
        with open(self.filename, 'r') as file:
            for line in file.readlines():
                l = jieba.cut(line.strip())
                word_index = [self.word2id[word] for word in l]
                for idx, center in enumerate(word_index):


    def __getitem__(self, index):

