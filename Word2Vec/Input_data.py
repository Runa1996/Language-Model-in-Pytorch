"""
    @Time    : 2020/1/5 0:08
    @Author  : Runa
"""

import numpy as np
import torch
import os
import pickle as pkl
import pickle
import jieba
from collections import deque
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

class Data(Dataset):
    def __init__(self, filename, min_count, windows_size, num_negs):
        self.filename = filename
        self.windows_size = windows_size
        self.num_negs = num_negs
        self.min_count = min_count
        self.word_pair_catch = []
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.pkl_data_path = os.path.join(os.path.dirname(filename), 'train.pkl')
        self.pkl_vocab_path = os.path.join(os.path.dirname(filename), 'vocab.pkl')
        self.pkl_sample_path = os.path.join(os.path.dirname(filename), 'sample.pkl')
        if os.path.exists(self.pkl_vocab_path):
            self.word2id, self.id2word, self.word_frequency = pkl.load(open(self.pkl_vocab_path, 'rb'))
            self.num_words = len(self.word2id)
        else:
            self.get_words()            # build word2id and id2word vocabulary
        if os.path.exists(self.pkl_sample_path):
            self.sample_map = pkl.load(open(self.pkl_sample_path, 'rb'))
        else:
            self.init_sample_maps()     # build negative word map
        if os.path.exists(self.pkl_data_path):
            self.word_pair_catch = pkl.load(open(self.pkl_data_path, 'rb'))
        else:
            self.build_batch_pairs()    # build pair of words dict

    def get_words(self):
        word_frequency = {}
        self.sentence_count = 0
        self.sentence_length = 0
        print('Start to build words vocabulary...')
        with open(self.filename, 'r', encoding='utf8') as file:
            for line in file.readlines():
                l = list(jieba.cut(line.strip()))
                self.sentence_count += 1
                self.sentence_length += len(l)
                for word in l:
                    word_frequency[word] = word_frequency.get(word, 0) + 1
            self.word2id = {}
            self.id2word = {}
            self.word_frequency = {}
            for i, (word, count) in tqdm(enumerate(word_frequency.items())):
                if count < self.min_count:
                    self.sentence_length -= count
                    continue
                self.word2id[word] = i
                self.id2word[i] = word
                self.word_frequency[word] = count
            self.num_words = len(self.word2id)
            assert len(self.word2id) == len(self.id2word)
            pkl.dump((self.word2id, self.id2word, self.word_frequency), open(self.pkl_vocab_path, 'wb'))
        print('Build vocabulary done! size:{}'.format(len(self.word_frequency)))

    def init_sample_maps(self):
        self.sample_map = []
        sample_map_size = 1e6
        word_ids = []
        pow_frequency = np.array([])
        print('Start to build sample maps...')
        for word, count in tqdm(self.word_frequency.items()):
            word_ids.append(self.word2id[word])
            pow_frequency = np.append(pow_frequency, count ** 0.75)
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = np.round(ratio * sample_map_size)
        for ids, c in tqdm(zip(word_ids, count)):
            self.sample_map = np.append(self.sample_map, np.array([ids] * int(c)))
        pkl.dump(self.sample_map, open(self.pkl_sample_path, 'wb'))
        print('Build sample maps done!')

    def build_batch_pairs(self):
        print('Start to build word pairs...')
        with open(self.filename, 'r', encoding='utf8') as file:
            for line in tqdm(file.readlines()):
                l = jieba.cut(line.strip())
                word_index = [self.word2id[word] for word in l]
                for idx, center in enumerate(word_index):
                    for jdx, outer in enumerate(range(self.windows_size*2+1)):
                        if jdx == self.windows_size + 1 or idx-jdx < 0:
                            continue
                        assert center < self.num_words
                        assert outer < self.num_words
                        self.word_pair_catch.append((center, outer))
        pkl.dump(self.word_pair_catch, open(self.pkl_data_path, 'wb'))
        print('Build word pairs done!')

    def get_neg_sampling(self):
        neg_outer = np.random.choice(self.sample_map, self.num_negs).tolist()
        return neg_outer

    def get_vocab_size(self):
        return self.num_words

    def get_word2id(self):
        return self.word2id

    def get_id2word(self):
        return self.id2word

    def change_to_long(self, item):
        return torch.tensor(item, dtype=torch.long, device=self.device)

    def __len__(self):
        return len(self.word_pair_catch)

    def __getitem__(self, index):
        pos_u, pos_v = self.word_pair_catch[index]
        neg_v = self.get_neg_sampling()
        return self.change_to_long(pos_u), self.change_to_long(pos_v), self.change_to_long(neg_v)