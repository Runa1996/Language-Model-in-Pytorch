"""
    @Time    : 2020/1/4 21:14
    @Author  : Runa
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embed_size, save_path):
        super(Word2Vec, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.save_emb_data = save_path
        self.center_vec = nn.Embedding(self.vocab_size, self.embed_size)
        self.context_vec = nn.Embedding(self.vocab_size, self.embed_size)
        self.init_embedding()

    def init_embedding(self):
        init_range = 0.5 / self.embed_size
        self.center_vec.weight.data.uniform_(-init_range, init_range)
        self.context_vec.weight.data.uniform_(-0, 0)

    def forward(self, pos_u, pos_v, neg_v):
        loss = []
        embedding_u = self.center_vec(pos_u)
        embedding_v = self.context_vec(pos_v)
        embedding_neg = self.context_vec(neg_v)
        score = torch.mul(embedding_u, embedding_v).squeeze()
        score = torch.sum(score, dim=1)
        score = F.logsigmoid(score)
        loss.append(sum(score))
        neg_score = torch.bmm(embedding_neg, embedding_u.unsqueeze(2)).squeeze()
        neg_score = torch.sum(neg_score, dim=1)
        neg_score = F.logsigmoid(-1 * neg_score)
        loss.append(sum(neg_score))
        return -1 * sum(loss)

    def save_embedding(self, id2word):
        if self.device == 'cuda':
            embedding = self.center_vec.weight.cpu().data.numpy()
        else:
            embedding = self.center_vec.weight.data.numpy()
        fout = open(self.save_emb_data, 'w')
        fout.write('Word' + '\t' + 'Vector' + '\n')
        for word_id, word in id2word.items():
            e = embedding[word_id]
            e = ' '.join([str(x) for x in e])
            fout.write('{}\t{}'.format(e, word))

