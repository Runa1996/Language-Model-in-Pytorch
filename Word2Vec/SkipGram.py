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
    def __init__(self, vocab_size, embed_size, padding_idx, save_path):
        super(Word2Vec, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.padding_idx = padding_idx
        self.save_emb_data = save_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.center_vec = nn.Embedding(self.vocab_size, self.embed_size, padding_idx=self.padding_idx)
        self.context_vec = nn.Embedding(self.vocab_size, self.embed_size, padding_idx=self.padding_idx)
        self.init_embedding()

    def init_embedding(self):
        init_range = 0.5 / self.embed_size
        self.center_vec.weight.data.uniform_(-init_range, init_range)
        self.context_vec.weight.data.uniform_(-0, 0)

    def forward(self, pos_u, pos_v, neg_v):
        loss = []
        embedding_u = self.center_vec(torch.tensor(pos_u, dtype=torch.long, requires_grad=True))
        embedding_v = self.context_vec(torch.tensor(pos_v, dtype=torch.long, requires_grad=True))
        embedding_neg = self.context_vec(torch.tensor(neg_v, dtype=torch.long, requires_grad=True))
        score = torch.mul(embedding_u, embedding_v).squeeze()
        score = torch.sum(score, sim=1)
        score = F.logsigmoid(sum(score))
        loss.append(sum(score))
        neg_score = torch.mul(embedding_neg, embedding_u).squeeze()
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

