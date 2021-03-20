import os
import socket
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# specify GPU
GPU = torch.cuda.is_available()

# If you have a problem with your GPU, set this to "cpu" manually
device = torch.device("cuda:0" if GPU else "cpu")

from collections import Counter, defaultdict

import numpy as np
import torch



class GloveWordsDataset:
    # TODO: Need to refactor so that all datasets take a co-occurrence matrix
    # instead of building in here
    def __init__(self, text, n_words=200000, window_size=5, device='cpu'):
        # "text" is just an enormous string of all the words, in order,
        # joined together
        self._window_size = window_size
        self._tokens = text.split(" ")[:n_words]
        word_counter = Counter()
        word_counter.update(self._tokens)
        self._word2id = {w:i for i, (w,_) in enumerate(word_counter.most_common())}
        self._id2word = {i:w for w, i in self._word2id.items()}
        self._vocab_len = len(self._word2id)
        self.concept_len = self._vocab_len

        self._id_tokens = [self._word2id[w] for w in self._tokens]
        self.device = device

        self._create_coocurrence_matrix()
        print("# of words: {}".format(len(self._tokens)))
        print("Vocabulary length: {}".format(self._vocab_len))

    def _create_coocurrence_matrix(self):
        device = self.device
        cooc_mat = defaultdict(Counter)
        for i, w in enumerate(self._id_tokens):
            start_i = max(i - self._window_size, 0)
            end_i = min(i + self._window_size + 1, len(self._id_tokens))
            for j in range(start_i, end_i):
                if i != j:
                    c = self._id_tokens[j]
                    cooc_mat[w][c] += 1 / abs(j-i)

        self._i_idx = list()
        self._j_idx = list()
        self._xij = list()

        #Create indexes and x values tensors
        for w, cnt in cooc_mat.items():
            for c, v in cnt.items():
                self._i_idx.append(w)
                self._j_idx.append(c)
                self._xij.append(v)

        self._i_idx = torch.LongTensor(self._i_idx).to(device)
        self._j_idx = torch.LongTensor(self._j_idx).to(device)
        self._xij = torch.FloatTensor(self._xij).to(device)
        self.N = len(self._i_idx)

    def get_batches(self, batch_size):
        #Generate random idx
        rand_ids = torch.LongTensor(np.random.choice(len(self._xij), len(self._xij), replace=False))

        for p in range(0, len(rand_ids), batch_size):
            batch_ids = rand_ids[p:p+batch_size]
            yield self._xij[batch_ids], self._i_idx[batch_ids], self._j_idx[batch_ids]


class GloveModel(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(GloveModel, self).__init__()
        self.wi = nn.Embedding(num_embeddings, embedding_dim)
        self.wj = nn.Embedding(num_embeddings, embedding_dim)
        self.bi = nn.Embedding(num_embeddings, 1)
        self.bj = nn.Embedding(num_embeddings, 1)

        self.wi.weight.data.uniform_(-1, 1)
        self.wj.weight.data.uniform_(-1, 1)
        self.bi.weight.data.zero_()
        self.bj.weight.data.zero_()

    def forward(self, i_indices, j_indices):
        w_i = self.wi(i_indices)
        w_j = self.wj(j_indices)
        b_i = self.bi(i_indices).squeeze()
        b_j = self.bj(j_indices).squeeze()

        x = torch.sum(w_i * w_j, dim=1) + b_i + b_j

        return x


def weight_func(x, x_max, alpha):

    wx = (x/x_max)**alpha
    wx = torch.min(wx, torch.ones_like(wx))
    return wx.to(device)


def wmse_loss(weights, inputs, targets):
    loss = weights * F.mse_loss(inputs, targets, reduction='<none')
    return torch.mean(loss).to(device)


def train(model, dataset, n_epochs, batch_size, x_max=100, alpha=0.75,
          output_filename='glove'):
    optimizer = optim.Adagrad(glove.parameters(), lr=0.05)

    n_batches = int(dataset.N / batch_size)
    loss_values = list()
    min_loss = np.inf
    l = np.inf
    for e in range(1, n_epochs+1):
        batch_i = 0

        for x_ij, i_idx, j_idx in dataset.get_batches(batch_size):

            batch_i += 1

            optimizer.zero_grad()

            outputs = model(i_idx, j_idx)
            weights_x = weight_func(x_ij, x_max, alpha)
            loss = wmse_loss(weights_x, outputs, torch.log(x_ij))

            optimizer.step()
            l = loss.item()
            loss_values.append(l)

            #if batch_i % 1024 == 0:
            print("Epoch: {}/{} \t Batch: {}/{} \t Loss: {}".format(e, n_epochs, batch_i, n_batches, np.mean(loss_values[-20:])))
        print("Saving model...")
        if l < min_loss:
            min_loss = l
            torch.save(model.state_dict(), f"{output_filename}_min.pt")
        torch.save(model.state_dict(), f"{output_filename}.pt")


if __name__ == '__main__':
    import sys
    import json
    cfg_filename = sys.argv[1]
    cfg = json.load(open(cfg_filename))
    glove_opts = cfg['glove_options']
    if glove_opts['words_dataset']:
        dataset = GloveWordsDataset(open("text8").read(), 10000000, device=device)
    else:
        dataset = GloveDataset(cfg['co_occurrence_file'], device)

    glove = GloveModel(dataset.concept_len, cfg['glove_options']['embed_dim']).to(device)

    if cfg['train']:
        train(glove, dataset, glove_opts['n_epochs'], glove_opts['batch_size'],
              glove_opts['x_max'], glove_opts['alpha'],
              glove_opts['output_file'])
    else:
        kws = {}
        if device == 'cpu':
            kws['map_location'] = device
        glove.load_state_dict(
            torch.load(f"{glove_opts['output_file']}_min.pt", **kws))
    # plotting is auxiliary so not in a function yet
    if cfg['plot']:
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE

        if glove_opts['words_dataset']:
            emb_i = glove.wi.weight.cpu().data.numpy()
            emb_j = glove.wj.weight.cpu().data.numpy()
            emb = emb_i + emb_j
            top_k = 300
            tsne = TSNE(metric='cosine', random_state=123)
            embed_tsne = tsne.fit_transform(emb[:top_k, :])
            fig, ax = plt.subplots(figsize=(14, 14))
            for idx in range(top_k):
                plt.scatter(*embed_tsne[idx, :], color='steelblue')
                plt.annotate(dataset._id2word[idx],
                             (embed_tsne[idx, 0], embed_tsne[idx, 1]),
                             alpha=0.7)
            plt.savefig('glove_words.png')
        else:
            # Download from https://storage.googleapis.com/openimages/v6/oidv6-class-descriptions.csv
            #labelsfn = 'oidv6-class-descriptions.csv'

            import sys

            if hostname == 'tempoyak':
                ppath = '/opt/github.com/spond/spond/experimental'
            else:
                ppath = '/home/petra/spond/spond/experimental'
            labelsfn = cfg['labels_file']
            sys.path.append(ppath)

            TOP_K = 500
            from openimage.readfile import readlabels

            labels, names = readlabels(labelsfn, rootdir=None)
            idx_to_name = {
                v: names[k] for k, v in labels.items()
            }

            emb_i = glove.wi.weight.data.numpy()
            emb_j = glove.wj.weight.data.numpy()
            emb = emb_i + emb_j
            # find the most commonly co-occuring items
            # These are the items which appear the most times
            dense = dataset.cooc_mat.to_dense()
            incidences = dense.sum(axis=0)
            nonzero = np.nonzero(incidences)
            nonzero_incidences = incidences[nonzero]
            indexes = np.argsort(nonzero_incidences.t()).squeeze()
            top_k = min(TOP_K, indexes.shape[0])
            top_k_indices = nonzero[indexes[-top_k:]].t().squeeze()

            if glove_opts['embed_dim'] == 2:
                embeddings = emb[top_k_indices, :]
            else:

                tsne = TSNE(metric='cosine', n_components=2, random_state=123)
                embeddings = tsne.fit_transform(emb[top_k_indices, :])

            fig = plt.figure(figsize=(14, 14))

            for idx, concept_idx in enumerate(top_k_indices):
                m = embeddings[idx, :]
                plt.scatter(*m, color='steelblue')
                concept = idx_to_name[concept_idx.item()]
                plt.annotate(concept, (embeddings[idx, 0], embeddings[idx, 1]),
                             alpha=0.7)
            plt.savefig(f"{glove_opts['output_file']}.png")
