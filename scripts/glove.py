import os
import socket
from copy import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# specify GPU
GPU = torch.cuda.is_available()
GPU = False

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
    loss = weights * F.mse_loss(inputs, targets, reduction='none')
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
        #torch.save(model.state_dict(), f"{output_filename}.pt")


if __name__ == '__main__':
    import os
    cfg = {
        "train": True,
        "plot": True,
        "co_occurrence_file": None,
        "glove_options": {
            "words_dataset": True,
            # I tried various values; only got good clustering with 3-5
            "embed_dim": 3,
            "n_epochs": 100,
            "batch_size": 1000000,
            "x_max": 100,
            "alpha": 0.75,
            "output_file": None
        }
    }
    # just run for all the text files in the datasets/glove directory
    basepath = os.path.abspath(os.path.join("..", "datasets", "glove"))
    files = os.listdir(basepath)
    outdir = os.path.join(basepath, "embeddings")
    glove_opts = cfg['glove_options']
    for fn in files:
        # only process text files
        if not fn.endswith(".txt"):
            continue
        inputfile = os.path.join(basepath, fn)
        outputfile = os.path.join(outdir,
                                  os.path.basename(inputfile).replace('.txt', ''))
        dataset = GloveWordsDataset(open(inputfile).read(), 10000000, device=device)
        glove = GloveModel(dataset.concept_len, glove_opts['embed_dim']).to(device)
        if cfg['train']:
            train(glove, dataset, glove_opts['n_epochs'], glove_opts['batch_size'],
                  glove_opts['x_max'], glove_opts['alpha'],
                  outputfile)
        else:
            kws = {}
            if device == 'cpu':
                kws['map_location'] = device
            glove.load_state_dict(
                torch.load(f"{outputfile}_min.pt", **kws))
        # plotting is auxiliary so not in a function yet
        if cfg['plot']:
            import matplotlib.pyplot as plt
            from sklearn.manifold import TSNE

            emb_i = glove.wi.weight.cpu().data.numpy()
            emb_j = glove.wj.weight.cpu().data.numpy()
            emb = emb_i + emb_j
            top_k = 500
            tsne = TSNE(metric='cosine', random_state=123)
            embed_tsne = tsne.fit_transform(emb[:top_k, :])
            fig, ax = plt.subplots(figsize=(30, 30))
            for idx in range(top_k):
                plt.scatter(*embed_tsne[idx, :], color='steelblue')
                plt.annotate(dataset._id2word[idx],
                             (embed_tsne[idx, 0], embed_tsne[idx, 1]),
                             alpha=0.7)
            plt.savefig(f'{outputfile}.png')
