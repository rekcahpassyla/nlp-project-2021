from glove import GloveWordsDataset, GloveModel
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


device = 'cpu'
# This file is the text data that went into these embeddings
# It will be ../datasets/train_set_uk.txt
textfile = "../datasets/glove/train_set_uk.txt"
# This contains the glove embeddings generated from the glove.py script
inputfile = '../datasets/glove/embeddings/train_set_uk_min.pt'

dataset = GloveWordsDataset(open(textfile).read(), 10000000, device=device)
glove = GloveModel(dataset.concept_len, 3).to(device)

kws = {}
if device == 'cpu':
    kws['map_location'] = device
glove.load_state_dict(
    torch.load(inputfile, **kws))

# From the Glove embeddings paper, the embeddings we use are the
# sum of the i and j embeddings.
emb_i = glove.wi.weight.cpu().data.numpy()
emb_j = glove.wj.weight.cpu().data.numpy()
emb = emb_i + emb_j
# Pick the top 500 - can change this
top_k = 500
tsne = TSNE(metric='cosine', random_state=123)
# Call TSNE fit on only those top 500
embed_tsne = tsne.fit_transform(emb[:top_k, :])
fig, ax = plt.subplots(figsize=(30, 30))
for idx in range(top_k):
    plt.scatter(*embed_tsne[idx, :], color='steelblue')
    # dataset,_id2word[i] gives the word correponding to the
    # i'th most frequent word in that dataset.
    plt.annotate(dataset._id2word[idx],
                 (embed_tsne[idx, 0], embed_tsne[idx, 1]),
                 alpha=0.7)
