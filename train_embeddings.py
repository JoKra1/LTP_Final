import time
import pdb
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader 
from transformers import BertTokenizer, BertForTokenClassification
from helpers.embedding_data_loader import EmbeddingDataset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from gensim.models.keyedvectors import KeyedVectors
from gensim.models import Word2Vec

def tsne_plot(path_to_embeddings,n=1000):
    """
    Based on my submission for assignment 3.

    Sources:

    For TSNE: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    For plot: https://github.com/antot/ltp-notebooks-201920/blob/master/05_Representations.ipynb
    """
    print("Plotting embeddings, this might take a while...")
    embeddings = KeyedVectors.load(path_to_embeddings)
    X_TSNE = embeddings.vectors[104:n,:] # First 104 are unsued. Leftovers from Bert tokenizer.
    reduced_embed = TSNE(n_components= 2).fit_transform(X_TSNE)
    for i,pair in enumerate(reduced_embed):
        plt.scatter(pair[0],pair[1])
        plt.text(pair[0] + 0.02,pair[1],embeddings.index_to_key[i + 104])
    plt.title("TSNE reduced embedding visualization")
    plt.xlabel("TSNE Dimension 1")
    plt.ylabel("TSNE Dimension 2")
    plt.show()

def convertEmbeddings(pathTow2v,tokenizer):
    """
    Creates matrix with correct indexing for RNN & BERT.

    The vocab created by Gensim's w2v is not in the same
    order as the one provided by the BERT tokenizer. This
    function basically reverse traces the ordering to ensure
    the embeddings correspond to the correct index when training
    with the RNN and the BERT.

    We cast to FloatTensor in the end because that is what is
    requested by the call to nn.Embedding.from_pretrained(), as
    discussed in the references below.

    Source:
    https://stackoverflow.com/questions/49710537/pytorch-gensim-how-to-load-pre-trained-word-embeddings
    https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html#torch.nn.Embedding.from_pretrained
    """
    vocab = tokenizer.get_vocab()
    keys = list(vocab.keys())
    model = Word2Vec.load(pathTow2v)
    wv = model.wv
    corrected = []

    for correct_index in range(tokenizer.vocab_size):
        token = keys[correct_index]
        gensim_index = wv.key_to_index[token]
        vect = wv.vectors[gensim_index]
        if correct_index == 0:
            corrected.append(np.array([0 for _ in range(len(vect))],dtype=np.float32))
        else:
            corrected.append(vect)
    
    corrected = np.stack(corrected)
    corrected = torch.FloatTensor(corrected)
    return corrected


if __name__ == "__main__":

    # Get original tokenizer
    pretrained = 'bert-base-multilingual-cased'
    tokenizer = BertTokenizer.from_pretrained(pretrained)
    tokenizer.do_basic_tokenize = False

    # Load training set for embeddings
    train_set = EmbeddingDataset("data/train_emb_merged.csv",tokenizer)
    vocab = tokenizer.get_vocab()
    keys = list(vocab.keys())
    
    # Initialize w2v model
    """
    We looked at the documentation and examples (see below)
    and the GENSIM implementation is based on the original C implementation
    that was also recommended by Lukas (Thanks again!!!).
    However, the Gensim implementation allowed us to pre-define the
    vocabulary, which allowed us to just re-create it from the BERT tokenizer one,
    ensuring that the vocabulary would be exactly the same across all models.

    This of course means that some vectors returned by the w2v model will just be
    randomly initialized ones, in case the corresponding tokens were part of the
    original vocabulary but not part of the actual corpus used for training.

    The parameters below were based on the ones recommended by Lukas, and we opted
    for cbow (sg=0).

    Sources:
    https://radimrehurek.com/gensim/models/word2vec.html
    https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html

    """
    model = Word2Vec(vector_size=768,sg=0,negative=10,min_count=0,window=10,workers=16)
    model.build_vocab([keys]) # Pre-instantiate vocabulary using the keys from Bert tokenizer

    print("Beginning training")
    model.train(train_set,epochs=10,total_examples=train_set.len,compute_loss=True)

    # Save actual model
    model.save("embeddings/w2v.model")

    # Save embeddings
    wv = model.wv
    wv.save("word_embeddings.kv")
    tsne_plot("word_embeddings.kv")
    