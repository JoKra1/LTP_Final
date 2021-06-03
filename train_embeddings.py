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
from models.cbow import CBOW
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from gensim.models.keyedvectors import KeyedVectors

"""
Optionally run on CUDA as discussed in https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(train,val,model,epochs=10,save_embeddings_path="embeddings/",save_model_path="models/"):
    """
    Implementation below is based on the optimization routine
    on the pytorch website, also used in the previous assignments.

    Also the loss calculation parts are taken from the code and solutions that were provided for
    assignments 2 and 3.

    Source: https://pytorch.org/docs/stable/optim.html#taking-an-optimization-step
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_f = nn.CrossEntropyLoss() # Based on assignment 2 referring to NLP book, page 47!
    for epoch in range(epochs):
        total_loss = 0
        n = 0
        print(f"Epoch: {epoch}")
        for i, batch in enumerate(tqdm(train)):
            data, labels = batch

            data = data.to(device)
            labels = labels.long()
            labels = labels.to(device)
        
            # i. zero gradients
            optimizer.zero_grad()

            # ii. do forward pass
            outp = model(data)
            
            # iii. get loss
            loss = loss_f(outp,labels)

            # add loss to total_loss
            total_loss += loss.item()
            n += 1

            # iv. do backward pass
            loss.backward()

            # v. take an optimization step
            optimizer.step()
        print(f"Average trainings loss: {total_loss/n}")

        # Evaluate
        """
        As recommended in chapter 5 of the NLP book, we do perform evaluation
        during training of the embeddings. Since the model essentially solves a
        classification task, we check how many words in the validation set
        are classified correctly based on their context. This does not allow
        us to directly infer whether embeddings become more meaningful, but it
        still allows us to infer whether the embeddings are starting to reflect
        more useful representations for solving the classification problem when
        confronted with new input.
        """
        val_acc = 0
        n = 0
        model.eval()
        if save_embeddings_path:
            save_embeddings(model,vocab,vocab_size,model.embedding_size,f"{save_embeddings_path}embeddings{epoch}.txt")
        if save_model_path:
            torch.save(model,f"{save_model_path}model{epoch}.pt")
        for batch in val:
            data, labels = batch
            data = data.to(device)

            # Predictions
            outp = model(data)
            outp = torch.max(outp, dim=1).indices

            # Copy back to cpu for conversion
            labels = labels.cpu()
            outp = outp.cpu()

            # Calculate batch accuracy
            batch_acc = accuracy_score(labels.numpy(),outp.numpy())
            val_acc += batch_acc
            n += 1
        print(f"Validation Accuracy: {val_acc/n}")
        model.train()

def save_embeddings(model,vocab,vocab_size,embedding_dim,output_path):
    """
    Writes to gensim format!

    Based on my submission for assignment 3.
    
    Source for format: https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/test/test_data/word2vec_pre_kv_c
    """
    print("Saving embeddings, this might take a while...")
    W = model.embedding.weight
    vocab_words = list(vocab.keys())
    with open(output_path, mode="w", encoding="utf8") as output:
        output.write(f"{vocab_size} {embedding_dim}\n")
        for i,r in enumerate(W):
            row = r.tolist()
            row.insert(0,vocab_words[i])
            row = " ".join([str(c) for c in row])
            output.write(row + "\n")

def tsne_plot(path_to_embeddings,n=1000):
    """
    Based on my submission for assignment 3.

    Sources:

    For TSNE: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    For plot: https://github.com/antot/ltp-notebooks-201920/blob/master/05_Representations.ipynb
    """
    print("Plotting embeddings, this might take a while...")
    embeddings = KeyedVectors.load_word2vec_format(path_to_embeddings, binary=False)
    X_TSNE = embeddings.vectors[104:n,:] # First 104 are unsued. Leftovers from Bert tokenizer.
    reduced_embed = TSNE(n_components= 2).fit_transform(X_TSNE)
    for i,pair in enumerate(reduced_embed):
        plt.scatter(pair[0],pair[1])
        plt.text(pair[0] + 0.02,pair[1],embeddings.index_to_key[i + 104])
    plt.title("TSNE reduced embedding visualization")
    plt.xlabel("TSNE Dimension 1")
    plt.ylabel("TSNE Dimension 2")
    plt.show()

if __name__ == "__main__":
    print(device)
    # Get original tokenizer
    pretrained = 'bert-base-multilingual-cased'
    tokenizer = BertTokenizer.from_pretrained(pretrained)
    tokenizer.do_basic_tokenize = False

    # Training set for embeddings
    train_set = EmbeddingDataset("data/train_emb_merged.csv",tokenizer,max_size=100,window_size=4)
    train_loader = DataLoader(train_set,
        shuffle=True,
        batch_size=256)
    
    # Validation set (test) for embeddings
    val_set = EmbeddingDataset("data/test_emb_merged.csv",tokenizer, max_size = 100,window_size=4)
    val_loader = DataLoader(val_set,
        shuffle=False,
        batch_size=256)

    # Define model
    vocab = tokenizer.get_vocab()
    vocab_size = tokenizer.vocab_size
    cbow = CBOW(vocabulary_size = vocab_size,
                embedding_size = 300)

    # Or load
    # cbow = torch.load("models/cbow.pt")
    cbow.to(device)
    
    # train
    cbow.train()
    train(train_loader,
          val_loader,
          cbow,
          epochs=100)
    

    # save model
    torch.save(cbow,"models/cbow.pt")

    # plot
    tsne_plot("embeddings/embeddings0.txt")
    