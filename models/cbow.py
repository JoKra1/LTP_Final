import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CBOW(nn.Module):
    """
    Cbow model tries to predict the target word given its context (of specified window length).
    The code is taken from The NLP book by Rao, Delip, and Brian McMahan: 
    Natural Language Processing with Pytorch:
    Build Intelligent Language Applications Using Deep Learning chapter 3!
    """
    def __init__(self, vocabulary_size, embedding_size, padding_idx=0):
        super(CBOW, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=vocabulary_size,
        embedding_dim=embedding_size,
        padding_idx=padding_idx)
        self.fc1 = nn.Linear(in_features=embedding_size,
        out_features=vocabulary_size)

    def forward(self, x_in, apply_softmax=False):
        
        x_embedded_sum = self.embedding(x_in).sum(dim=1)
        y_out = self.fc1(x_embedded_sum)

        if apply_softmax:
            y_out = F.softmax(y_out, dim=1)

        return y_out
