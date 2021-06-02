import time
import pdb
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader 
from transformers import BertTokenizer, BertForTokenClassification
from helpers.embedding_data_loader import EmbeddingDataset
from models import cbow



if __name__ == "__main__":
    # Get original tokenizer
    pretrained = 'bert-base-multilingual-cased'
    tokenizer = BertTokenizer.from_pretrained(pretrained)
    tokenizer.do_basic_tokenize = False

    data = EmbeddingDataset("data/train_merged.csv",tokenizer)