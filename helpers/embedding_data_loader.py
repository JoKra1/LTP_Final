import torch
import numpy as np
import csv
from tqdm import tqdm

class EmbeddingDataset:
    """
    Based on iterator shown in Gensim w2v tutorial:
    See: https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html
    """
    def __init__(self, data_filepath, tokenizer):
        self.tokenizer = tokenizer
        self.path = data_filepath
        self.len = 0

        with open(data_filepath, newline="", encoding='utf8') as data_file:
            reader = csv.reader(data_file, delimiter=",")
            self.len = len(list(reader))
        
    
    def __iter__(self):
        
        for line in open(self.path, newline="", encoding='utf8',mode="r"):
            line = line.strip()
            # R wraps each tweet into quotes, which need to be removed when
            # reading like this!
            line = line[1:-1]
            """
            We want to learn embeddings for the same tokens that the BERT models
            have access to, this has been previously been recommended and shown
            to also increase performance!

            See: https://bpemb.h-its.org/
            """
            tokens = self.tokenizer.tokenize(line) # Wordpiece (see Devlin et al. 2018)
            yield tokens
