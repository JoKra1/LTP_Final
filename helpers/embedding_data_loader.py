# Authors: Ella Collins, Manon Heinhuis & Joshua Krause
import torch
import numpy as np
import csv
from torch.utils.data import DataLoader, Dataset 

class EmbeddingDataset(Dataset):
    def __init__(self, data_filepath, tokenizer, window_size = 2):
        super().__init__()

        data = [] # target words
        labels = [] # context

        with open(data_filepath, newline="", encoding='utf8') as data_file:
            reader = csv.reader(data_file, delimiter=",")
            for index, line in enumerate(reader):
                """
                Use Bert encoder to transform to byte-pair and then to index
                """
                tokens = tokenizer.tokenize(line[1]) # Byte-pair
                data_idxs = tokenizer.encode(line[1]) # maps bp to index

                """
                Now prepare for cbow training, as shown in pytorch's embedding tutorial.
                Here for variable window size.
                Source: https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html
                """

                for i in range(window_size, len(data_idxs) - window_size):
                    context = []
                    target = None
                    context_index =  i - (window_size)
                    while context_index <= i + window_size:
                        if context_index == i:
                            target = data_idxs[context_index]
                        else:
                            context.append(data_idxs[context_index])

                    data.append(context)
                    labels.append(target)
                
            self.data = np.array(data)
            self.labels = np.array(labels)
