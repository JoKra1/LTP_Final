# Authors: Ella Collins, Manon Heinhuis & Joshua Krause
import torch
import numpy as np
import csv
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset 

class EmbeddingDataset(Dataset):
    """
    Based on dataloading.py provided for assignment 4.
    Modified to work for the CBOW model.
    """
    def __init__(self, data_filepath, tokenizer, window_size = 2,header=True, max_size = None):
        super().__init__()

        data = [] # target words
        labels = [] # context

        with open(data_filepath, newline="", encoding='utf8') as data_file:
            reader = csv.reader(data_file, delimiter=",")
            for index, line in enumerate(tqdm(reader)):
                if index == 0 and header:
                    continue

                if max_size and index > max_size:
                    break
                """
                Use Bert encoder to transform to byte-pair and then to index
                """
                tokens = tokenizer.tokenize(line[1]) # Byte-pair

                # Padd first elements
                for i in range(window_size):
                    tokens.insert(0,'[PAD]')

                data_idxs = tokenizer.convert_tokens_to_ids(tokens) # map tokens directly to indices (no CLS token)

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
                        context_index += 1

                    data.append(context)
                    labels.append(target)
                
            self.data = np.array(data)
            self.labels = np.array(labels)
    
    def __getitem__(self, index):
        return torch.Tensor(self.data[index]).long(), self.labels[index]

    def __len__(self):
        return len(self.data)
