# Authors: Ella Collins, Manon Heinhuis & Joshua Krause
import torch
import numpy as np
import csv
from torch.utils.data import DataLoader, Dataset 
from tqdm import tqdm

cat2idx = {"NewsFeed":0,
          "RightTroll":1,
          "LeftTroll":2}

idx2cat = ["NewsFeed","RightTroll","LeftTroll"]

class TwitterDataset(Dataset):
    def __init__(self, data_filepath, tokenizer,max_size = None):
        super().__init__()

        #data_file = csv.reader(data_filepath, delimiter = ",")
        data = []
        labels = []
        with open(data_filepath, newline="", encoding='utf8') as data_file:
            reader = csv.reader(data_file, delimiter=",")
            for index, line in enumerate(tqdm(reader)):
                if index == 0:
                    continue

                if max_size and index > max_size:
                    break
                """
                row == line -> line[1] content, line[0] label
                """
                tokens = tokenizer.tokenize(line[1]) # Byte-pair
                data_idxs = tokenizer.encode(line[1]) # maps bp to index

                data.append(data_idxs) # we want indices
                labels.append(cat2idx[line[0]])

        self.data = np.array(data)
        self.labels = np.array(labels)


    def __getitem__(self, index):
        return torch.Tensor(self.data[index]).long(), self.labels[index]

    def __len__(self):
        return len(self.data)

def padding_collate_fn(batch):
    """ Pads data with zeros to size of longest sentence in batch. """
    data, labels = zip(*batch)
    largest_sample = max([len(d) for d in data])
    padded_data = torch.zeros((len(data), largest_sample), dtype=torch.long)
    for i, sample in enumerate(data):
        padded_data[i, :len(sample)] = sample
    labels = torch.tensor(list(labels),dtype=torch.long)
    return padded_data, labels # if doesn't work, don't return labels :)