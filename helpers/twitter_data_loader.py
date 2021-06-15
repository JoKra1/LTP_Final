import torch
import numpy as np
import csv
from torch.utils.data import DataLoader, Dataset 
from tqdm import tqdm
from enum import Enum

cat2idx = {"NewsFeed":0,
          "RightTroll":1,
          "LeftTroll":2,
          "Fearmonger":3,
          "HashtagGamer":4}

idx2cat = ["NewsFeed","RightTroll","LeftTroll","Fearmonger","HashtagGamer"]

class SupportedFormat(Enum):
    RNN = 1
    TRANSFORMER = 2

class TwitterDataset(Dataset):
    """
    We used the dataloading code provided for assignment 4 as a template to
    build this dataset class. We adapted the code to work for both the RNN
    and the (M)BERT, since the former does not need the special tokens used
    by BERT (e.g. cls for classification problem, and the seperator token).
    """
    def __init__(self, data_filepath, tokenizer,max_size = None,format = SupportedFormat.TRANSFORMER):
        super().__init__()

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

                if len(tokens) == 0:
                    """
                    There is a single example somewhere in the training data
                    with no text. This one catches it.
                    """
                    print("Skipping non-encodeable example.")
                    continue

                if format == SupportedFormat.TRANSFORMER:
                    # Permit for transformer specific encodings
                    data_idxs = tokenizer.encode(line[1]) # Indices
                elif format == SupportedFormat.RNN:
                    # map tokens directly to indices (no CLS token)
                    data_idxs = tokenizer.convert_tokens_to_ids(tokens) 

                data.append(data_idxs) # we want indices
                labels.append(cat2idx[line[0]])

        self.data = np.array(data)
        self.labels = np.array(labels)


    def __getitem__(self, index):
        """
        Taken from dataloading code provided for assignment 4. However, we
        just return a single label so no cast to tensor is necessary.
        """
        return torch.Tensor(self.data[index]).long(), self.labels[index]

    def __len__(self):
        """
        Taken from dataloading code provided for assignment 4.
        """
        return len(self.data)

def padding_collate_fn(batch):
    """
    Taken from dataloading code provided for assignment 4. However,
    we only pad the data, since we have only one label per tweet.
    Thus we cast labels to a long tensor in the end.

    We included a sanity check that should basically never print, since
    the RNN cannot deal with empty (e.g. only pad tokens) tweets, due
    to the packing mechanisms used.
        
    """
    data, labels = zip(*batch)
    lens = [len(d) for d in data]
    largest_sample = max(lens)
    smallest_sample = min(lens)

    padded_data = torch.zeros((len(data), largest_sample), dtype=torch.long)
    for i, sample in enumerate(data):
        padded_data[i, :len(sample)] = sample
    # Until here labels is a tuple, so cast to tensor.
    labels = torch.tensor(list(labels),dtype=torch.long)

    if smallest_sample <= 0:
        """
        Just a sanity check. This should NEVER execute. Otherwise RNN training will fail.
        """
        print("invalid.")

    return padded_data, labels