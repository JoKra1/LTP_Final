import torch
import numpy as np
import torch.nn as nn


class RNN(nn.Module):
    
    #Implement freeze for initialising embeddings if there are embeddings inputted as args
    def __init__(self, input_dim, output_dim):
        """
        Args:
            input_dim (int): size of the input features ( vocab size)
            output_dim (int): number of classes
        """
        
        super(RNN, self).__init__()

        self.emb = nn.Embedding(input_dim, 400)
        self.gru = nn.GRU(400, 100) # Haven't looked into good amount of nodes
        self.den = nn.Dense(100, output_dim)

    def forward(self, x):
        """The forward pass of the NN

        Args:
            x (torch.Tensor): an input data tensor. 
                x.shape should be (batch_size, input_dim)
        Returns:
            the resulting tensor. tensor.shape should be (batch_size, num_classes)
        """

        x = self.emb(x)
        x = self.gru(x)
        x = self.den(x)
        x = F.log_softmax(x, dim=1)

        return x

    '''
    def print_params(self):
        """Print the parameters (theta) of the network. Mainly for debugging purposes"""
        for name, param in model.named_parameters():
            print(name, param.data)
    '''

    ''' IDK WHERE TO PUT THIS
	pretrained = 'bert-base-multilingual-cased'
	tokenizer = BertTokenizer.from_pretrained(pretrained)
	tokenizer.do_basic_tokenize = False
	'''

    def load_data():
	    train_dataset = TwitterDataset("data/train_merged.csv", tokenizer)
		train_data = DataLoader(train_dataset,
			shuffle = True,
			collate_fn = padding_collate_fn,
			batch_size = batch_size)
	    val_dataset = TwitterDataset("data/val_merged.csv", tokenizer)
		val_data = DataLoader(val_dataset,
			collate_fn = padding_collate_fn,
			batch_size = batch_size)
	    test_dataset = TwitterDataset("data/test_merged.csv", tokenizer)
		test_data = DataLoader(test_dataset,
		collate_fn = padding_collate_fn,
		batch_size = batch_size)
		''' Idk what need from this yet and what not. 
	    ### load data
	    train_sents, train_y = load_animacy_sentences_and_labels(trainfile)
	    dev_sents, dev_y = load_animacy_sentences_and_labels(devfile)
	    test_sents, test_y = load_animacy_sentences_and_labels(testfile)

	    ### create mapping word to indices
	    word2idx = {"_UNK": 0}  # reserve 0 for OOV

	    ### convert training etc data to indices
	    X_train = [[get_index(w,word2idx) for w in x] for x in train_sents]
	    freeze=True
	    X_dev = [[get_index(w,word2idx,freeze) for w in x] for x in dev_sents]
	    X_test = [[get_index(w,word2idx,freeze) for w in x] for x in test_sents]
		
	#    print(X_train[0])

	    vocab_size = len(word2idx)
	    print("#vocabulary size: {}".format(len(word2idx)))
	    X_train = convert_to_n_hot(X_train, vocab_size)
	    X_dev = convert_to_n_hot(X_dev, vocab_size)
	    X_test = convert_to_n_hot(X_test, vocab_size)

	    ### convert labels to one-hot
	    label2idx = {label: i for i, label in enumerate(set(train_y))}
	    num_labels = len(label2idx.keys())
	    print("#Categories: {}, {}".format(label2idx.keys(), label2idx.values()))
	    y_train = convert_to_index(train_y, label2idx, num_labels)
	    y_dev = convert_to_index(dev_y, label2idx, num_labels)
	    y_test = convert_to_index(test_y, label2idx, num_labels)

	    return X_train, y_train, X_dev, y_dev, X_test, y_test, word2idx, label2idx
		'''


