import torch
import numpy as np
import torch.nn as nn

batch_size = 32


class RNN(nn.Module):
    
    #Implement freeze for initialising embeddings if there are embeddings inputted as args
    def __init__(self, input_dim, output_dim):
        """
        Args:
            input_dim (int): size of the input features ( vocab size)
            output_dim (int): number of classes
        """
        
        super(RNN, self).__init__()

        self.emb = nn.Embedding(input_dim, 2000) #The embeddings has 300 ## A hacky math method on the  internet said 2000 could be something
        self.gru = nn.GRU(2000, 1000) # Haven't looked into good amount of nodes
        self.den = nn.Dense(1000, output_dim)

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
        x = F.softmax(x, dim=1)

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

if __name__ == "__main__":
	torch.manual_seed(0)
	np.random.seed(0)

	# load tokenizer
	pretrained = 'bert-base-multilingual-cased'
	tokenizer = BertTokenizer.from_pretrained(pretrained)
	tokenizer.do_basic_tokenize = False
	print("loading data...")

	# load data
	train_dataset = TwitterDataset("data/train_merged.csv", tokenizer)
	train_data = DataLoader(train_dataset,
		shuffle = True,
		collate_fn = padding_collate_fn,
		batch_size = batch_size)
	print("train loaded")
	val_dataset = TwitterDataset("data/val_data.csv", tokenizer)
	val_data = DataLoader(val_dataset,
		collate_fn = padding_collate_fn,
		batch_size = batch_size)
	print("val loaded")
	test_dataset = TwitterDataset("data/test_merged.csv", tokenizer)
	test_data = DataLoader(test_dataset,
		collate_fn = padding_collate_fn,
		batch_size = batch_size)
	print("Data has loaded")

	model = RNN(input_dim=tokenizer.vocab_size, output_dim=3)

	criterion = nn.NLLLoss()
	optimizer = optim.Adam(params=model.parameters())

	#  Actual training
	''' Though I don't actually know whether this is how that's done? I miss  tensorflow.
	num_batches = len(X_train) // batch_size
	print("#Batch size: {}, num batches: {}".format(size_batch, num_batches))
	for epoch in range(10):
	    epoch_loss = 0
	    for batch in range(num_batches):
	        batch_begin = batch*size_batch
	        batch_end = (batch+1)*(size_batch)
	        X_data = X_train[batch_begin:batch_end]
	        y_data = y_train[batch_begin:batch_end]
	        
	        y_tensor = torch.tensor(y_data, dtype=torch.int64)
	        optimizer.zero_grad()
	        
	        y_pred = model(X_tensor)
	#        print("#Y_pred")
	#        tensor_desc(y_pred)
	        loss = criterion(y_pred, y_tensor)
	#        print("#Loss: {}".format(loss))
	    
	#        model.print_params()
	        loss.backward()
	        optimizer.step()
	#        model.print_params()
	        
	        epoch_loss += loss.item()
	        
	    print("  End epoch {}. Average loss {}".format(epoch, epoch_loss/num_batches))
	'''
















