import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader 
from transformers import BertTokenizer
from helpers.twitter_data_loader import TwitterDataset, padding_collate_fn, idx2cat

batch_size = 32

#The RNN class, which takes the input dimensions vocab_size, and outputs a softmax for the amount of classes
class RNN(nn.Module):
	
	#Implement freeze for initialising embeddings if there are embeddings inputted as args
	def __init__(self, input_dim, output_dim):
		
		super(RNN, self).__init__()

		self.emb = nn.Embedding(input_dim, 2000) #The embeddings has 300 ## A hacky math method on the  internet said 2000 could be something
		self.gru = nn.GRU(2000, 1000) 
		self.lin = nn.Linear(1000, output_dim)

	def forward(self, x):

		x = self.emb(x)
		x = self.gru(x)
		x = self.lin(x)
		x = F.softmax(x, dim=1)

		return x


if __name__ == "__main__":
	torch.manual_seed(0)
	np.random.seed(0)

	# load tokenizer
	pretrained = 'bert-base-multilingual-cased'
	tokenizer = BertTokenizer.from_pretrained(pretrained)
	tokenizer.do_basic_tokenize = False
	print("loading data...")

	# load data
	train_dataset = TwitterDataset("data/train_merged.csv", tokenizer, max_size=100)
	train_data = DataLoader(train_dataset,
		shuffle = True,
		collate_fn = padding_collate_fn,
		batch_size = batch_size)
	print("train loaded")
	val_dataset = TwitterDataset("data/val_merged.csv", tokenizer, max_size=100)
	val_data = DataLoader(val_dataset,
		collate_fn = padding_collate_fn,
		batch_size = batch_size)
	print("val loaded")
	test_dataset = TwitterDataset("data/test_merged.csv", tokenizer, max_size=100)
	test_data = DataLoader(test_dataset,
		collate_fn = padding_collate_fn,
		batch_size = batch_size)
	print("Data has loaded")

	model = RNN(input_dim=tokenizer.vocab_size, output_dim=3)

	criterion = nn.NLLLoss()
	optimizer = optim.Adam(params=model.parameters())

	#  Actual training
	dataT, labelT = train_data
	#Though I don't actually know whether this is how that's done? I miss tensorflow.
	num_batches = len(list(train_data)) // batch_size
	print(batch_size)
	print(train_data)
	print("#Batch size: {}, num batches: {}".format(batch_size, num_batches))
	for epoch in range(10):
		epoch_loss = 0
		for batch in range(num_batches):
			data, labels = batch
			batch_begin = batch*batch_size
			batch_end = (batch+1)*(batch_size)
			train_input_data = data[batch_begin:batch_end]
			train_label_data = labels[batch_begin:batch_end]
			
			X_tensor = torch.tensor(train_input_data, dtype=torch.float32)
			y_tensor = torch.tensor(train_label_data, dtype=torch.int64)
			optimizer.zero_grad()
			
			output = model(X_tensor)
	#        print("#Y_pred")
	#        tensor_desc(y_pred)
			loss = criterion(output, y_tensor)
	#        print("#Loss: {}".format(loss))
		
	#        model.print_params()
			loss.backward()
			optimizer.step()
	#        model.print_params()
			
			epoch_loss += loss.item()
			
		print("  End epoch {}. Average loss {}".format(epoch, epoch_loss/num_batches))












