import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader 
from transformers import BertTokenizer
from helpers.twitter_data_loader import TwitterDataset, padding_collate_fn, idx2cat

batch_size = 32
epochs = 10

#The RNN class, which takes the input dimensions vocab_size, and outputs a softmax for the amount of classes
class RNN(nn.Module):
	
	#Implement freeze for initialising embeddings if there are embeddings inputted as args
	def __init__(self, input_dim, output_dim,padding_idx=0):
		
		super(RNN, self).__init__()

		self.emb = nn.Embedding(input_dim, 300,padding_idx=padding_idx) #The embeddings has 300 ## A hacky math method on the  internet said 2000 could be something
		self.gru = nn.GRU(300, 250) 
		self.lin = nn.Linear(250, output_dim)

	def forward(self, x):

		x = self.emb(x)
		x = self.gru(x)
		x = self.lin(x[0])
		x = nn.functional.softmax(x, dim=1)

		return x


if __name__ == "__main__":
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
		collate_fn=padding_collate_fn,
		batch_size = batch_size)
	print("train loaded")
	val_dataset = TwitterDataset("data/val_merged.csv", tokenizer, max_size=100)
	val_data = DataLoader(val_dataset,
		collate_fn=padding_collate_fn,
		batch_size = batch_size)
	print("val loaded")
	test_dataset = TwitterDataset("data/test_merged.csv", tokenizer, max_size=100)
	test_data = DataLoader(test_dataset,
		collate_fn=padding_collate_fn,
		batch_size = batch_size)
	print("Data has loaded")

	model = RNN(input_dim=tokenizer.vocab_size, output_dim=3)

	criterion = nn.NLLLoss()
	optimizer = optim.Adam(params=model.parameters())

	#  Actual training
	for epoch in range(epochs):
		epoch_loss = 0
		for i, batch in enumerate(train_data):
			data, labels = batch

			optimizer.zero_grad()
			
			output = model(data)

			loss = criterion(output, labels)

			loss.backward()
			optimizer.step()
			
			epoch_loss += loss.item()
			
		print("  End epoch {}. Average loss {}".format(epoch, epoch_loss/num_batches))












