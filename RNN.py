import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader 
from transformers import BertTokenizer
from helpers.twitter_data_loader import TwitterDataset, padding_collate_fn, idx2cat, SupportedFormat
from sklearn.metrics import accuracy_score

batch_size = 32
epochs = 10

"""
Optionally run on CUDA as discussed in https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#The RNN class, which takes the input dimensions vocab_size, and outputs a softmax for the amount of classes
class RNN(nn.Module):
	
	#Implement freeze for initialising embeddings if there are embeddings inputted as args
	def __init__(self, input_dim, output_dim,padding_idx=0,gru_dim=250,embedding_size=300):
		
		super(RNN, self).__init__()
		self.gruSize = gru_dim
		self.emb = nn.Embedding(input_dim, embedding_size,padding_idx=padding_idx) #The embeddings has 300
		self.gru = nn.GRU(embedding_size, gru_dim) 
		self.lin = nn.Linear(gru_dim, output_dim)

	def forward(self, x):
		"""
		We want to exclude all padded zeros from the RNN later.
		So we just count how many non-padding tokens there are in here.
		Apparently, the length vector needs to be on cpu! It crashed
		without the explicit ensurance policy implemented below.
		"""
		x_valid = torch.count_nonzero(x,dim=1)
		if not device == "cpu":
			x_valid = x_valid.cpu()

		"""
		Embedding step can happen on entire batch
		since we explicitly ignore the padding_idx.
		"""
		x = self.emb(x)

		"""
		Now use packing!
		https://discuss.pytorch.org/t/simple-working-example-how-to-use-packing-for-variable-length-sequence-inputs-for-rnn/2120
		https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html
		"""

		x = nn.utils.rnn.pack_padded_sequence(x,x_valid,batch_first=True,enforce_sorted=False)

		o,h = self.gru(input=x)
		"""
		# The last output in o is equal to the last hidden state
		# for a given tweet.
		# See: https://discuss.pytorch.org/t/how-to-get-the-output-at-the-last-timestep-for-batched-sequences/12057/3
		# Thus we can simply work with h, which according to the docs
		# is the last hidden state:
		# https://pytorch.org/docs/stable/generated/torch.nn.GRU.html
		"""
		# Uncomment those to verify!
		# o = nn.utils.rnn.pad_packed_sequence(o,batch_first=True)
		# last = o[0][0,x_valid[0]-1,:]

		"""
		Finally, we use one additional linear layer to bring
		the output from the hidden state to class level.
		No softmax, since we use cross-entropy based on the
		recommendations in the NLP book.

		See: https://discuss.pytorch.org/t/linear-layers-on-top-of-lstm/512
		and: https://discuss.pytorch.org/t/how-to-get-the-output-at-the-last-timestep-for-batched-sequences/12057/3
		"""
		x = self.lin(h.view(-1,self.gruSize))
		return x


def train(model,train,val,epochs):
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(params=model.parameters())
	accuracies = []

	#  Actual training
	for epoch in range(epochs):
		epoch_loss = 0
		n = 0
		print(f"Epoch {epoch}")
		for batch in train:
			data, labels = batch
			data = data.to(device)
			labels = labels.to(device)

			optimizer.zero_grad()
			
			output = model(data)

			loss = criterion(output, labels)

			loss.backward()
			optimizer.step()
			
			epoch_loss += loss.item()
			n += 1
		print(f"Average trainings loss {epoch_loss/n}")
		acc_val = evaluate(model,val)
		print(f"Validation accuracy: {acc_val}")
		accuracies.append(acc_val*100)
	return accuracies

def evaluate(model,val):
	#Validation of model
	print("Validation")
	epoch_acc = 0
	n = 0
	model.eval()
	for batch in val:
		val_data, val_labels = batch
		val_data = val_data.to(device)

		output_val = model(val_data)
		output_val_classes = torch.max(output_val, dim=1).indices
		if not device == "cpu":
			output_val_classes = output_val_classes.cpu()

		epoch_acc += accuracy_score(val_labels.numpy(), output_val_classes.numpy())
		n += 1
	model.train()
	return epoch_acc/n

if __name__ == "__main__":
	print(device)
	np.random.seed(0)
	# load tokenizer
	pretrained = 'bert-base-multilingual-cased'
	tokenizer = BertTokenizer.from_pretrained(pretrained)
	tokenizer.do_basic_tokenize = False
	print("loading data...")

	# load data
	train_dataset = TwitterDataset("data/train_merged.csv", tokenizer,format=SupportedFormat.RNN)
	train_data = DataLoader(train_dataset,
		shuffle = True,
		collate_fn=padding_collate_fn,
		batch_size = batch_size)
	print("train loaded")
	val_dataset = TwitterDataset("data/val_merged.csv", tokenizer,format=SupportedFormat.RNN)
	val_data = DataLoader(val_dataset,
		collate_fn=padding_collate_fn,
		batch_size = batch_size)
	print("val loaded")
	test_dataset = TwitterDataset("data/test_merged.csv", tokenizer,format=SupportedFormat.RNN)
	test_data = DataLoader(test_dataset,
		collate_fn=padding_collate_fn,
		batch_size = batch_size)
	print("Data has loaded")

	# Setup model
	model = RNN(input_dim=tokenizer.vocab_size, output_dim=3)
	model.to(device)
	model.train()

	# Train
	accuracies = train(model,train_data,val_data,epochs=epochs)

	#Write the accuracies to a file
	change = "What's this" #Really what is this and what do we want it to be
	with open("RNNaccuracies.txt", "a+") as file:
		file.write("%s,%s\n" %(change, ",".join(map(str,accuracies))))
	print("Written to file.")
