import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader 
from transformers import BertTokenizer
from helpers.twitter_data_loader import TwitterDataset, padding_collate_fn, idx2cat, SupportedFormat
from sklearn.metrics import accuracy_score

batch_size = 64
epochs = 20

"""
Optionally run on CUDA as discussed in https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#The RNN class, which takes the input dimensions vocab_size, and outputs a softmax for the amount of classes
class RNN(nn.Module):
	
	#Implement freeze for initialising embeddings if there are embeddings inputted as args
	def __init__(self, input_dim, output_dim,padding_idx=0,gru_dim=250,embedding_size=300,dropout_prob=0.1):
		
		super(RNN, self).__init__()
		self.gruSize = gru_dim
		self.emb = nn.Embedding(input_dim, embedding_size,padding_idx=padding_idx) #The embeddings has 300
		self.gru = nn.GRU(embedding_size, gru_dim)
		self.drop = nn.Dropout(p=dropout_prob) 
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
		# last = o[0][[i for i in range(batch_size)],[l-1 for l in x_valid],:]

		"""
		Bring h into the correct form and apply dropout.
		See: https://discuss.pytorch.org/t/how-to-get-the-output-at-the-last-timestep-for-batched-sequences/12057/3
		"""
		
		h_shaped = h.view(-1,self.gruSize)
		h_shaped = self.drop(h_shaped)

		"""
		Finally, we use one additional linear layer to bring
		the output from the hidden state to class level.
		Before that we use a relu activation function on the outputs
		from the GRU.
		No softmax, since we use cross-entropy based on the
		recommendations in the NLP book.

		See: https://discuss.pytorch.org/t/linear-layers-on-top-of-lstm/512
		"""

		x = self.lin(F.relu(h_shaped))
		return x


def train(model,train,val,epochs,sub_evals=None):
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(params=model.parameters())
	accuracies = []
	sub_accuracies =[]
	if sub_evals:
		for sub in sub_evals:
			sub_accuracies.append([])
	#  Actual training
	for epoch in range(epochs):
		epoch_loss = 0
		n = 0
		print(f"Epoch {epoch}")
		for batch in tqdm(train):
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
		if sub_evals:
			for index,sub in enumerate(sub_evals):
				sub_acc = evaluate(model,sub)
				sub_accuracies[index].append(sub_acc*100)
				print(f"Validation sub-accuracy: {index}: {sub_acc}")
	return accuracies,sub_accuracies

def evaluate(model,val):
	#Validation of model
	correct = 0
	n = 0
	model.eval()
	for batch in val:
		val_data, val_labels = batch
		val_data = val_data.to(device)

		output_val = model(val_data)
		output_val_classes = torch.max(output_val, dim=1).indices

		for pred,true in zip(output_val_classes.tolist(),
							 val_labels.tolist()):
			if pred == true:
				correct += 1
			n += 1
	model.train()
	return correct/n

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

	# load sub-eval sets (per language)
	val_eng = TwitterDataset("data/eng_val.csv", tokenizer,format=SupportedFormat.RNN)
	val_eng = DataLoader(val_eng,
		collate_fn=padding_collate_fn,
		batch_size = batch_size)
	
	val_rus = TwitterDataset("data/rus_val.csv", tokenizer,format=SupportedFormat.RNN)
	val_rus = DataLoader(val_rus,
		collate_fn=padding_collate_fn,
		batch_size = batch_size)

	val_ger = TwitterDataset("data/ger_val.csv", tokenizer,format=SupportedFormat.RNN)
	val_ger = DataLoader(val_ger,
		collate_fn=padding_collate_fn,
		batch_size = batch_size)

	"""
	test_dataset = TwitterDataset("data/test_merged.csv", tokenizer,format=SupportedFormat.RNN)
	test_data = DataLoader(test_dataset,
		collate_fn=padding_collate_fn,
		batch_size = batch_size)
	print("Data has loaded")
	"""

	### Model optimization ###
	GRU_sizes = [100,250,500]
	dropout_probs = [0.1,0.2,0.3,0.4,0.5,0.7,1.0]
	sub_evals =[val_eng,val_rus,val_ger]
	sub_ids = ["eng","rus","ger"]

	for gru_size in GRU_sizes:

		for dropout_prob in dropout_probs:
			# Setup model
			model = RNN(input_dim=tokenizer.vocab_size, output_dim=5,
						gru_dim=gru_size,dropout_prob=dropout_prob)
			model.to(device)
			model.train()

			# Train and collect overall + sub accuracies
			accuracies,sub_accuracies = train(model,train_data,val_data,epochs=epochs,sub_evals=sub_evals)

			# Write the accuracies to a file
			change = f"Size: {gru_size} Dropout: {dropout_prob}"
			with open("RNNaccuracies.txt", "a+") as file:
				file.write("%s,%s\n" %(change, ",".join(map(str,accuracies))))
			
			# Write sub-accuracies
			for index, identifier in enumerate(sub_ids):
				sub_acc = sub_accuracies[index]
				with open(f"RNNaccuracies_{identifier}.txt", "a+") as file:
					file.write("%s,%s\n" %(change, ",".join(map(str,sub_acc))))
			
			print("Written to file.")
