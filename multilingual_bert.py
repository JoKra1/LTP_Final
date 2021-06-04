# Authors: Ella Collins, Manon Heinhuis & Joshua Krause
import time
import torch
import torch.nn as nn
import numpy as np
from helpers.twitter_data_loader import TwitterDataset, padding_collate_fn
from torch.utils.data import DataLoader 
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score

batch_size = 16
epochs = 10
num_hidden = 3
num_attention_heads = 12
num_labels = 3

"""
Optionally run on CUDA as discussed in https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------

def evaluate(model, dataset):
	model.eval()
	with torch.no_grad():
		correct = 0.0
		total = 0.0
		for batch in dataset:
			data, labels = batch
			data = data.to(device)

			y_pred = model(data).logits
			y_pred = torch.argmax(y_pred, dim=1)
			if not device == "cpu":
				y_pred = y_pred.cpu()
			
			correct += torch.sum(y_pred == labels).detach().numpy()
			total += len(labels)

	model.train()
	return correct/total

def train(model, train_data, val_data, epochs):
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

	for epoch in range(epochs):
		total_loss = 0.0
		n = 0
		for i, batch in enumerate(train_data):
			data, labels = batch
			data = data.to(device)
			labels = labels.to(device)

			optimizer.zero_grad()

			output = model(data, labels= labels)

			loss = output.loss

			total_loss += loss.item()

			loss.backward()

			optimizer.step()

			n += 1
		print(f"Average trainings-loss {total_loss/n}")
		acc = evaluate(model, val_data)
		print("[Epoch %d] Accuracy (validation): %.4f" %(epoch, acc))


# ----------------------------------------------------------

if __name__ == "__main__":
	print(device)
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
	val_dataset = TwitterDataset("data/val_merged.csv", tokenizer)
	val_data = DataLoader(val_dataset,
		collate_fn = padding_collate_fn,
		batch_size = batch_size)
	print("val loaded")
	"""
	test_dataset = TwitterDataset("data/test_merged.csv", tokenizer)
	test_data = DataLoader(test_dataset,
		collate_fn = padding_collate_fn,
		batch_size = batch_size)
	print("Data has loaded")
	"""

	# load the multilingual Bert model
	model = BertForSequenceClassification.from_pretrained(pretrained, 
		num_labels = num_labels,
		num_hidden_layers= num_hidden,
	    num_attention_heads= num_attention_heads,
	    output_attentions=True)

	# Send device to gpu
	model.to(device)
	print("Training the model.")
	model.train()
	train(model, train_data, val_data, epochs)
	#evaluate(model, val_data)
	print("Training is complete.")

