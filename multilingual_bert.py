# Authors: Ella Collins, Manon Heinhuis & Joshua Krause
import time
import torch
import torch.nn as nn
import numpy as np
from helpers import twitter_data_loader
from dataloading import TwitterDataset, padding_collate_fn
from torch.utils.data import DataLoader 
from transformers import BertTokenizer, BertForMultipleChoice

batch_size = 32
epochs = 10
num_hidden = 1
num_attention_heads = 12
num_labels = 3

# ----------------------------------------------------------

def evaluate(model, dataset):
	model.eval()
	with torch.no_grad():
		for batch in dataset:
			data, labels = batch
			y_pred = model(data).logits
			print("data: ", data)
			print("labels: ", labels)

	model.train()
	return

def train(model, train_data, val_data, epochs):
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

	for epoch in range(epochs):
		total_loss = 0.0
		for i, batch in enumerate(train_data):
			data, labels = batch
			optimizer.zero_grad()

			output = model(data, labels= labels)
			y_pred = output.logits

			loss = output.loss

			total_loss += loss.item()

			loss.backward()

			optimizer.step()

		acc = evaluate(model, val_data)
		print("[Epoch %d] Accuracy (validation): %.4f" %(epoch, acc))


# ----------------------------------------------------------

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

	# load the multilingual Bert model
	model = BertForMultipleChoice.from_pretrained(pretrained, 
		num_labels = num_labels,
		num_hidden_layers= num_hidden,
	    num_attention_heads= num_attention_heads,
	    output_attentions=True)

	print("Training the model.")
	#train(model, train_data, val_data, epochs)
	evaluate(model, val_data)
	print("Training is complete.")


