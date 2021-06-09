# Authors: Ella Collins, Manon Heinhuis & Joshua Krause
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from helpers.twitter_data_loader import TwitterDataset, padding_collate_fn, idx2cat
from torch.utils.data import DataLoader 
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from sklearn.metrics import accuracy_score

batch_size = 128
epochs = 20
num_attention_heads = 12
num_labels = len(idx2cat)

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

def train(model, train_data, val_data, epochs, sub_evals=None):
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
	accuracies = []
	sub_accuracies =[]
	if sub_evals:
		for _ in sub_evals:
			sub_accuracies.append([])
	for epoch in range(epochs): 
		total_loss = 0.0
		n = 0
		for i, batch in enumerate(tqdm(train_data)):
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
		acc = evaluate(model, val_data)*100
		print("[Epoch %d] Accuracy (validation): %.2f" %(epoch, acc))
		accuracies.append(acc)
		if sub_evals:
			for index,sub in enumerate(sub_evals):
				sub_acc = evaluate(model,sub)
				sub_accuracies[index].append(sub_acc*100)
				print(f"Validation sub-accuracy: {index}: {sub_acc}")
	return accuracies,sub_accuracies


# ----------------------------------------------------------

if __name__ == "__main__":
	print(device)
	torch.manual_seed(0)
	np.random.seed(0)

	# load tokenizer
	pretrained = 'bert-base-multilingual-cased'
	tokenizer = BertTokenizer.from_pretrained(pretrained)
	tokenizer.do_basic_tokenize = False

	# load data
	print("loading data...")
	train_dataset = TwitterDataset("data/train_merged.csv", tokenizer, max_size =100)
	train_data = DataLoader(train_dataset,
		shuffle = True,
		collate_fn = padding_collate_fn,
		batch_size = batch_size)
	
	val_dataset = TwitterDataset("data/val_merged.csv", tokenizer, max_size =100)
	val_data = DataLoader(val_dataset,
		collate_fn = padding_collate_fn,
		batch_size = batch_size)
	"""
	test_dataset = TwitterDataset("data/test_merged.csv", tokenizer, max_size =100)
	test_data = DataLoader(test_dataset,
		collate_fn = padding_collate_fn,
		batch_size = batch_size)
	"""
	print("Data has loaded.")

	# load sub-eval sets (per language)
	val_eng = TwitterDataset("data/eng_val.csv", tokenizer,max_size=100)
	val_eng = DataLoader(val_eng,
		collate_fn=padding_collate_fn,
		batch_size = batch_size)
	
	val_rus = TwitterDataset("data/rus_val.csv", tokenizer)
	val_rus = DataLoader(val_rus,
		collate_fn=padding_collate_fn,
		batch_size = batch_size)

	val_ger = TwitterDataset("data/ger_val.csv", tokenizer)
	val_ger = DataLoader(val_ger,
		collate_fn=padding_collate_fn,
		batch_size = batch_size)
	
	print("Loaded sub-eval sets.")

	### Model optimization ###
	num_hiddens = [1,5,10,12]
	dropout_probs = [0.1,0.2,0.3,0.4,0.5,0.7,1.0]
	sub_evals =[val_eng,val_rus,val_ger]
	sub_ids = ["eng","rus","ger"]

	for num_hidden in num_hiddens:

		for dropout_prob in dropout_probs:
			### Untrained BERT ###
			config = BertConfig.from_pretrained(pretrained)
			config.num_labels = num_labels
			config.num_hidden_layers = num_hidden
			config.num_attention_heads = num_attention_heads
			config.hidden_dropout_prob = dropout_prob
			config.output_attentions = True

			# Setup an untrained model
			model = BertForSequenceClassification(config)

			# Send device to gpu
			model.to(device)
			model.train()
			accuracies,sub_accuracies = train(model, train_data, val_data, epochs,sub_evals=sub_evals)
			# Write the accuracies to a file
			change = f"n_hidden: {num_hidden} Dropout: {dropout_prob}"
			with open("BERTaccuracies.txt", "a+") as file:
				file.write("%s,%s\n" %(change, ",".join(map(str,accuracies))))
			
			# Write sub-accuracies
			for index, identifier in enumerate(sub_ids):
				sub_acc = sub_accuracies[index]
				with open(f"BERTaccuracies_{identifier}.txt", "a+") as file:
					file.write("%s,%s\n" %(change, ",".join(map(str,sub_acc))))
			
			print("Written to file.")


