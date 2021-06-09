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
from models.RNN import RNN
from train_embeddings import convertEmbeddings

batch_size = 128
epochs = 20

"""
Optionally run on CUDA as discussed in https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
		if not device == "cpu":
			output_val_classes = output_val_classes.cpu()
		correct += torch.sum(output_val_classes == val_labels).detach().numpy()
		n += len(val_labels)
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
	
	### Model optimization: Random embeddings ###
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
	
	### Model optimization: Pretrained embeddings ###
	pretrainedEmbeddings = convertEmbeddings("embeddings/w2v.model",tokenizer)

	for gru_size in GRU_sizes:

		for dropout_prob in dropout_probs:
			# Setup model
			model = RNN(input_dim=tokenizer.vocab_size, output_dim=5,
						gru_dim=gru_size,dropout_prob=dropout_prob,preEmbeddings=pretrainedEmbeddings)
			model.to(device)
			model.train()

			# Train and collect overall + sub accuracies
			accuracies,sub_accuracies = train(model,train_data,val_data,epochs=epochs,sub_evals=sub_evals)

			# Write the accuracies to a file
			change = f"Size: {gru_size} Dropout: {dropout_prob}"
			with open("RNNPREaccuracies.txt", "a+") as file:
				file.write("%s,%s\n" %(change, ",".join(map(str,accuracies))))
			
			# Write sub-accuracies
			for index, identifier in enumerate(sub_ids):
				sub_acc = sub_accuracies[index]
				with open(f"RNNPREaccuracies_{identifier}.txt", "a+") as file:
					file.write("%s,%s\n" %(change, ",".join(map(str,sub_acc))))
			
			print("Written to file.")
