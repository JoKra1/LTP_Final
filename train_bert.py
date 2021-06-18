import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from helpers.twitter_data_loader import TwitterDataset, padding_collate_fn, idx2cat
from torch.utils.data import DataLoader 
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from sklearn.metrics import accuracy_score
from train_embeddings import convertEmbeddings

"""
Basically copy of mbert file, just seperated for better overview.
"""

batch_size = 128 # 32 for 6 layer BERT
epochs = 10
num_attention_heads = 12 # This one needs to remain fixed!!
num_labels = len(idx2cat)

"""
Optionally run on CUDA as discussed in https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ----------- EVALUATE -----------

def evaluate(model, dataset):
	"""
	We used EC's submission for lab 4 as a starting point.
	We added the masking, which was recommended for the
	sequence classification model (see references below).
	"""
	model.eval()
	with torch.no_grad():
		correct = 0.0
		total = 0.0
		for batch in dataset:
			data, labels = batch

			# mask
			"""
			Masking for the attention mechanisms as also done in the
			transformer examples linked below.
			
			Source: https://huggingface.co/transformers/custom_datasets.html#fine-tuning-with-native-pytorch-tensorflow
			and: https://huggingface.co/transformers/model_doc/bert.html#transformers.BertForSequenceClassification
			and: https://huggingface.co/transformers/glossary.html#attention-mask
			"""
			mask = torch.zeros_like(data)
			mask[data != 0] = 1
			mask = mask.float()
			mask = mask.to(device)

			data = data.to(device)

			y_pred = model(data,attention_mask=mask).logits
			y_pred = torch.argmax(y_pred, dim=1)
			if not device == "cpu":
				y_pred = y_pred.cpu()
			
			correct += torch.sum(y_pred == labels).detach().numpy()
			total += len(labels)

	model.train()
	return correct/total

# ----------- TRAIN -----------

def train(model, train_data, val_data, epochs):
	"""
	We used EC's submission for lab 4 as a starting point.
	We added the masking, which was recommended for the
	sequence classification model (see references below).

	Generally, the routine is the one recommended in the
	pytorch example on network training (linked below) that was
	used throughout the entire course, with the adaptations recommended
	in the transformers documentation for the sequence classification BERT:

	Source: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#train-the-network
	and: https://huggingface.co/transformers/model_doc/bert.html#bertforsequenceclassification
	"""
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
	accuracies = []
	for epoch in range(epochs): 
		total_loss = 0.0
		n = 0
		for i, batch in enumerate(tqdm(train_data)):
			data, labels = batch
			
			# mask
			"""
			Masking for the attention mechanisms as also done in the
			transformer examples linked below.
			
			Source: https://huggingface.co/transformers/custom_datasets.html#fine-tuning-with-native-pytorch-tensorflow
			and: https://huggingface.co/transformers/model_doc/bert.html#transformers.BertForSequenceClassification
			and: https://huggingface.co/transformers/glossary.html#attention-mask
			"""

			mask = torch.zeros_like(data)
			mask[data != 0] = 1
			mask = mask.float()
			mask = mask.to(device)

			data = data.to(device)
			labels = labels.to(device)

			optimizer.zero_grad()

			output = model(data,attention_mask=mask,labels=labels)

			loss = output.loss

			total_loss += loss.item()

			loss.backward()

			optimizer.step()

			n += 1
		print(f"Average trainings-loss {total_loss/n}")
		acc = evaluate(model, val_data)*100
		print("[Epoch %d] Accuracy (validation): %.2f" %(epoch, acc))
		accuracies.append(acc)
	return accuracies


# ----------- MAIN -----------

if __name__ == "__main__":
	"""
	For the tokenizer loading, data loading/pre-processing 
	and initialization of the untrained/pre-trained models we
	rely on the steps in the code for lab 4, since we adapted the
	dataloading parts to work for our case here as well.
	"""
	print(device)
	torch.manual_seed(0)
	np.random.seed(0)

	# load tokenizer
	pretrained = 'bert-base-cased'
	tokenizer = BertTokenizer.from_pretrained(pretrained)
	tokenizer.do_basic_tokenize = False

	# ----------- LOAD DATA -----------

	# load data
	print("loading data...")
	train_dataset = TwitterDataset("data/eng_train.csv", tokenizer)
	train_data = DataLoader(train_dataset,
		shuffle = True,
		collate_fn = padding_collate_fn,
		batch_size = batch_size)
	
	val_dataset = TwitterDataset("data/eng_val.csv", tokenizer)
	val_data = DataLoader(val_dataset,
		collate_fn = padding_collate_fn,
		batch_size = batch_size)
	print("Data has loaded.")

	# ----------- EXPERIMENTS -----------
	
	### Model: Pre-trained BERT ###
	print("Training pretrained BERT")
	num_hiddens = [1,3,6] # No improvement for layer 6 model.
	dropout_probs = [0.1] # For comparison no fine-tuning here!

	for num_hidden in num_hiddens:

		for dropout_prob in dropout_probs:
			# Setup pre-trained BERT
			model = BertForSequenceClassification.from_pretrained(pretrained,
			num_labels = num_labels,
			num_hidden_layers=num_hidden,
			num_attention_heads=num_attention_heads,
			output_attentions=True)

			# Send device to gpu
			model.to(device)
			model.train()
			accuracies = train(model, train_data, val_data, epochs)
			# Write the accuracies to a file
			change = f"n_hidden: {num_hidden} Dropout: {dropout_prob}"
			with open("MONOBERTFINEaccuracies.txt", "a+") as file:
				file.write("%s,%s\n" %(change, ",".join(map(str,accuracies))))
			
			torch.save(model.state_dict(), f"FINEbert{num_hidden}_{dropout_prob}.pt")
			print("Written to file.")
