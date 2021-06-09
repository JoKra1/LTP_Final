import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#The RNN class, which takes the input dimensions vocab_size, and outputs a softmax for the amount of classes
class RNN(nn.Module):
	
	def __init__(self, input_dim, output_dim,
				 padding_idx=0,gru_dim=250,embedding_size=768,
				 dropout_prob=0.1,preEmbeddings=None):
		
		super(RNN, self).__init__()
		self.gruSize = gru_dim
		"""
		Load in pre-trained embeddings if provided.
		"""
		if preEmbeddings is not None:
			self.emb = nn.Embedding.from_pretrained(preEmbeddings,freeze=True,padding_idx=padding_idx)
		else:
			self.emb = nn.Embedding(input_dim, embedding_size,padding_idx=padding_idx) #The embeddings has 768
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
