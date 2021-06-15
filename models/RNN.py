import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#The RNN class, which takes the input dimensions vocab_size, and outputs a softmax for the amount of classes
class RNN(nn.Module):
	"""
	For the RNN we rely to a large degree on the architecture provided
	in the NLP pytorch book chapter 6 that was made available during
	the course. However, instead of using two linear layers on top we
	use a single one.
	"""
	def __init__(self, input_dim, output_dim,
				 padding_idx=0,gru_dim=250,embedding_size=768,
				 dropout_prob=0.1,preEmbeddings=None):
		
		super(RNN, self).__init__()
		self.gruSize = gru_dim
		
		"""
		Setup embeddings. Basically initializes from pre-trained embeddings or
		initializes embeddings randomly to be learned during the classification task
		(as we did during lab 3.)

		Source: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
		"""
		if preEmbeddings is not None:
			self.emb = nn.Embedding.from_pretrained(preEmbeddings,freeze=True,padding_idx=padding_idx)
		else:
			self.emb = nn.Embedding(input_dim, embedding_size,padding_idx=padding_idx)
		self.gru = nn.GRU(embedding_size, gru_dim)
		self.drop = nn.Dropout(p=dropout_prob) 
		self.lin = nn.Linear(gru_dim, output_dim)

	def forward(self, x):
		"""
		We want to exclude all padded zeros from the RNN later (see links below).
		So we just count how many non-padding tokens there are in here.
		Apparently, the length vector needs to be on cpu! It crashed
		without the explicit cast implemented below.
		"""

		x_valid = torch.count_nonzero(x,dim=1)
		x_valid = x_valid.cpu()

		"""
		Embedding step can happen on entire batch
		since we explicitly ignore the padding_idx.
		"""
		x = self.emb(x)

		"""
		Now use packing, as discussed in the links below!

		We do not sort the batch, as done in some of the examples below, so
		we need to explicitly set enforce_sorted to False, otherwise the model
		crashes (see also documentation).

		Also, due to the format of x matching ([batch_size,max_len(batch),embedding_dim]), we
		need to set batch_first=True, as discussed in the documentation.

		https://discuss.pytorch.org/t/simple-working-example-how-to-use-packing-for-variable-length-sequence-inputs-for-rnn/2120
		https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html
		https://discuss.pytorch.org/t/how-to-get-the-output-at-the-last-timestep-for-batched-sequences/12057/3
		"""

		x = nn.utils.rnn.pack_padded_sequence(x,x_valid,batch_first=True,enforce_sorted=False)

		"""
		We do not provide an initial hidden state and instead rely
		on the default, as mentioned in the docs.
		See: https://pytorch.org/docs/stable/generated/torch.nn.GRU.html
		"""

		o,h = self.gru(input=x)

		"""
		According to the reference below, the last output in o
		is equal to the last hidden state for a given tweet.
		See: https://discuss.pytorch.org/t/how-to-get-the-output-at-the-last-timestep-for-batched-sequences/12057/3
		We checked the docs and they also support this.
		See: https://pytorch.org/docs/stable/generated/torch.nn.GRU.html
		Thus we can simply work with h, which
		first needs to be brought into the correct shape, since it has
		a third dimension set to 1 as discussed in the documentation.
		"""
		
		h_shaped = h.view(-1,self.gruSize)
		h_shaped = self.drop(h_shaped)

		"""
		Finally, we use one additional linear layer to bring
		the output from the last hidden state to class level, as done
		in the NLP book (chapter 6).

		Before that we use a relu activation function on the outputs
		from the GRU to add some further non-linearities.

		No softmax, since we use cross-entropy directly based on the
		recommendations in the NLP book.
		"""

		x = self.lin(F.relu(h_shaped))
		return x
