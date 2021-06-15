import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
	"""
	Code used to generate accuracy plots over the validation set.
	"""
	acc_values = []
	labels = []
	x = [1,2]
	with open("accuracies.txt", "r", newline="") as file:
	
		for line in file: 
			split = line.split(',')
			acc_values.append(np.array(split[1:]).astype(int))
			labels.append(split[0])

	x = np.arange(len(acc_values))

	for idx in range(len(acc_values)):
		plt.plot(x, acc_values[idx], label = labels[idx])
		
	plt.xlabel("epoch")
	plt.ylabel("accuracy")
	plt.legend()
	plt.savefig("bert_acc_graph.png")