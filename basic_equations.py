import numpy as np

# x: np array
def softmax(x):
	# - max: to prevent from infinity
	exp_x = np.exp(x - np.max(x))
	return exp_x / np.sum(exp_x)

# X: output from fully connected layer (num_examples x num_classes)
# y: labels (num_examples x 1)
def cross_entropy(X, y):
	N = y.shape[0]
	p = softmax(X)
	loss = - np.sum(y * np.log(p)) / N
	return loss

def to_onehot(label, n_classes):
	N = label.shape[0]
	onehot = np.zeros((N, n_classes))
	for i in range(N):
		onehot[i][label[i]] = 1
	return onehot