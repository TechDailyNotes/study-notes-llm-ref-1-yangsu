import numpy as np
import torch

### ML

# attention for single-batch
# x: output from token embedding + pos encoding, shape (seq_len, embed_dim)
# w_q, w_k, w_v: shape (embed_dim, dim_q/k/v)
# dim_q = dim_k, dim_v is the output embed_dim
# mask shape (seq_len x seq_len), lower-triangle matrix with zero and ones
def attention(x, w_q, w_k, w_v, mask):
	q = x @ w_q # q shape (seq_len x dim_q/k)
	k = x @ w_k # k shape (seq_len x dim_q/k)
	v = x @ w_v # v shape (seq_len x dim_v)
	k_t = k.T # k_t shape (dim_q/k x seq_len)
	qk_t = q @ k_t # qk_t shape (seq_len x seq_len)
	qk_t /= sqrt(q.shape[-1]) # scale down by dim_q/k, yes
	qk_t.masked_fill(mask == 0, float("-inf")) # mask -inf to softmax to 0
	attn = softmax(qk_t, dim=-1) # along last seq_len dim
	output = qk_t @ v # output shape (seq_len x dim_v)
	return output

# attention for multi-batches
# X_batch: shape (batch_size x seq_len x embed_dim)
# w_q, w_k, w_v: same as above
# mask_batch: shape (batch_size x seq_len x seq_len)
def batch_attention(X_batch, w_q, w_k, w_v, mask_batch):
	Q = X_batch @ w_q # Q shape (batch_size x seq_len x dim_q/k)
	K = X_batch @ w_k # K shape (batch_size x seq_len x dim_q/k)
	V = X_batch @ w_v # V shape (batch_size x seq_len x dim_v)
	K_t = K.T # K_t shape (batch_size x dim_q/k x seq_len)
	K_t = K.permute(0, 2, 1) # equivalent, batch_size dim does NOT transpose
	QK_t = torch.bmm(Q, K_t) # QK_t shape (batch_size x seq_len x seq_len)
	QK_t /= sqrt(Q.shape[-1])
	QK_t.masked_fill(mask_batch == 0, float("-inf"))
	attn = softmax(QK_t, dim=-1)
	output = torch.bmm(QK_t, V) # output shape (batch_size x seq_len x dim_v)
	return output

# multihead attention for multi-batches
# X_batch: same as above
# W_q, W_k, W_v: hidden dim multiplied by head (embed_dim x num_head x dim_q/k/v)
# attn has an additional shape of num_head (batch_size x seq_len x seq_len x num_head)
# same mask_batch is applied to all heads
# output is concat by num_head, shape (batch_size x seq_len x (dim_v x num_head))
def multihead_batch_attention(X_batch, W_q, W_k, W_v, mask_batch):
	pass

# xi: scalar
# softmax scales all xi so that they sum to 1
def softmax_i(xi, [x1, ..., xi-1, xi+1, ...]):
	return np.exp(xi) / np.sum(j=1->n, j!=i: np.exp(xj))

# X: shape (..., softmax_dim)
# output: same shape as X
def softmax(X, softmax_dim=-1):
	# why (X - max(X)): to prevent overflow to infinity
	X_exp = np.exp(X - np.max(X, dim=softmax_dim))
	return X_exp / np.sum(X_exp, dim=softmax_dim)

# x: scalar
# == logsitic (activation) function
# output: range (0, 1)
# derivative: sigmoid(x) * (1 - sigmoid(x))
def sigmoid(x):
	return 1 / (1 + np.exp(- x))

# p: the predicted label probability (range 0 to 1), shape (num_examples)
# y: the true label (either 0 or 1), shape (num_examples)
# BCE loss measures the difference of these two
# minimizing BCE loss == maximizing the negative log probability (likelihood)
def binary_cross_entropy_loss(p, y):
	loss_class_0 = y * log(p) # multiply by each example and sum up
	loss_class_1 = (1 - y) * log(1 - p)
	return - (loss_class_0 + loss_class_1)

# X: output from fully connected layer, shape (num_examples x num_classes)
# y: shape (num_examples x num_classes)
def cross_entropy_loss(p, y):
	class_dim = y.shape[-1]
	return - np.sum(y * np.log(p), dim=class_dim)

# ROC: receiver operating characteristic
# AUC: area under (ROC) curve
# suppose there are 2 classes, the AUC is constructed by
# 1. sort the predictions in descending order by probability belonging to class 1
# 2. take the ordered true labels, suppose there are m class 1 and n class 0
# 3. draw a (m, n) grid, start from axis (0, 0), go through the label list
# 4. go up (along m) when we encounter class 1, and go right (along n) for class 0
# ideal AUC: full - which means all class 1 happens before 0
# == fraction of pair of objects of the form (object of class 1, object of class 0)
def auc_roc_score(X, y):
	pass

def to_onehot(label, n_classes):
	N = label.shape[0]
	onehot = np.zeros((N, n_classes))
	for i in range(N):
		onehot[i][label[i]] = 1
	return onehot

### Data Structure

# find kth smallest
def quick_select(nums, k):
	pivot = random.choice(nums)
	l, m, r = [], [], []
	for i in nums:
		if i < pivot: # change to ">" for finding kth largest
			l.append(i)
		elif i == pivot:
			m.append(i)
		else:
			r.append(i)
	if k <= len(l):
		return quick_select(l, k)
	elif len(l) + len(m) < k:
		return quick_select(r, k - len(l) - len(m))
	else:
		return pivot

def partition(nums, lo, hi):
	if lo >= hi:
		return -1
	pivot = lo
	l, r = pivot + 1, hi
	while l <= r:
		if nums[l] < nums[pivot]:
			l += 1
		elif nums[r] >= nums[pivot]:
			r -= 1
		else:
			swap(nums[l], nums[r])
	# r becomes the smaller one after while loop end
	# hence to ensure correctness on the left side, swap and return r
	swap(nums[pivot], nums[r])
	return r # everything before AND at r is correctly partitioned

def quick_sort(nums, lo, hi):
	if lo >= hi:
		return
	swap(nums[random.randint(lo, hi)], nums[lo])
	pivot = partition(nums, lo, hi)
	quick_sort(nums, lo, pivot)
	quick_sort(nums, pivot + 1, hi)

class Heap:
	pass

def heap_sort(nums):
	pass
