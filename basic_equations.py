import numpy as np
import torch
import torch.nn.functional as F

### ML

# attention for single-batch
# memory O(3 * (embed_dim ** 2 + embed_dim))
# some attention has bias term (LLaMA does not use, ChatGLM v1 uses)
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
	qk_t /= sqrt(dim_q/k) # scale down by dim_q/k, yes
	qk_t.masked_fill(mask == 0, float("-inf")) # mask -inf to softmax to 0
	attn = F.softmax(qk_t, dim=-1) # along last seq_len dim
	output = qk_t @ v # output shape (seq_len x dim_v)
	return output

# attention for multi-batches
# memory O(3 * (embed_dim ** 2 + embed_dim))
# X_batch: shape (batch_size x seq_len x embed_dim)
# w_q, w_k, w_v: shape (embed_dim, embed_dim)
# mask_batch: shape (batch_size x seq_len x seq_len)
def batch_attention(X_batch, w_q, w_k, w_v, mask_batch):
	Q = X_batch @ w_q # Q shape (batch_size x seq_len x embed_dim)
	K = X_batch @ w_k # K shape (batch_size x seq_len x embed_dim)
	V = X_batch @ w_v # V shape (batch_size x seq_len x embed_dim)
	K_t = K.T # K_t shape (batch_size x embed_dim x seq_len)
	K_t = K.permute(0, 2, 1) # equivalent, batch_size dim does NOT transpose
	QK_t = torch.matmul(Q, K_t) # QK_t shape (batch_size x seq_len x seq_len)
	QK_t /= sqrt(embed_dim)
	# QK_t += position_bias, shape (1 x seq_len x seq_len)
	QK_t = QK_t.masked_fill(mask_batch == 0, float("-inf"))
	attn = F.softmax(QK_t, dim=-1)
	output = torch.matmul(QK_t, V) # output shape (batch_size x seq_len x embed_dim)
	return output

# MHA (Multihead attention)
# let hidden_dim = embed_dim x num_head
# memory O(4 * (hidden_dim ** 2 + hidden_dim))
# time O(seq_len ** 2 * hidden_dim)
# X_batch: shape (batch_size x seq_len x hidden_dim)
# W_qkv: embed dim multiplied by head (hidden_dim x (3 * hidden_dim))
# W_o: linear layer after multihead output are computed (hidden_dim x hidden_dim)
# attn has an additional shape of num_head (batch_size x num_head x seq_len x seq_len)
# mask_batch: shape (batch_size x seq_len x seq_len)
# same mask_batch is applied to all heads
# output is concat by num_head, shape (batch_size x seq_len x hidden_dim)
# final output is computed going through W_o
def multi_head_attention(X_batch, W_qkv, W_o, mask_batch):
	QKV = X_batch @ W_qkv # QKV shape (batch_size x seq_len x (3 * hidden_dim))
	QKV = QKV.reshape(batch_size, seq_len, num_head, 3 * embed_dim)
	QKV = QKV.permute(0, 2, 1, 3) # QKV shape (batch_size x num_head x seq_len x (3 * embed_dim))
	Q, K, V = QKV.split(3, dim=-1)
	K_t = K.permute(-2, -1) # K_t shape (batch_size x num_head x embed_dim x seq_len)
	QK_t = torch.matmul(Q, K_t) # shape (batch_size x num_head x seq_len x seq_len)
	QK_t /= sqrt(embed_dim) # not hidden_dim! remember the distribution of this
	# QK_t += position_bias, shape (1 x seq_len x seq_len)
	mask_batch = mask_batch.unsqueeze(1) # (batch_size x 1 x seq_len x seq_len)
	QK_t = QK_t.masked_fill(mask_batch == 0, float("-inf"))
	attn = F.softmax(QK_t, dim=-1)
	values = torch.matmul(attn, V) # shape (batch_size x num_head x seq_len x embed_dim)
	values = values.permute(0, 2, 1, 3) # shape (batch_size x seq_len x num_head x embed_dim)
	values = values.reshape(batch_size, seq_len, hidden_dim) # shape (batch_size x seq_len x hidden_dim)
	output = values @ W_o # shape (batch_size x seq_len x hidden_dim)
	return output

# MQA (Multihead Query Attention)
# let hidden_dim = num_head x embed_dim
# W_q: multi-head, shape (batch_size x seq_len x hidden_dim)
# W_kv: single-head, shape (batch_size x seq_len x (2 * embed_dim))
def multi_query_attention(X_batch, W_q, W_kv, W_o, mask_batch):
	pass

# GQA (Grouped Query Attention)
# source: https://github.com/fkodom/grouped-query-attention-pytorch/blob/main/grouped_query_attention_pytorch/attention.py
# let hidden_dim = num_head x embed_dim, kv_dim = kv_heads x embed_dim
# then group_num = hidden_dim // kv_dim
# where kv_heads < num_head for query
# W_q: multi-head, shape (batch_size x seq_len x hidden_dim)
# W_kv: grouped-head, shape (batch_size x seq_len x (2 * kv_dim))
def grouped_query_attention(X_batch, W_q, W_kv, W_o, mask_batch):
	# after Q, K, V is calculated
	# Q shape (batch_size x num_head x seq_len x embed_dim)
	Q = Q.reshape(batch_size, group_num, kv_heads, seq_len, embed_dim)
	# then do the same for others
	# K, V shape (batch_size x kv_heads x seq_len x embed_dim)

# attention of llama, see https://zhuanlan.zhihu.com/p/643829565
# implement KV cache, decoder mask, fp16 and fp32 softmax
def llama_attention(past_kvs, attention_mask, **kwargs):
	Q, K, V = ... # pass
	# KV Cache
	seq_len_new = K.seq_len + past_kvs[0].seq_len
	K = torch.cat([past_kvs[0], K], dim=seq_len_new)
	V = torch.cat([past_kvs[1], V], dim=seq_len_new)
	past_kvs = (K_new, V_new)

	# decoder mask
	attn_weights = atten_weights + attention_mask
	dtype_min = torch.tensor(
		torch.finfo(attn_weights.dtype).min,
		device=attention_weights.device,
		dtype=attn_weights.dtype
	)
	attn_weights = torch.max(attn_weights, dtype_min)

	# upcast attention to fp32 in softmax -> down to fp16
	attn_weights = nn.functional.softmax(attn_weights, 
		dim=-1, dtype=torch.float32).to(Q.dtype)
	pass

def beam_search():
	pass

# xi: scalar
# softmax scales all xi so that they sum to 1
def softmax_i(xi, [x1, ..., xi-1, xi+1, ...]):
	return np.exp(xi) / np.sum(j=1->n, j!=i: np.exp(xj))

# X: shape (..., softmax_dim)
# output: same shape as X
def softmax(X, softmax_dim=-1):
	# why (X - max(X)): to prevent overflow to infinity
	X_max = np.max(X, axis=softmax_dim, keepdims=True)
	X_exp = np.exp(X - X_max)
	return X_exp / np.sum(X_exp, axis=softmax_dim, keepdims=True)

# x: scalar
# == logistic (activation) function
# output: range (0, 1)
# derivative: sigmoid(x) * (1 - sigmoid(x)) = sigmoid(x) * sigmoid(- x)
def sigmoid(x):
	return 1 / (1 + np.exp(- x))

# derivative: (log(sigmoid(x)))' 
# = 1 / sigmoid(x) * (sigmoid(x))' 
# = (1 - sigmoid(x)) = sigmoid(- x)
def log_sigmoid(x):
	return np.log(1 / (1 + np.exp(- x)))

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
	return - np.sum(y * np.log(p), axis=class_dim)

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
	pivot = nums[hi]
	l = lo
	for r in range(lo, hi):
		if nums[r] < pivot:
			swap(nums[l], nums[r])
			l += 1
	# now [lo, l-1] <= [l+1, hi-1] <= [hi] = pivot
	swap(nums[l], nums[hi])
	# now [lo, l-1] <= pivot <= [l+1, hi]
	return l

def quick_sort(nums, lo, hi):
	if lo >= hi:
		return
	swap(nums[random.randint(lo, hi)], nums[lo])
	pivot = partition(nums, lo, hi)
	quick_sort(nums, lo, pivot - 1)
	quick_sort(nums, pivot + 1, hi)

class MinHeap:
	def __init__(self, maxsize):
		self.maxsize = maxsize
		self.size = 0
		self.heap = [0] * (self.maxsize + 1)
		self.heap[0] = -1 * sys.maxsize
		self.front = 1

	def parent(self, pos):
		return pos // 2

	def leftChild(self, pos):
		return 2 * pos

	def rightChild(self, pos):
		return (2 * pos) + 1

	def isLeaf(self, pos):
		return pos * 2 > self.size

	def swap(self, pos1, pos2):
		self.heap[pos1], self.heap[pos2] = self.heap[pos2], self.heap[pos1]

	def minHeapify(self, pos):
		if not self.isLeaf(pos):
			cur = self.heap[pos]
			l = self.leftChild(pos)
			r = self.rightChild(pos)
			if cur >= l >= r:
				self.swap(cur, l)
				self.minHeapify(l)
			elif cur >= r >= l:
				self.swap(cur, r)
				self.minHeapify(r)

	def push(self, val):
		if self.size >= self.maxsize:
			return
		self.size += 1
		self.heap[self.size] = val
		pos = self.size
		while self.heap[pos] < self.heap[self.parent(pos)]:
			self.swap(self.heap[pos], self.heap[self.parent(pos)])
			pos = self.parent(pos)

	def pop(self):
		val = self.heap[self.front]
		self.heap[self.front] = self.heap[self.size]
		self.size -= 1
		self.minHeapify(self.front)
		return val

	def AllHeapify(self):
		for pos in range(self.size // 2, 0, -1):
			self.minHeapify(pos)
