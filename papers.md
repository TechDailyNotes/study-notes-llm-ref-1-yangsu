### 1 Transformer

- [Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf)

**Intro**

- sequence transduction model (seq2seq): RNN, LSTM, GRU - encoder-decoder
- previous works are sequential, h_t = f(h_t-1, input at t)
- also cause memory loss when sequence is long
- our work is parallelized

**Background**

- multi-head attention replaces CNN
- first transduction model on self-attention w/o RNN, CNN

**Attention**

- encoder: input symbol representations to continuous representations
	- N = 6 idential layers
	- each layer has two sub-layers: multi-head self-attention & FC network
	- empoly residual connection
	- empoly layer normalization
	- output of each layer is d = 512
- decoder: encoder output is the input, output is autoregressive and becomes the next input (because output is generated one by one)
	- N = 6 too
	- has a 3rd layer: multi-head attention over the output of encoder
	- 1st layer change to masked multi-head attention to prevent from seeing later inputs while doing prediction
- scaled dot-product attention
	- Q query, K key, V value
	- first do dot product of (Q,K): cosine value, measure similarity
	- scale by 1/sqrt(|K|), to deviate the output to two ends
	- mask. value after t by replacing with a big negative, so that softmax will make it almost 0
	- do softmax to deviate to [0,1], then * V
	- multi-head. concatenate the output & FC
		- h = 8 parallel attention layers
	- previous work
		- additive attention
		- dot-product attention
- parameters
	- encoder input embedding: n -> n x d
	- Q, K, V: n x d_Q, n x d_K, n x d_V each (d_Q=d_K), by MLP on input (multi-head. has h of these; mask. need input embedding after t = 0)
	- Attention(Q,K,V) = softmax(QK^T / sqrt(d_K)) V, dim = softmax(n x n) * n x d_V = n x d_V
	- FC to encoder output: n x d_V * d_V x d = n x d [1]
	- similarly decoder after masked attention gives: n x d [2]
	- input2: [1] * FC -> (K2, V2); [2] * FC -> Q2
	- since [1] is learned from input embedding and [2] is learned from ouput embedding, make [2] to be Q can be beneficial wrt. [1] to be K
	- output2: n x d too
	- [here sequence info is extracted]
	- point-wise FC twice wrt. *each position* (1 x d), with the hidden state is 4 times bigger (1 x 4d)
	- [here semantic info is extracted]
	- softmax to final output
- aside: compare to RNN
	- historical sequence info is passed to future, and sequentially move on
	- transformer first extracts all sequence info, then extract individual semantic info
- aside: other optimization
	- each embedding weights are multiplied by sqrt(d)
	- then add positional encoding (by sin/cos function)
- why attention: compare complexity, sequential operation, and maximum info passing length
	- self-attention: n^2 x d, O(1), O(1)
	- RNN: n x d^2, O(n), O(n)
	- CNN: k x n x d^2, O(1), O(log_k(n)) where k is kernel size
	- self-attention with neighbor of r: r x n x d, O(1), O(n/r)
	- overall self-attention is parallel, info passing is fast, with smiliar complexity

**Training**

- use WMT embedding since it use 词根 from both English and German/French
- 8 P100 GPU, 3.5 days
- lr: adam, ~ 1/sqrt(d), ~ 1/sqrt(training step)
- reg: add residual dropout before the output is added by residuals
- ls (label smoothing): softmax gives 1 when 10% sure
- final (multi-head)
	- N = 6, d = 512, h = 8, d_fc = 4 x d = 2048, d_K = d_Q = d_V = 512
	- drop% = 0.1, ls = 0.1, train step = 100k

### 2 BERT

- [BERT: Bidirectional Encoder Representations from Transformers](https://arxiv.org/pdf/1810.04805.pdf)
- GPT预测 -> BERT完形填空
- GPT decoder -> BERT encoder

**Intro**

- pre-training is effective in NLP
- strategy in pre-training: *feature-based* (ELMo) & *fine-tuning* (GPT)
	- property: both do prediction
	- limit: both one directional
- BERT do bidirectional
	- used a "masked language model" (MLM)
	- randomly mask word and predict it based on left and right
	- also predict if 2 given sentences are neighbors
- contribution
	- bidirectional
	- pre-trained reoresentation reduce the need to train from scratch
	- can tackle a broad set of NLP tasks with pre-trained architecture

**Related Work**

- unsurpervised feature-based approaches (ELMo etc.)
- unsurpervised fine-tuning approaches (GPT etc.)
- transfer learning from supervised data (many used in CV, but little in NLP)

**BERT**

- *pre-training* with unlabeled data
- *fine-tuning* with labeled data initialized by pre-trained parameters
- architecture
	- multi-layer bidirectional transformer encoder
- parameters
	- 嵌入层: 30k * H
	- every transformer: H^2 * 12, which has L of them
		- K,V,Q each have H * H
		- project to H * H
		- go to MLP with input H * 4, output H * 4
		- go to MLP with input H * 4, output H
	- total = 30kH + 12LH^2
	- H = 768, L = 12, # total = 110 M
	- H = 1024, L = 24, # total = 340 M
- input "sequence" can be either a single sentence / a pair of sentences
	- WordPiece embeddings (cut words into 词根)
	- input representation = sum(token (词根), segment (which sentence), position)

**pre-training**

- Masked LM with 15% of all WordPiece tokens in each "sequence" at random
	- 80% replace the word with [MASK]
	- 10% replace the word with random word
	- 10% keep word unchanged (to bias the representation towards the observed word)
- Next Sentence Prediction (NSP)
	- choose sentence pair A and B
	- 50% B is the actual next sentence
	- 50% B is a random sentence
	- useful in QA, NLI (natural language inference 自然语言推理)
- data
	- use document-level instead of sentence-level
	- good to extract long sequences

**fine-tuning**

- different from decoder-encoder: can see both sides, but cannot do MT (machine translation) now (or other generation tasks like summary)
- to work on other end-to-end tasks: change input pair represebtation, output add softmax etc.
- experiement
	- GLUE (general language understanding evaluation)
	- SQuAD 1.1, 2.0 (stanford QA dataset)
	- SWAG (situations with adversarial generations)

## 3 GNN

- [A Gentle Introduction to GNNs](https://distill.pub/2021/gnn-intro/)
