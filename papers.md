## 0 General

- [2017 KG-Embedding Survey](2017_kge_survey.md)
- [2021 KG-Embedding Survey](2021_kge_survey.md)

## 1 AlexNet

- [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
- remove per layer decrease 2% result: turns out its just params choice
- no unsupervised pre-training: affect DL world, make people focus on supervised only until recently (2018+)

**Intro**

- make network deep without overfitting: CNN on GPU, with *unusual technique*
- image preprocess: reshape to 256x256, if longer cut it, if shorter fill with gray (this is bad)
- the deep network architecture maps image representation to a 4096-long tensor that can be understood by machine, similar to our encoder/embedding technique now
- SGD with momentum and weight decay
- Gaussian N(0, 0.1^2) initialization

**The “technique”**

- ReLU and tanh activation
- model parallel: slicing to fit GPU, in middle layer do attention-like transport
- local response normalization (not important now)
- overlapping pooling

**Reduce Overfit**

- data augmentation
	- spatial crop to 2048 (but similar) images
	- PCA on RGB channel
- dropout
	- 50% prob set hidden layer neuron output to 0
	- double the runtime to converge

## 2 ResNet

- [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)

**Intro**

- intuition
	- previous problem of Deep Network has vanishing/exploding gradients
	- need proper initialization
	- but still suffer from degradation: as layer goes deeper, model become less accurate (both training and testing error)
	- in theory, deeper network should be at least as good as shallower ones, as long as the added layers do *identity mapping*
- method
	- explicitly learn the identity mapping
	- do a *shortcut connection*
	- if the desired mapping is H(x), now learn F(x)=H(x)-x
	- so the output now is F(x)+x

**Residual Learning**

- deal with F(x) and x dimension dismatch
	- extra 0 padding, or 1x1 conv (now) to increasing dimensions
	- if go across feature map of two sizes, both use stride 2 
- residual block
	- res18: [conv[3x3,64-128-256-512] x2] x2
	- res34: [conv[3x3,64-128-256-512] x2] x3-4-6-3
	- deeper: *bottleneck design*, use 1x1 conv to project channels, after normal conv, do 1x1 conv again to transform back to desired size
	- res50: [conv[1x1-3x3-1x1,(64-128-256-512)-(same)-(samex4)]] x3-4-6-3
	- res101: [conv[1x1-3x3-1x1,(64-128-256-512)-(same)-(samex4)]] x3-4-23-3
	- res152: [conv[1x1-3x3-1x1,(64-128-256-512)-(same)-(samex4)]] x3-8-36-3
- result
	- as model go deeper, later layer becomes more stable and closer to 0 (identity)
- further thought
	- later people find the main reason ResNet works well is that it "solves" the gradient vanishing problem
	- because normally, the gradient is [H(g(x))]' = H(g(x))'g'(x) ~ approx 0
	- now [F(g(x))+g(x)]' = F(g(x))'g'(x)+g'(x) ~ approx g'(x)
	- where g'(x) is the shallower network and eaiser to train
	- so the network can at least be sth around g'(x), which is the shallower network's result
- aside: why modern network has billions of parameters w/o overfitting
	- with "residual like" architecture, the *intrinsic complexity* of the model is not that big
	- in fact, smaller model may also be possible to learn a similar power as bigger ones, but hard to find it
	- bigger model now has this *flexibility* to learn or not learn w/o losing anything
	- different in a way to idea of gradient boosting in machine learning: feature level residual instead of gradient level
	- may be seen as a type of *inductive bias*

## 3 Transformer

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
	- employ residual connection
	- employ layer normalization
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

## 4 BERT

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

## 5 GNN Intro

- [A Gentle Introduction to GNNs](https://distill.pub/2021/gnn-intro/)
- graph property: node, edge, global-context, connectivity
- what data can be represented as graph
	- graph (picture)
	- text
	- molecule: qm9
	- social network: karate club
	- citation network: cora
	- knowledge graph: wiki
- what tasks are generated by graph
	- graph-level task
	- node-level task
	- edge-level task
- challenge
	- cannot store as adj matrix - too big and sparse
	- adj matrix is not deterministic - heterogeneous graph
	- aggregation method is naive
- [Message Passing Neural Network (MPNN)](https://arxiv.org/pdf/1704.01212.pdf)
	- simple message passing
	- pooling information (aggregation) edge <-> node <-> global (master node)
	- issue: not fully use the connected information
	- aggregate neighborhoods info to FC
	- like attention
- sample
	- batch sampling
	- random walk sampling
	- random walk with neighborhood
	- diffusion sampling
- inductive biases
	- graph symmetries (permutation invariance)
- further
	- GCN as subgraph function approximations
	- edge and graph dual
	- graph convolution as matrix mult., matrix mult. as random walks
	- aggregation by GAT
	- graph interpretability
	- generative modeling

## 6 GAN

- [Generative Adversarial Nets](https://proceedings.neurips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf)
- Generative Net G vs. Discriminative Net D

**Intro**

- when G is a MLP, it can transform a random noise to any distribution
- when D is a MLP, we refer to it *adversarial net*
- when both are MLP, we can do backprogagation like usual

**Related Work**

- deep Boltzmann Machine
	- learn a probability distribution function
	- the target is to maximize the likelihood function
	- this is hard when dimension becomes higher
- generative stochastic network
	- just approximate the result by model fitting
	- instead of deriving and approximating the numerial distribution
	- pros: easy to calculate
	- cons: no distribution is found
- variational autoencoders (VAE)
	- develop backpropgation rules for one to learn the conditional variance of the generator
	- also use 2 models G and D
	- but the D here performs approximate inference
- noise-contrastive estimation (NCE)
	- also use a discriminative criterion to train a generative network
	- but the loss function is hard to optimize
- predictability minimization (debate)
	- reverse PM (by LSTM author, Huber)
- adversarial example
	- this is different, it is used to generative new samples

**GAN**

- min over G[max over D[V(D,G)]]
- suppose x is the distribution G wants to approximate and D wants to discrimate from
- D(.), G(.) take in a random input z and project it to x
- V(D,G) = Expectation over x(log D(x)) + Expectation over x(log(1-D(G(x))))
- train D to maximize the probability of assigning the correct label to both training examples and samples from G, the maximum is 1 (truth)
- train G to minimize the probability of making D correctly label the generative examples to be "generated" examples (instead of real examples)
- if D is not perfect, log term will make it negative
- if G is not perfect, D(G(x)) will be big
- if both D and G are balanced, the training is done
- procedure of each iteration
	- for k steps, sample m real samples xi and m random noises zi, update D by + gradient of log D(xi)+log (1-D(G(zi)))
	- sample m random noises zi, update G by - gradient of log(1-D(G(zi)))
- optimize
	- the D(G(z)) term is easy to be 0, ie. D is trained too well but G is weak
	- better: train G to maximize log(D(G(z)))
- analysis
	- if G fixed, the optimal D = % data(x) / (% data(x) + % G(x)), if G(x) = data(x), then this is 50%
	- this is called *two sample test*: if a classifier cannot split G from data, then G is good
	- we can show this by proving the maximization of V(G,D)
	- we can also show the global min is achieved if % data = % G: *KL divergence*
	- KL(p||q) = Expectation over x on p(log(p(x)/q(x))): when knowing p, how many bits need to present q
- cons
	- hard to go to closure (generate good result) when G and D is not good enough
	- many optimization needed
- influence
	- a new era on generative models
	- unsupervised loss function motivates further research

## 7 Vision Transformer (ViT)

- [An Image is Worth 16x16 Words: Transformers For Image Recognition at Scale](https://arxiv.org/pdf/2010.11929.pdf)
- In CV, transformer input sequence is toooo long

**Intro**

- use 16x16 window as one element, so that a 224x224 image -> 14x14 as input and put into transformer
- and it must be supervised (unlike NLP tasks)
- since replaced CNN, it does not have inductive basis (locality & translation invariance), but it outperformed by large scale pre-trainings (ImageNet 88%)

**Related Work**

- use feature map as input
- use local window as input
- Sparse Transformer: approximate self-attention
- block of variable size, extreme is to put 2D width and height to self-attention (so that it becomes 1D) as input

**ViT**

- linearly project each (flattened) patch of image block to an element
- add position embeddings
- add a special char (mimic NLP begin-of-word), as [class] token
- procedure
	- input 224x224x3 image
	- cut to 16x16x3 block, we get 196 of them
	- FC embedding each block, where FC has 768x768 (768=16x16x3)
	- add begin [class] token, we get input of 197x768
	- add positional embedding, still 197x768
	- layer norm
	- do multi-head attention, suppose 12 heads
	- then K,Q,V each has 197x64 (64=768/12)
	- concat each head's output, so the final K,Q,V each has 197x768
	- layer norm, FC to 197x3012, then FC to 197x768
	- [class] token is FC to classify
- can also use 2D or relative positional embedding: similar to 1D
- can also use global avg pooling of last layer to classify: similar to [class]
- hybrid (mix CNN and ViT)
	- use the feature map of res50 (14x14) as elements of input to ViT
- positional embedding is the only place ViT use inductive basis
- pre-trained ViT is hard to deal with larger image size
	- if keeping patch size same, element sequence length is longer
	- pre-trained positional info is lost (can be handled by interpolation but not that good)
- result show that last layer *pays attention to* the object of classification
- result show that *masked patch prediction* is good as well (encourage contrastive pre-training)

**Conclusion**

- *core* is image patch can be treated like words
- can move to detection (ViT-FRCNN) and segmentation (SETR), later Swin-Transformer makes it better
- next
	- do self-supervision (MAE)
	- do larger: Scaling ViT (ImageNet 90%)
	- encourage multimodal tasks
- although ViT needs larger dataset to perform well, later people find it can be also powerful on smaller dataset when strong regularization

## 8 Masked Autoencoders (MAE)

- [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/pdf/2111.06377.pdf)
- BERT technique applied in ViT

**Intro**

- 75% mask image, input 25% unmasked into ViT (encoder)
- output add back mask (with only positional embedding), use a decoder to recreate image
- move from NLP(BERT) to CV
	- mask indicator is different
	- info density is different: in image network tends to do interplation if masked, but if masking all random neighbors, network would learn beyond low-level statistics
	- decoder needs pixel-level accuracy in image, but word-level in NLP
- more masks make it effecient as well
- also go on transfer learning (object detection, instance/semantic segmentation)

**MAE**

- use partial obs. (asymmetric) to reconstruct original signal
- encoder (ViT): map obs. signal to latent representation
- random sampling with most masked
- decoder: another series of transformers
- when use in transfer learning, decoder is not needed

**Conclusion**

- backbone: ViT, "simple" and scale well
- image compare to words, one masked block do not form a *semantic segment/entities*, but can still learn semantics
- can generate inexistent content (social use, be cautious)

## 9 MoCo

- [Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/pdf/1911.05722.pdf)

**Contrastive Learning**

- is part of self-supervision
- CL learns a feature space where similar samples are close to each other
- there are many self-supervised *pretext tasks* (encoding method)
- one is *instance discrimination*
	- one sample is encoded to 2 positive samples, one of them is called archor
	- all other encoded samples are negative samples
	- metric is called *NCE loss*
	- the power is that we can define pos, neg sample any way we want
- inspire multimodal: CLIP from OpenAI

**Intro**

- in NLP, unsupervised learning is performed by mapping words to features
- in CV, it is hard to do the mapping
	- in NLP, words have strong semantic meaning themselves, they are discrete
	- in CV, pixel does not contains much high level info, they are continuous
- previous work builds *dynamic dictionary* (pos, neg samples)
	- the goal is to minimize the *contrastive loss*
	- treat the encoded pos, neg samples as *key*
	- the archor as *query*
	- want to learn a dictionary matching problem
	- where the encoded query should be similar to its matching key
	- and far away from other keys
- dictionary type
	- end-to-end: use same encoder for key and query (cannot big)
	- memory bank: no encoder for key, save all encoded querys by different set of keys (not consistent -> but large)
	- MoCo: ours
- goal of dictionary
	- large: avoid shortcut and weak learning
	- consistent: keys should be encoded the same way
	- MoCo designs a *momentum encoder* for keys to achieve this

**Method**

- CL builds discrete dictionary on high-dim continuous input (ie. image)
- momentum encoder
	- suppose the query encoder is Q_q
	- then for key_k, the encoder Q_k = m * Q_(k-1) + (1-m) * Q_q
	- experiment show big m (~= 1) is better: a slowly changing Q_k (->consistent)
	- so it is a moving-average encoder of the key (-> consistent)
	- also every key is put in a FIFO *queue* to save space (-> large)
- CL as dictionary look-up
	- only one pair of key k_i and query q matches
	- want to measure their similarity
	- since there are too many keys (neg), use noise controstive estimation
	- NCE is an estimate of the real full data from 2-class classification: pos, neg
	- NCE use softmax of the inner product of k_i & q
	- InfoNCE loss: reduce the # of neg to K
- Suffling BN
	- use pure BN would leak info of key, make model do shortcut
	- so shuffle sample order
- result
	- can outperform in 7 downstream tasks
	- since it is unsupervised, MoCo can replace imageNet pretrained network
	- because there are much more unlabeled images than labeled ones

**Conclusion**

- move from (labeled, 1M) imagenet to (unlabeled, 1B) instagram only gives < 1% increase in result
	- maybe large scale dataset is not fully exploited
	- hope an advanced pretext task will improve it
	- masked auto-encoding ?
- the main part to improve
	- pretext task (?)
	- loss function (MoCo)
- pretext task
	- reconstruct image, patches, colors ...
	- data transformation, reordering, segmenting, tracking in video ...
- loss
	- "fixed" loss (vs. ground truth)
	- contrastive loss: measure similarity
	- adversarial loss: measure probabilistic distribution
- in MoCo
	- pretext task: instance discrimination
	- loss: NCE
	- these two can be matched pretty well

## 10 Swin Transformer

- [Swin Transformer: Hierarchical Vison Transformer using Shifted Windows](https://arxiv.org/pdf/2103.14030.pdf)
- ViT to all vision tasks
- challenge: scale and resolution of different tasks
- method: (hierarchical) shifted windows, reduce size & share global info

**Intro**

- ViT 16x16 patch for segementation task for example, is both low resolution and comput. expensive (caz segmentation use over 1000x1000 pixels, but classification uses 224x224)
- swin transformer uses self-attention on *each window* of the image
	- first use 4x4 window, each has 4x4 patch
	- shift each window and do self-attention on them to share global info
- future: do shifted window on NLP

**Method**

- stage 1
	- input image 224x224x3
	- partition to 4x4 windows -> 56x56x48, where 48=4x4x3
	- linear embedding -> 56x56x96, where 48 embedded to 96
	- reshape to 1D (as NLP input for transformer)-> 3136x96, where 3136 is the length of input feature, 96 is the dim of each feature
	- [notice the length 3136 is very long, in ViT it is 196, so...]
	- swin transformer block (details later) -> 56x56x96, same dim
- stage 2
	- *patch merging* (mimic pooling) -> 28x28x192
		- suppose input has dim HxWxC
		- cut each patch to 4 patches with labeled positions
		- merge each labeled position of all patches to one patch
		- now we have 4 new patches (with global receptive field)
		- each new patch has dim H/2xW/2xC
		- now concatenate them by dim C -> H/2xW/2x4C
		- pass through 1x1 conv to get H/2xW/2x2C (do this to match the usual process in ResNet that doubles the C dim)
	- swin transformer block -> same dim
- stage 3
	- patch merging -> 14x14x384
	- swin transformer block -> same dim
- stage 4, same -> 7x7x768

**Swin Transformer Block**

- block 1
	- layer norm
	- self-attention on *non-overlapping patches*
		- suppose stage 1, input window 56x56x96
		- every window has MxM patches, here use M=7
		- so input length of each patch is always 7x7=49
		- there are 56/7x56/7=8x8=64 patches
		- do self-attention on each patch
		- complexity is lowered depending on size of M
	- skip-connection
	- layer norm
	- MLP
	- skip-connection
- block 2
	- layer norm
	- shifted window partitioning: connection across windows
		- for each MxM patches, shift them by M/2xM/2
		- so we have 1 big patch in the middle, 4 long patches and 4 small patches, in total 9 patches
		- sizes are MxM, 4 of MxM/2 (M/2xM), 4 of M/2xM/2 -> 4xMxM
		- problem: since sizes are different, we cannot do self-attention in batches
		- simple solution(?): pad to MxM for all, too expensive
		- better: cyclic shift -> *masked MSA* -> reversed cyclic shift
	- skip-connection
	- layer norm
	- MLP
	- skip-connection
- variants: Swin-T,S,B,L (different params)
