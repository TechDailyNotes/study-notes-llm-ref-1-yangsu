# NLP

### 词嵌入(Word Embedding)

- 引入
	- one-hot的缺点：每个词是孤立，onto itself的，两个词的内积永远是0
	- 改进：featurized representation
		- 把每个词设置K个feature并用他们表示word
		- 如：word[man] -> feature[gender, age, action, food, ...]
		- embedding matrix E with (K, vocab_count)
		- E * O_k = e_k, embedding vector, 表示第k个词的embedding
		- embedding layer: 直接用E的第k个列作lookup，如keras
- 概念
	- 降维在NLP的应用
	- 1-of-N Encoding -> dimension for "other", word hashing -> Word Class -> Word Embedding
	- 把word映射到高维空间，想要实现类似语义的词距离相近
	- 生成词向量是unsupervised, 不能用auto-encoder
	- 词在context中可以被machine理解
- 方法
	- count based, 计算两词co-occur的频率(代表：Glove Vector)
	- prediction based, 训练一个网络预测下一个word是什么
		- shared parameters, n个1-of-N encoding的词作input预测下一个词，不同词*encoding*的权重必须一样
		- continuous bag of word model (CBOW) model, 用前后词预测中间词
		- skip-gram, 用中间词预测前后词
- 特性
	- visualization：t-SNE(2008) 降维可视化word embeddings
	- analogy reasoning：linguistic regularities(2013)
		- 如：V(hotter)-V(hot)=V(bigger)-V(big)
		- 类别的性质如“包含”可以被相似的表示出来，已知3个可以用来解另一个
		- argmax(w: sim(V(?), V(hotter)-V(hot)+V(big)))
		- sim: cosine similarity, sim(u, v) = (u^Tv)/(||u||\_2||v||\_2), 或在表达相似性时不太常用的Euclidean dist
- 迁移学习transfer learning : A -> B
	- 从大数据库(1B-100B)中学习词嵌入, A
	- 迁移嵌入到100k左右大小的新训练集, B
	- repeat

**类别**

- multi-lingual embedding
	- 不同语言间没有关联，如果input只有一种语言
	- 如果提前把语言词汇做projection，可以做到对新语言（在某些语言里没有训练集，某些语言里有时）翻译的效果
- multi-domain embedding
	- 把图像和语义提前做projection，可以做到对新图像（在原图像集里没有，但有对应的语义时）分类的效果
- document embedding
	- semantic embedding：变成bag-of-word，但忽略了word之间的位置信息
	- (unsupervised) paragraph vector, seq2seq auto-encoder, skip thought
	- more (todo)

**发展历程**

- probabilistic neural network, 2003 
	- 用前k(k=4)个词的embedding matrix E连接FC网络预测下一个词
	- 更好的： 4 words on left & right/last 1 word
- Word2Vec/skip-gram(word representation in vector space), 2013
	- 概念
		- 假设vocab size = 10k
		- sample一个词c，并在上下文d个词距内sample一个词t
		- 学习词c(context)对词t(target)的映射
		- 希望看到input x=c时预测y=t的概率
	- 实现
		- EO_c = e_c -> softmax -> y
		- softmax: p(t|c) = (exp(theta_t^T * e_c) / sum(j=1->10k : exp(theta_j^T * e_c)))
		- theta_t : 关于output t的参数
	- 问题：经过整个vocab，计算量太大
	- 解决：分级softmax分类器hierarchical softmax classifier，一个分支定界的二叉树，不完美平衡
	- 实际上会用别的方法平衡过于常用或rare的词来计算p(c)
- 负采样negative sampling in skip-gram, 2013
	- 概念
		- 设定d之后，先进行一次和之前一样的sample，标记词c为context，词t为word，target标记为1代表正采样
		- 再进行k次sample，使用相同的词c，但word从整个vocab里随机选择，target全部标记为0代表负采样（如果word其实正好是正采样也不care）
	- 任务
		- 学习x=(c, word)对y=target的映射
		- 同样的，希望看到input x=c时预测y=t的概率
		- 数据集小时，k可以选择5-20；大时2-5
	- 实现
		- 使用p(y=1|c,t) = lambda(theta_t^T * e_c)代替softmax
		- 相对于原来softmax取最大概率，这个过程在学习[vocab size]个二分类问题
		- 而且，每次训练迭代只使用k+1个样本，其中k个是负样本，大大减少了计算量
	- 发展
		- softmax相当于uniform均匀采样，这个相当于empirical经验/观测采样
		- 论文通过实验发现更好的折中公式(todo)
			- p(w_i)=f(w_i)^(3/4)/sum(j:f(w_j)^(3/4))
- GloVe(global vectors for word representation), 2014
	- 概念
		- todo
	- 实现
		- x_{ij}: i(target)出现在j(context)(d词距)中的次数
		- 目标：minimize sum(i,j: f(X_{ij}) * sim^2)
		- sim: theta_i^T * e_j + b_i + b'\_j - log(X_{ij}), 表示目标的词嵌入与context词的内积与他们的权重的相似性
		- f(X_{ij}):加权项，f(X_{ij})=0 if X_{ij}=0, 是一种启发式计算word frequency的方法
	- visualization problem
		- 这种相似性的学习是把featurization的很多维度结合在一起的，很难humanly可解释

### Transformer

- 是一种seq2seq的model：output的长度由机器自己决定，例子：
	- speech recognition, machine translation（长度N->N'）
	- speech translation, text-to-speech(TTS) Synthesis
	- chatbot, (NLP) QA
	- Suntactic Parsing 文法剖析
	- multi-label classification (not multi-class)
	- object detection
- seq2seq (2014)
	- Input -> Encoder -> Decoder -> Output

**Encoder**

- can use CNN, RNN or self-attention, 输入一排向量输出一排向量
- (原始, 2002) Input -> positional embedding -> multi-head attention -> + Input (residual connection) -> layer norm -> FC -> + Input -> layer norm -> Output
- (PowerNorm, 2003) Input -> positional embedding -> layer norm -> multi-head attention -> + Input -> layer norm -> FC -> + Input -> Output

**Decoder**

- Autoregressive (AT) decoder(例：Tacotron)
	- begin of sentence (BOS, special token)
	- 用自己的每一个输出当作下一个输入 (error propgation)，最后一个叫end of sentence (EOS, end token)
	- 与encoder不同的
		- masked self-attention, 第i个位置的key只关注<=i位置的query和v来输出
		- 只考虑上文，因为decoder的方法也是从前往后产生block
- Non-autoregressive (NAT) decoder(例：FastSpeech)
	- 与AT decoder不同，一次输入所有input长度的BOS，然后直接完成句子每个token的输出
	- 如何确定生成的长度
		- learn另一个classifier预测
		- 设置生成长度的阈值，截断后面的output(如果有的话)
	- 优势：parallel并行，controllable的生成长度
	- 劣势：performance经常比AT的差(Multi-modality)

**Cross Attention**

- 把encoder输出的结果和decoder的mask self-attention的结果再次做attention，叫做cross attention
- 原始算法
	- encoder输出的结果被当作一般attention的input a
	- a进行矩阵变换得到key和v
	- 然后decoder的mask self-attention的结果进行矩阵变换得到query
	- 进行基于每个query的attention操作
	- 最终的结果b被当作decoder后续FC的input
- 发展算法(2020)
	- todo

**延申**

- Teaching Forcing：训练时decoder的input会是正确答案(训练集)
- Copy Mechanism：复述一部分input的词汇作output
- Summarization：把题目当作摘要的训练答案
- Guided Attention (todo)：人为引导attention的样貌，方式，顺序etc
	- monotonic attention
	- location-aware attention
- Beam Search 束搜索
	- Greedy Decoding 贪心的选择方式，一直选择概率最大的结果做decoding
	- 但可能有全局的更优解，虽然单个output不是最佳的
	- 因为无法检查所有可能的path，就有了beam search的方法 (todo)
	- 对于比较确定答案的任务，beam search比较有帮助
	- 对于比较发散性答案的任务，beam search需要加入一些randomness随机性，不然有陷入重复循环的问题(如句子补全，TTS)
- 评估标准 BLEU score
	- train时看cross-entropy
	- validate，test时把结果和标准答案作比较(就叫BLEU score)
	- exposure bias：指训练和测试的标准不同的问题
		- 解决方向：train时加入错误讯息(Scheduled Sampling)
		- 但有会伤害transformer平行化能力的问题(todo)

### HMM

### BERT

- Bidirectional Encoder Representations from Transformers (BERT)
- "the new era (of BIG models)"

**Self-Supervised Learning**

### CRF

### LDA
