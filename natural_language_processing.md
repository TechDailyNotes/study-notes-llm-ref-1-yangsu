# NLP

### Source

- [知乎](https://zhuanlan.zhihu.com/p/643560888)
- [PaddlePaddle](https://paddlepedia.readthedocs.io/en/latest/index.html)
- [NLP Course YouTube](https://www.youtube.com/playlist?list=PLoROMvodv4rOSH4v6133s9LFPRHjEmbmJ)
- [NLP GitHub Repo](https://github.com/km1994/nlp_paper_study)

### Word Embedding

- WordNet: word dictionary by similarity
	- missing contextualized meaning
	- missing new meanings of words
	- subjective
	- require human labor to create and adapt
	- cannot compute accurate word similarity
- localized representation: represent word as discrete symbols
	- eg. one-hot
	- word vectors are orthogonal, no similarity
	- expensive vector dimension
- distributional representation: represent word by their context
	- a word's meaning is given by the words that frequently appear close-by
	- eg. word vectors/embeddings

**Word2Vec**

- we have a large piece of text
- for each position in the text, their is a center word and other words (called context words) differing by distance d
- compute the % that the context word occurs by distance d given the center word appears
- maximize this % across a fixed window size m (d=-m->m)
- algo
	- for each word, we use two word vectors: as center word and as context word
	- then the % of the context word *o* occurs given a center word *c* is their dot product similarity *u_o u_c*, normalized over entire vocab, then taking softmax
	- loss function is the neg log likelihood of this similarity across all word vectors
	- we calculate gradient and use SGD to optimize the loss function
- property
	- vector composition (arithmetic): word_a - word_a2 = word_b - word_b2
	- can convey several different meanings for one word
	- Word2Vec is part of the *bag of words* model, it makes the same prediction at each position (which is simple and can be optimized)
- category
	- skip-grams (SG): predict context (outside) words given center word
	- continous bag of words (CBOW): predict center word given (bag of) context words
- negative sampling: improve training effeciency
	- sample k negative center-context word pairs and 1 pos pair
	- maximize the % of pos pair and minimize the % of neg pairs
	- change softmax objective to sigmoid/logisitic
	- instead of sampling words based on co-occurrence, sample by a unigram distribution P = U**(3/4) / Z, which is proved to be better
- co-occurrence count matrix
	- window-based: suppose window of len 1 -> next-to words count matrix, symmetric
	- count co-occurrence vectors problem: vectors increase in size with vocab, very high dim and sparse -> less robust
	- improve: dim reduction, SVD works bad because normal-distributed error assumption not satisfied
	- problem: function words (the, has, he) too frequent -> syntax has too much impact
	- findings in 2014: ratios of co-occurrence % can encode meaning components
	- we want a log-bilinear model wi * wj = log P(i|j), or their vector difference wx * (wa - wb) = log (P(x|a) / P(x|b))
	- leads to GloVe

**GloVe** 

- combine Word2Vec and co-occurrence count matrix
	- we want the difference between the cosine distance (similarity) of word vectors wi * wj and the log-co-occurence log P(i|j) is small
	- then this difference * (a heuristic way of calculating) frequency of co-occurrence becomes our loss for vectors i and j
- evaluate
	- intrinsic 1: word vector analogies -> evaluate by how well word vectors cosine distance after addition captures intuitive [semantic and syntactic analogy questions] (as scores)
	- intrinsic 2: word vector distance and correlation with human judgments
	- extrinsic: all subsequent NLP tasks (later)
- aside: word senses (ambiguity)
	- most words have many meanings
	- idea: cluster word windows around words, retain  with each word assigned to (pre-defined) k different clusters
	- problem: cannot precisely define how k should be chosen
	- findings in 2018: different senses of a word vector actually resides in a linear superposition (weighted sum) in standard word embeddings like word2vec -> use one word vector is sufficient enough
	- surprising result: we can actually sparse different word senses from one word vector by idea from *sparse coding* because their vectors are so sparse in the high-dim space

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

**发展历程**

- probabilistic neural network, 2003 
	- 用前 k(k=4) 个词的embedding matrix 连接 FC 网络预测下一个词
	- 更好：4 words on left & right/last 1 word
- Word2Vec (word representation in vector space), 2013
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
	- 解决：hierarchical softmax classifier，一个分支定界的二叉树，不完美平衡
	- 实际上会用别的方法平衡过于常用或rare的词来计算p(c)
- 负采样negative sampling, 2013
	- 概念
		- 概率采样，可以根据词频进行随机抽样，倾向于选择词频较大的负样本
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
		- window 比较大，会提取更多的topic信息
		- window 比较小，会更加关注于词本身
	- 发展
		- softmax相当于uniform均匀采样，这个相当于empirical经验/观测采样
		- 论文通过实验发现更好的折中公式(todo)
			- p(w_i)=f(w_i)^(3/4)/sum(j:f(w_j)^(3/4))
- 分层softmax
	- 思想：将一个全局多分类的问题，转化成为了若干个二元分类问题，从而将计算复杂度从O(V)降到O(logV)；每个二元分类问题，由一个基本的逻辑回归单元来实现
	- 用哈夫曼树把预测one-hot编码改成预测一组01编码，进行层次分类
	- 越常用的词拥有更短的编码
	- 高频词靠近树根，需要更少的时间会被找到，符合贪心优化思想
- 下采样
	- 文本数据通常有“the”“a”和“in”等高频词，它们在非常大的语料库中甚至可能出现数十亿次
	- 然而，这些词经常在上下文窗口中与许多不同的词共同出现，提供的有用信息很少
	- 此外，大量高频单词的训练速度很慢
	- 因此，当训练词嵌入模型时，数据集中的每个词将有概率地被丢弃
	- 该词的相对比率越高，被丢弃的概率就越大
- GloVe(global vectors for word representation), 2014
	- 概念
		- 交叉熵损失可能不是衡量两种概率分布差异的好选择，特别是对于大型语料库，GloVe使用平方损失来拟合预先计算的全局语料库统计数据
	- 实现
		- x_{ij}: i(target)出现在j(context)(d词距)中的次数
		- 目标：minimize sum(i,j: f(X_{ij}) * sim^2)
		- sim: theta_i^T * e_j + b_i + b'\_j - log(X_{ij}), 表示目标的词嵌入与context词的内积与他们的权重的相似性
		- f(X_{ij}):加权项，f(X_{ij})=0 if X_{ij}=0, 是一种启发式计算word frequency的方法
	- visualization problem
		- 这种相似性的学习是把featurization的很多维度结合在一起的，很难humanly可解释

### TF-IDF

- 思想
	- 对于句子中的某一个词（字）随着其在整个句子中的出现次数的增加，其重要性也随着增加；（正比关系）【体现词在句子中频繁性】
	- 对于句子中的某一个词（字）随着其在整个文档中的出现频率的增加，其重要性也随着减少；（反比关系）【体现词在文档中的唯一性】
	- 如果某个单词在一篇文章中出现的频率TF高，并且在其他文章中很少出现，则认为此词或者短语具有很好的类别区分能力，适合用来分类
- 定义
	- 表示某一特定句子内的高词语频率，以及该词语在整个文档集合中的低文档频率，可以产生出高权重的TF-IDF
	- 因此，TF-IDF倾向于过滤掉常见的词语，保留重要的词语
- 优点
	- 容易理解和实现
- 缺点
	- 其简单结构并没有考虑词语的语义信息，无法处理一词多义与一义多词的情况
- 对比Word2Vec
	- word2vec 是稠密的向量， tf-idf 则是稀疏的向量
	- word2vec 的向量维度一般远比 tf-idf 的向量维度小得多，在计算时更快
	- word2vec 的向量可以表达语义信息, tf-idf 的向量不可以
	- word2vec 可以通过计算余弦相似度来得出两个向量的相似度, tf-idf 不可以

### Syntactic Structure and Dependency Parsing

- NER, named entity recognition (linking)
	- goal: find and classify names in text
	- application: track mentions of particular entities in docs, QA
	- idea: classify each word in its context window

### FastText

- word-level Model 存在的问题
	- OOV：容易出现单词不存在于词汇库中的情况 -> 需要设置最佳语料规模，使系统能够获得更多的词汇量
	- 误拼障碍：如果遇到了不正式的拼写, 系统很难进行处理 -> 需要矫正或加规则约束
	- 做翻译问题时, 音译姓名比较难做到
- char-level Model
	- 优点：能够解决 Word-level 所存在的 OOV 问题，拼写类似的单词 具有类似的 embedding
	- 问题：Character-level 的输入句子变长；数据变得稀疏；对于远距离的依赖难以学到；训练速度降低
	- 解决方法
		- 利用多层 conv 和 pooling 和 highway layer，输入的字符首先需要经过 Character embedding 层，并被转化为 character embeddings 表示
		- 采用 不同窗口大小的卷积核对输入字符的 character embeddings 表示进行卷积操作，原论文中采用的窗口的大小分别为 3、4、5 ，也就是说学习 Character-level 的 3-gram、4-gram、5-gram
		- 对不同卷积层的卷积结果进行 max-pooling 操作，即捕获其最显著特征生成 segment embedding
		- segment embedding 经过 Highway Network (有些类似于Residual network，方便深层网络中信息的流通，不过加入了一些控制信息流量的gate）输出结果，再经过单层 BiGRU，得到最终的 encoder output
		- 之后，decoder 再利用 Attention 机制以及 character level GRU 进行 decode
	- 通过这种方式不仅能够解决 Word-level 所存在的 OOV 问题，而且能够捕获 句子的 3-gram、4-gram、5-gram 信息，这个也是 FastText 的想法雏形
- fasttext: subword Model
	- 利用 subword 将 word2vec 扩充，有效的构建 embedding
	- 将每个 word 表示成 bag of character n-gram 以及单词本身的集合，例如对于 where 这个单词和 n=3 的情况，它可以表示为 \<wh,whe,her,ere,re\> ，其中"<",">"为代表单词开始与结束的特殊标记
	- 每个单词可以表示成其所有 n-gram 的矢量和的形式，之后就可以按照经典的 word2vec 算法训练得到这些特征向量
	- 这种方式既保持了 word2vec 计算速度快的优点，又解决了遇到 training data 中没见过的 oov word 的表示问题
- 结构
	- 每个单词通过嵌入层可以得到词向量
	- 然后将所有词向量平均可以得到句子的向量表达
	- 再输入分类器，使用softmax计算各个类别的概率
- 问题
	- 每个n-gram都会对应训练一个向量，由于需要估计的参数多，模型可能会比较膨胀
- 压缩模型的建议
	- 采用hash-trick：由于n-gram原始的空间太大，可以用某种hash函数将其映射到固定大小的buckets中去，从而实现内存可控；
	- 采用quantize命令：对生成的模型进行参数量化和压缩；减小最终向量的维度
	- 采用分层softmax：见上

**发展历程**

- Byte Pair Encoding (BPE) 字节对编码, 2015
	- 在fastText中，所有提取的子词都必须是指定的长度，例如3到6，因此词表大小不能预定义
	- 为了在固定大小的词表中允许可变长度的子词，可以应用 BPF 压缩算法来提取子词
	- 字节对编码执行训练数据集的统计分析，以发现单词内的公共符号，诸如任意长度的连续字符。从长度为1的符号开始，字节对编码迭代地合并最频繁的连续符号对以产生新的更长的符号
	- 注意，为提高效率，不考虑跨越单词边界的对
	- 最后，可以使用像子词这样的符号来切分单词
	- 字节对编码及其变体用于诸如 GPT-2 和 RoBERTa 等LLM中的输入表示
- 小结
	- fastText模型提出了一种子词嵌入方法：基于word2vec中的跳元模型，它将中心词表示为其子词向量之和
	- 子词嵌入可以提高稀有词和词典外词的表示质量
	- 预训练的词向量可以应用于词的相似性和类比任务
	- 下面来到新时代

### Transformer - 2017, the new NLP era

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
	- 因为无法检查所有可能的path，就有了beam search的方法
	- 对于比较确定答案的任务，beam search比较有帮助
	- 对于比较发散性答案的任务，beam search需要加入一些randomness随机性，不然有陷入重复循环的问题(如句子补全，TTS)
	- 过于repetitive
- Greedy Search (todo)
- Top-k Sampling & Top-p Sampling (todo)
- 评估标准 BLEU score
	- train时看cross-entropy
	- validate，test时把结果和标准答案作比较(就叫BLEU score)
	- exposure bias：指训练和测试的标准不同的问题
		- 解决方向：train时加入错误讯息(Scheduled Sampling)
		- 但有会伤害transformer平行化能力的问题(todo)

**问题**

- transformer 不能很好的处理超长输入问题，因为它固定了句子长度，如bert里是512
	- 短于512：padding neg inf to right
	- 长于512方法一：截断
	- 长于512方法二：将文本划分为多个segments；训练的时候，对每个segment单独处理
	- 但是，因为segments之间独立训练，所以不同的token之间，最长的依赖关系，就取决于segment的长度
	- 出于效率的考虑，在划分segments的时候，不考虑句子的自然边界，而是根据固定的长度来划分序列，导致分割出来的segments在语义上是不完整的
	- 在预测的时候，会对固定长度的 segment 做计算，一般取最后一个位置的隐向量作为输出，为了充分利用上下文关系，在每做完一次预测之后，就对整个序列向右移动一个位置，再做一次计算，这导致计算效率非常低 
	- 长于512方法三：Segment-Level Recurrence (Transformer-XL 处理方式),在对当前segment进行处理的时候，缓存并利用上一个segment中所有layer的隐向量序列；上一个segment的所有隐向量序列只参与前向计算，不再进行反向传播
- 方向信息以及相对位置的缺失问题 (todo)
- 缺少Recurrent Inductive Bias (todo)
- 非图灵完备(通俗的理解，就是无法解决所有的问题)
- 缺少conditional computation
	- transformer在encoder的过程中，所有输入元素都有相同的计算量，比如对于“I arrived at the bank after crossing the river", 和"river"相比，需要更多的背景知识来推断单词"bank"的含义，然而transformer在编码这个句子的时候，无条件对于每个单词应用相同的计算量，这样的过程显然是低效的
- 时间和空间复杂度过大的问题
	- 自注意力与长度n呈现出$O(n^2)$的时间和空间复杂度
	- 解决：[Linformer](https://arxiv.org/abs/2006.04768)

**面试**

- attention公式: softmax(QK/sqrt(dk)) * V
- 为什么scale，layernorm: 让softmax输入的数据分布变好，数值进入梯度敏感区间，能防止梯度消失，让模型好训练
- 为什么除根号dk：一个是为了缓解梯度消失；另一个是xavier初始化embedding的方差是1/dk，position encoding的方差是1（三角函数-1到1），为了保证加起来规模一致需要除根号dk。bert是position embedding（学习式），所以不用除根号dk
- 是否可以不用除根号dk: 有，只要能缓解梯度消失的问题就可以，如T5模型的初始化
- 什么时候softmax之后梯度消失：输入基本是one-hot的时候，y=softmax(x), y'(x) = diag(y) - y * y.T = 0 matrix
- self-attention一定要这样表达吗：不一定，只要可以建模相关性就可以。当然，最好是能够高速计算（矩阵乘法），并且表达能力强（query可以主动去关注到其他的key并在value上进行强化，并且忽略不相关的其他部分），模型容量够（引入了project_q/k/v，att_out，多头）
- 为什么不用batchnorm
	- 对不同样本同一特征的信息进行归一化没有意义，因为：样本之间仍然具有可比较性，但特征与特征之间的不再具有可比较性
	- 在CV中，以跨样本的方式开展归一化，也就是对不同样本的同一channel间的所有像素值进行归一化，因此不会破坏不同样本同一特征之间的关系，而3个channel之间的可比较性也可以舍弃
	- 在NLP中，如果用batchnorm，则一句话中的词不再具有比较性，何谈同一句子之间的注意力机制呢，反之，layernorm舍弃了不同句子之间的可比较性，这个则是可以的
- 为什么用三个QKV，多头: 增强网络的容量和表达能力，多头可以类比CV中的不同channel
- self attention 计算复杂度：O（hidden_dim * seq_len ** 2）
	- 有什么技术降低复杂度提升输入长度：sparse attention放弃对全文的关注，只关心局部的语义组合，在整个计算矩阵上挖了空格子。这样做的好处是，下游任务的语义关联性的体现往往是局部的

### Transformer ++

**Longformer**

**Transformer-XL**

**Linformer**

**Performer**

**Efficient-Transformer**

**面试**

- Pre Norm vs. Post Norm
	- Norm = Layer Norm in NLP, or other Norm in other field
	- Pre Norm: y = x + Network(Norm(x))
	- Post Norm: y = Norm(x + Network(x))
	- 现在大模型主要还是用的Pre Norm, Bert 是 Post Norm
	- Post Norm 会有一定梯度消失的问题，但...
		- 好处？稳定了前向传播的数值，并且保持了每个模块的一致性
		- 在 Finetune 的时候，我们通常希望优先调整靠近输出层的参数，不要过度调整靠近输入层的参数，以免严重破坏预训练效果。而梯度消失意味着越靠近输入层，其结果对最终输出的影响越弱，这正好是 Finetune 时所希望的
		- 所以，预训练好的 Post Norm 模型，往往比 Pre Norm 模型有更好的 Finetune 效果
	- 同一设置之下，Pre Norm 结构往往更容易训练，但最终效果通常不如 Post Norm
	- 一个 L 层的 Pre Norm 模型，其实际等效层数不如 L 层的 Post Norm 模型
	- Pre Norm 结构无形地增加了模型的宽度而降低了模型的深度，而深度通常比宽度更重要，所以是无形之中的降低深度导致最终效果变差了。而 Post Norm 刚刚相反，它每 Norm 一次就削弱一次恒等分支的权重，所以 Post Norm 反而是更突出残差分支的，因此 Post Norm 中的层数更加 “足秤”，一旦训练好之后效果更优
	- see "DeepNet: Scaling Transformers to 1,000 Layers"

### ELMo

- 引入：以前方法的局限性
	- 多义词问题：one-hot、word2vec、fastText 为静态方式，即训练好后，每个词表达固定；
	- 单向性：one-hot、word2vec、fastText 都是从左向右学习，不能同时考虑两边信息
- 思想
	- 预训练时，使用语言模型学习一个单词的emb（多义词无法解决）
	- 使用时，单词间具有特定上下文，可根据上下文单词语义调整单词的emb表示（可解决多义词问题）
	- 因为预训练过程中，emlo 中的 lstm 能够学习到每个词对应的上下文信息并保存在网络中，在 fine-turning 时，下游任务能够对该网络进行 fine-turning，使其 学习到新特征
- 问题
	- 在做序列编码任务时，使用 LSTM 不如 attention
	- ELMo 采用双向拼接的融合特征，比Bert一体化融合特征方式弱

### BERT

- 模型架构
	- Transformer的双向编码器，旨在通过在左右上下文中共有的条件计算来预先训练来自无标号文本的深度双向表示
- 输入
	- Token embedding 字向量: 通过查询 WPE (word Piece Embedding) 字向量表将文本中的每个字转换为一维向量
	- Segment embedding 文本向量: 表示句子对 A/B 的向量，用来帮助 next sentence prediction task，对分类/tagging任务，B=空集
	- Position embedding 位置向量：由于出现在文本不同位置的字/词所携带的语义信息存在差异, 对不同位置的字/词分别附加一个不同的向量以作区分
- 特别注意
	- 最大长度：512，词汇表长度：3w
	- CLS：输入的第一个token，也是输出时的第一个token，表示分类任务的hidden state
	- SEP：区分句子对 A/B 的token
- 预训练任务
	- (MLM) Masked LM: 15% mask, 其中：80% [MASK], 10% random replace, 10% unchanged
		- 为什么不是 100% [MASK]: 微调期间从未看到[MASK]词块, 更好匹配预训练和微调
		- 为什么加 10% 概率不变: 将该表征偏向于实际观察到的单词
		- 但从 appendix 的不同 % 组合看，最多差1个点，问题不大
	- (NSP) Next Sentence Prediction, 其中：50% 真，50% 假
		- 为什么做这个任务：问答(QA)和自然语言推理(NLI)下游任务都是基于对两个文本句子间关系的理解，这种关系并非通过语言建模直接获得
		- 但roberta证明其实不需要？
	- 损失函数：两个预训练分类任务的和
- 与 Elmo 的区别
	- ELMo模型是通过语言模型任务得到句子中单词的embedding表示，以此作为补充的新特征给下游任务使用；bert直接对不同任务fine-tune embedding
- bert 的问题
	- MLM 中 mask 的词如果不是条件独立的话，预训练的假设不成立，但emperically影响不大
	- [MASK]在fine-tune中不会出现，导致预训练和微调不统一，但emperically也影响不大
	- [MASK] subword 可能有的词只mask了一部分，导致预测是根据没mask的部分来的，而不是上下文关系，可以做改动让整个词都mask
	- 收敛速度慢：每batch中只预测了15％的词块

**面试**

- 为什么bert三个embedding（token，segment，position）可以相加
	- 见https://www.zhihu.com/question/374835153/answer/1080315948
- 为什么bert中要用BPE这样的subword Token
	- 传统的处理方式往往是将这些 OOV 映射到一个特殊的符号，如 <UNK>，但这种方式无法充分利用 OOV 中的信息
	- BPE能很好的解决单词上的OOV，在语义粒度是比较合适的表达，一个token不会太大，也不会小到损失连接信息（如一个字母），使用subword把词拆碎，常见的typo或者语言特殊表达，都能有一部分照顾到
	- 中文是如何处理溢出词表词(oov)的语义学习的：中文是字级别，词级别的oov，在字级别上解决
	- 为什么以前char level/subword level的NLP模型表现一般都比较差，但是到了bert这里就比较好：模型参数和复杂度上来了
- 为什么bert要在开头加个[CLS]，为什么[CLS]可以建模整句话的语义表征
	- 这个无明显语义信息的符号会更“公平”地融合文本中各个词的语义信息，从而更好的表示整句话的语义
	- 为什么无明显语义？因为训练的时候BERT发现每个句子头都有，这样他能学到什么语义呢
	- 为什么要公平？因为控制变量，我们不希望做其他下游任务的时候用于区分不同样本间特征的信息是有偏的
	- 不放在句子开头的其他位置是否可行？一个未经考证的臆想是，任何其他位置的position embedding都无法满足放在开头的一致性。所以不同句子间可能会有一定的不同，这并不是我们做一些句间的分类问题想要的
	- 能不能不用[CLS]建模整句话的语义表征，用别的代替？对BERT的所有输出词向量（忽略[CLS]和[SEP]）应用MaxPooling或AvgPooling
- bert中有哪些地方用到了mask
	- 预训练任务：把输入的其中一部分词汇随机掩盖，模型的目标是预测这些掩盖词汇。这种训练方式使得每个位置的BERT都能学习到其上下文的信息
	- attention计算中的mask：不同样本的seq_len不一样，但是由于输出的seq_len需要一致，所以需要通过补padding来对齐，而在attention中我们不希望一个token去注意到这些padding的部分，因为实际场景下它们是不存在的，所以attention中的mask就是来处理掉这些无效的信息的。具体来说就是在softmax前每个都设为-inf（或者实际的场景一个很小的数就可以），然后过完softmax后"padding"部分的权重就会接近于零，query token就不会分配注意力权重
	- 下游任务的decoder：防止语言模型利用了leak的未来信息
- bert为什么要使用warmup的学习率trick: 为了让开始阶段的数据分布更好，更容易训练，防止过拟合到少数几个batch上
- 为什么说GPT是单向的bert是双向的: 双向指的是语义上的双向，GPT模仿了语言模型屏蔽了序列后面的位置。bert没有这样做，在self-attention上都可以彼此交互，但是GPT不行
- bert如何处理一词多义: 利用self-attention中词和词的交互
- bert中的transformer和原生的transformer有什么区别: 原生用的是周期函数对相对位置进行编码，bert换成了position embedding

### BERT Distilliation

**TinyBERT**

**DistilBERT**

**AlBERT**

- Albert是通过什么方法压缩网络参数的: 多层transformer共享参数，推断需要重复使用这些参数，时间没有减少

**FastBERT**

### BERT ++

**RoBERTa**

- BERT 全方位参数改进
	- 更大的模型参数量
	- 更大bacth size，尝试过从 256 到 8000 不等的bacth size
	- 更多的训练数据（包括：CC-NEWS 等在内的 160GB 纯文本，而最初的BERT使用16GB BookCorpus数据集和英语维基百科进行训练）
- BERT 预训练方法改进
	- 去掉下一句预测(NSP)任务
	- 动态掩码：BERT 依赖随机掩码和预测 token。原版的 BERT 实现在数据预处理期间执行一次掩码，得到一个静态掩码。 而 RoBERTa 使用了动态掩码：每次向模型输入一个序列时都会生成新的掩码模式。这样，在大量数据不断输入的过程中，模型会逐渐适应不同的掩码策略，学习不同的语言表征。
	- 文本编码。Byte-Pair Encoding（BPE）是字符级和词级别表征的混合，支持处理自然语言语料库中的众多常见词汇。原版的 BERT 实现使用字符级别的 BPE 词汇，大小为 30K，是在利用启发式分词规则对输入进行预处理之后学得的。Facebook 研究者没有采用这种方式，而是考虑用更大的 byte 级别 BPE 词汇表来训练 BERT，这一词汇表包含 50K 的 subword 单元，且没有对输入作任何额外的预处理或分词
- 贡献小结
	- remove NSP 删除下一个句子预测目标
	- full-sentence training 输入序列有可能跨越多个文章边界
	- dynamic masking 动态更改应用于训练数据的掩蔽模式
	- byte-level BPE

**SBERT**

**ColBERT**

**XLNet**

**Bart**

**ELECTRA**

### BERT Evaluation

**Perplexity**

- 引入
	- 困惑度，是信息论中的一个概念，可以用来衡量一个随机变量的不确定性，也可以用来衡量模型训练的好坏程度
	- 一个随机变量的Perplexity数值越高，代表其不确定性也越高；一个模型推理时的Perplexity数值越高，代表模型表现越差，反之亦然
	- 对于离散随机变量X，假设概率分布可以表示为p(x)，那么对应的困惑度为2 ** H(p)
	- 其中，H(p)为概率分布p的熵 = -sum(x in X: p(x) * log2p(x))
- 在NLP领域，对于语句s=w1,w2,w3,...,wn,其困惑度为
	- p(s) ** (-1/n) = mult(i=1->n: p(wi|w1,w2,...,wi-1)) ** (-1/n)
	- 显然，测试集中句子的概率越大，困惑度也就越小

**BLEU**

- BiLingual Evaluation Understudy: 衡量生成序列和参考序列之间的重合度
- PN(x): 生成序列中的N元组合词在参考序列中出现的比例
	- 假设当前有一句源文s，以及相应的译文参考序列r1,r2,...,rn
	- 机器翻译模型根据源文s生成了一个生成序列x，且W为根据候选序列x生成的N元单词组合，这些N元组合的精度为
	- PN(x) = sum(w in W: min(cw(x), max(k=1->n, cw(rk)))) / sum(w in W: cw(x))
	- 其中，cw(x)为N元组合词w在生成序列x中出现的次数，cw(rk)为N元组合词w在参考序列rk中出现的次数
- PN(x) 的核心思想是衡量生成序列x中的N元组合词是否在参考序列中出现，其计算结果更偏好短的生成序列，即生成序列x越短，精度PN(x)会越高
- b(x): 长度惩罚因子
	- 如果生成序列x比参考序列rk短，则会对该生成序列x进行惩罚
	- b(x) = 1 if lx > lr; = exp(1-ls/lr) if ls <= lr
	- 其中，lx表示生成序列x的长度， lr表示参考序列lr的最短长度
- BLEU
	- 我们可以根据生成序列x构造不同长度的N元组合词，这样便可以获得不同长度组合词的精度
	- BLEU算法通过计算不同长度的N元组合的精度PN(x)，N=1,2,3...，并对其进行几何加权平均得到
	- BLEU算法的值域范围是[0,1]，数值越大，表示生成的质量越好
- 评价
	- BLEU算法能够比较好地计算生成序列x的字词是否在参考序列中出现过，但是其并没有关注参考序列中的字词是否在生成序列出现过，即BLEU只关心生成的序列精度，而不关心其召回率

**ROUGE**

- 引入
	- BLEU算法只关心生成序列的字词是否在参考序列中出现，而不关心参考序列中的字词是否在生成序列中出现，这在实际指标评估过程中可能会带来一些影响，从而不能较好评估生成序列的质量
	- ROUGE (Recall-Oriented Understudy for Gisting Evaluation)算法便是一种解决方案，它能够衡量参考序列中的字词是在生成序列中出现过，即它能够衡量生成序列的召回率
- ROUGE算法能比较好地计算参考序列中的字词是否在生成序列出现过，但没有关注生成序列的字词是否在参考序列中出现过，即ROUGE算法只关心生成序列的召回率，而不关心准确率

### LLM - 2022, the post-GPT era

**Optimizer**

- AdamW
	- L2 正则化是减少过拟合的经典方法，它会向损失函数添加由模型所有权重的平方和组成的惩罚项，并乘上特定的超参数以控制惩罚力度
	- AdamW本质上就是在损失函数里面加入了L2正则项，然后计算梯度和更新参数的时候都需要考虑这个正则项
	- AdamW使用在hugging face版的transformer中,BERT,XLNET,ELECTRA等主流的NLP模型，都是用了AdamW优化器

**Tokenizer**

- https://zhuanlan.zhihu.com/p/626080766 (todo)
- https://zhuanlan.zhihu.com/p/639557870 (todo)
- word和character的方法已经过时，目前我们只关注subword tokenization
- BPE，BBPE
	- 字节对编码，将最常出现的子词对合并，直到词汇表达到预定的大小时停止
- WordPiece
	- 子词粒度的tokenize算法，BERT/DistilBERT/Electra都使用了它
	- 它的原理非常接近BPE，不同之处在于它做合并时，并不是直接找最高频的组合，而是找能够最大化训练数据似然的merge
	- 即它每次合并的两个字符串A和B，应该具有最大的P(AB)/(P(A) * P(B))值
	- 合并AB之后，所有原来切成A+B两个tokens的就只保留AB一个token，整个训练集上最大似然变化量与P(AB)/(P(A) * P(B))成正比
- Unigram
	- 从一个巨大的词汇表出发，再逐渐删除trim down其中的词汇，直到size满足预定义
	- 初始的词汇表可以采用所有预分词器分出来的词，再加上所有高频的子串。
每次从词汇表中删除词汇的原则是使预定义的损失最小

**Architecture**

- Causal Decoder (Anything else)
- Prefix Decoder (GLM)
- MoE (Switch Transformer)
- SSM (S4)

**Layer Norm**

- https://zhuanlan.zhihu.com/p/643829565#ref_7 (todo)
- https://zhuanlan.zhihu.com/p/86765356 (todo)
- norm position vs. norm method
- pre RMSNorm (LLaMa, T5, Gopher, Chinchilla)
	- 对LayerNorm的一个改进，没有做re-center操作（移除了其中的均值项），可以看作LayerNorm在均值为0时的一个特例
- post DeepNorm (GLM)
- pre LayerNorm (GPT3, PanGU, OPT, PaLM, BLOOM, Galactica)
- Sandwich-LN

**Activation**

- ReLU
- GeLU
- Swish
- SwiGLU
- GeGLU

**Position Encoding**

- Absolute: xi = xi + pi
- Relative: Aij = Wq xi xj.T Wk.T + r_{i-j}
- T5 bias
- ALiBi
- RoPE
- xPos
- position encoding/embedding 区别
	- encoding固定式，embedding学习式
	- embedding，存一个hidden_dim x seq_len 的可学习矩阵，问题：不能随着seq_len的增加而改变
	- encoding，周期函数，为什么可以用周期函数？注意，不会走完一个周期，同时使用sin和cos三角函数时，第k个位置的encoding可以被第0个位置的encoding线性表示

**Attention**

- Multi-Head Attention (MHA)
- Multi-Query Attention (MQA)
- FlashAttention
- PageAttention

**Complexity Analysis**

- https://zhuanlan.zhihu.com/p/624740065 (todo)
- 现在都是fp16或者bf16训练和推理，那么如果是1个100亿参数量的模型（也就是储存了100亿个参数），它其实是一个10B大小的模型。（1Billion等于10的9次方，也就是10亿）
- 1个字节占8bits，那么fp16就是占2个字节（Byte），那么10B模型的模型大小是20GB，是 * 2的关系
- 那么对于 N billion 的模型
	- 推理时显存的下限是 2n GB ，至少要把模型加载完全
	- 训练时，如果用Adam优化器，有个2+2+12的公式，训练时显存下限是16n GB，需要把模型参数、梯度和优化器状态（4+4+4），保持在显存，具体可以参考微软的ZeRO论文

**Adaptation Tuning**

- Instruction Tuning
- Alignment Tuning
- Parameter-Efficient Model Adaptation
- Memory-Efficient Model Adaptation

**Utilization**

- In-Context Learning (ICL)
- Chain-of-Thought Reasoning (CoT)
	- tldr: few-shot learning by adding reasoning steps
	- additional token generated while doing the chain reasoning process is important for the final answer generation
	- this is also more interpretable to human
	- self-consistency improves CoT
		- samples multiple outputs, majority vote
	- auto-CoT
		- no need to label, just do "lets think step by step" and let the model generate the reasoing steps instead
		- no need to verify the model's intermediate outputs correctness either
		- shows the reasoning format is more important than reasoning correctness
- Planning for Complex Task Solving

**Evaluation**

### InstructGPT

- method
	- supervised pre-training (SFT) with prompt dataset (from API and labeler-written)
	- reward model (RM) fine-tuning with ranking dataset (from API and labeler-ranked)
	- RLHF (PPO-ptx) fine-tuning with RM (from API, no-labeler)
- data collection
	- 40 contractors, UpWork, ScaleAI
	- test on helpfulness
	- evaluate by truthfullness and harmlessness
	- inter-annotator agreement rate
- model 1: SFT
	- 16 epochs, overfit validation after 1 epoch
	- keep training is better caz step 2, 3 would align it back
- model 2: RM
	- final unembedding layer of SFT to FC -> scalar reward, only 6B param - more stable
	- pairwise ranking loss (PRL): maximize difference of higher ranked score to lower ranked score
	- multi-class ranking loss: for one batch, rank K = 4 to 9 potential answers, select pair of repsonses, softmax, maximize the expectation of PRL, average by k-choose-2
	- why not PRL directly: less over-fitting, more robust
- model 3: RLHF
	- the key difference of RL to supervised in that the data distribution changes wrt. reward model's change
	- init a RL-policy model (MRL) to our SFT model
	- objective: maximize A + f * B, f init to 0
	- A: expectation of the reward generated by MRL on the ranking dataset, with penalty on the difference in data distribution of MRL and SFT (KL-divergence) - by log division of 2 models on softmax % output - to encourage MRL to not diverge too much from the original prompt dataset
	- B: expectation of the % output generated by MRL on the prompt dataset
- model 2, 3 takeaway
	- essentially, we can just do step 1 if we have infinite labelers
	- but since labelers is costly, we first learn a reward model to automatically rank the generated responses in step 2
	- then we want the model to generate higher ranks possible in step 3, measured by RM, also not shift from prompt dataset too much
	- RLHF is good when human are not sure about how to design an objective to fit their needs, instead now they can rank stuff to achieve this
- aside
	- previous openai papers on RL also tried to do this 2, 3 round back and forward a couple of times, but seem to do little better, especially when the task is not very hard, so no need to do it
	- step 2, 3 model alignment data << step 1 data
	- future: how to eval is important, training objective, etc.
	- current main objective: helpfulness, side objective: truthfullness, harmfulness

### LLM ++

- [denote close source]

**2019**

- T5 (Google)
	- todo

**2020**

- GPT-3 (OpenAI)
	- todo
- [GShard] (Google)
- mT5 (Google)

**2021**

- PanGu-a (HuaWei)
- PLUG (Alibaba)
- [Codex] (OpenAI)
- [Ernie 3.0] (Baidu)
	- todo
- [Jurassic-1] (AI21 Labs, 以色列)
- CPM-2 (BAAI, 智源)
- T0 (HuggingFace)
	- todo
- [HyperCLOVA] (NAVER, 韩国)
- [FLAN] (Google)
	- todo
- [Yuan 1.0] (Inspur, 浪潮)
- [Anthropic] (Anthropic.ai)
- [WebGPT] (OpenAI)
- [Ernie 3.0 Titan] (Baidu)
- [Gopher] (DeepMind)
- [GLaM] (Google)

**2022**

- [InstructGPT] (OpenAI)
- CodeGen (Salesforce)
- [MT-NLG] (Microsoft)
- [LaMDA] (Google)
- [AlphaCode] (DeepMind)
- [Chinchilla] (DeepMind)
	- todo
- OPT (Meta)
	- todo
- GPT-NeoX-20B (EleutherAI, OpenSourced-OpenAI)
- TK-Instruct (AllenAI)
- [Cohere] (Cohere)
- UL2 (Google)
- PaLM (Google)
	- todo
- [YaLM] (Yandex, 俄罗斯)
- GLM (Tsinghua)
	- todo
- [AlexaTM] (Amazon)
- [WeLM] (Tencent Weixin)
- [Sparrow] (DeepMind)
- Flan-T5 (Google)
	- todo
- [Flan-PaLM] (Google)
- [Luminous] (Luminous.ai)
- NLLB (Meta)
- BLOOM (Huggingface)
	- todo
- mTo (Huggingface)
- BLOOMZ (Huggingface)
- Galatica (Meta)
- OPT-IML (Meta)
- [ChatGPT] (OpenAI)

**2023**

- Falcon (Tii UAE, 阿联酋)
- CodeGeeX (Tsinghua KG)
- Pythia (EleutherAI)
- Vicuna (LM-Sys)
	- todo
- [PanGu-s] (HuaWei)
- [Bard] (Google)
- LLaMa (Meta)
	- RoPE: 最主流的位置编码
	- KV cache: decoder-only模型必备的加速解码手段
	- decoder-only模型是单向注意力，attention mask是下三角阵
	- 模型参数大都是fp16的，但softmax时用fp32计算
	- BBPE: Byte-level BPE - 好处：很多字符出现的频率很低频，但是词表会变得很大，以至于存储和训练的成本都很高，并且稀有词往往很难学好，BBPE用UTF-8给所有可能的字符都编码，词表可以相比于BPE减少到1/8
	- 为什么LLaMA词表里只有700+的汉字，但是也能表示中文: 用BBPE，不存在的汉字用字节合成就可以
	- 既然LLaMA的分词器是基于UTF-8的BBPE，是不是所有的模型通用一个分词器即可，不需额外训练：不可以，LLaMA尽管能支持中文，但是表示效率很低，一个中文字符在UTF-8表示为3个字节，意味着：最差情况下，1个汉字要编码成3个token，但是如果在中文语料上训练分词器，很多2个汉字组成的词会被编码为1个token。大模型是逐个token生成，推理速度都是按token计算的，所以如果推理速度相同，LLaMA生成中文的速度会远慢于在专门在中文上训练的GLM系列模型
- GPT-4
- Alpaca
	- todo
- Claude
- LLaMa2
	- todo

### LLM Aside

**Gradient Caching**

**FlashAttention**

- https://zhuanlan.zhihu.com/p/639228219 (todo)
- 在标准注意力实现中，注意力的性能主要受限于内存带宽，是内存受限的。频繁地从HBM中读写NxN的矩阵是影响性能的主要瓶颈。针对内存受限的标准注意力，Flash Attention是IO感知的，目标是避免频繁地从HBM中读写数据
- tiling 分块计算
	- 从GPU显存分级来看，SRAM的读写速度比HBM高一个数量级，但内存大小要小很多。通过kernel融合的方式，将多个操作融合为一个操作，利用高速的SRAM进行计算，可以减少读写HBM的次数，从而有效减少内存受限操作的运行时间。但SRAM的内存大小有限，不可能一次性计算完整的注意力，因此必须进行分块计算，使得分块计算需要的内存不超过SRAM的大小
	- 为什么要进行分块计算？内存受限 --> 减少HBM读写次数 --> kernel融合 --> 满足SRAM的内存大小 --> 分块计算。因此分块大小block_size不能太大，否则会导致OOM
	- 分块计算的难点是什么呢？注意力机制的计算过程是“矩阵乘法 --> scale --> mask --> softmax --> dropout --> 矩阵乘法”，矩阵乘法和逐点操作（scale，mask，dropout）的分块计算是容易实现的，难点在于softmax的分块计算。由于计算softmax的归一化因子（分母）时，需要获取到完整的输入数据，进行分块计算的难度比较大。论文中也是重点对softmax的分块计算进行了阐述
	- 主要是把softmax上下两部分解耦合，然后并行计算各个block的结果
- 重计算 todo
- kernel 融合
- 后向传递
	- 在标准注意力实现中，后向传递计算Q,K,V的梯度时，需要用到  的中间矩阵，Flash-Attention没有保存这两个矩阵，而是保存了两个统计量m(x), l(x)，在后向传递时进行重计算

**KV Cache**

- inference时
	- 预填充阶段：输入一个prompt序列，为每个transformer层生成 key cache和value cache（KV cache）
	- 解码阶段：使用并更新KV cache，一个接一个地生成词，当前生成的词依赖于之前已经生成的词
- 显存占用分析
	- 假设输入序列的长度为input_len，输出序列的长度为 output_len, 以float16来保存KV cache，那么KV cache的峰值显存占用大小为 batch_size * (input_len + output_len) * num_layer * 2 * 2，这里第一个2表示K/V cache，第二个2表示float16占2个bytes

**Beam Search**

- todo

### RLHF

**面试**

- SFT（有监督微调）的数据集格式：一问一答
- RM（奖励模型）的数据格式：一个问题 + 一条好回答样例 + 一条差回答样例
- PPO（强化学习）的数据格式：理论上来说，不需要新增数据。需要提供一些prompt，可以直接用sft阶段的问。另外，需要限制模型不要偏离原模型太远（ptx loss），也可以直接用sft的数据
- 奖励模型需要和基础模型一致吗？
	- 不同实现方式似乎限制不同。（待实践确认）colossal-ai的coati中需要模型有相同的tokenizer，所以选模型只能从同系列中找。在ppo算法实现方式上据说trlx是最符合论文的
- LoRA权重是否可以合入原模型
	- 可以，将训练好的低秩矩阵（B * A）+原模型权重合并（相加），计算出新的权重

### GPT-4 and future

- engineering
	- predictive scaling
	- longer context
	- openai eval github
- capability
	- text and visual input: mutlimodal sentence embedding from tokenizer and visual encoder (ViT)? - objective still next token prediction
	- text prompt to visual (by code), music (by signal) output
	- steerability: generation with NPC style by prompting (system message)
- impact
	- 80% work impact by 10%+, 19% work impact by 50%+
	- science and critical thinking are negatively associated with LLMs, programming and writing are positively associated with LLMs
- to think about
	- confidently prediction wrongly -> post-training alignment -> more factual, less confidence, subjectiveness
	- low coding correctness (data contamination?) -> some claim that correct prompting fix most of it
	- bias, risk issue: human experts, safety reward signal by RLHF
- paradigm shift
	- lack of planning, long-term memory, backtrack (autoregressive?) -> LLM as agent?
	- simple reasoning, logical, inconsistency errors
	- confidence calibration - user persuasion still leads to factual errors (caz of RLHF?)
	- continual learning: update itself during changing environment
	- personalization
