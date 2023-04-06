# NLP

- source: [https://www.youtube.com/playlist?list=PLoROMvodv4rOSH4v6133s9LFPRHjEmbmJ](https://www.youtube.com/playlist?list=PLoROMvodv4rOSH4v6133s9LFPRHjEmbmJ)

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

### BERT Distilliation

### GPT-1, 2, 3

### COT (Chain of Thought)

- tldr: few-shot learning by adding reasoning steps
- additional token generated while doing the chain reasoning process is important for the final answer generation
- this is also more interpretable to human
- self-consistency improves CoT
	- samples multiple outputs, majority vote
- auto-CoT
	- no need to label, just do "lets think step by step" and let the model generate the reasoing steps instead
	- no need to verify the model's intermediate outputs correctness either
	- shows the reasoning format is more important than reasoning correctness

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

### GPT-4

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
	- generating high-level content, lack plans, coherence
	- context length limitation - no memory
	- simple reasoning, logical, inconsistency errors
	- user persuasion still leads to factual errors (caz of RLHF?)
