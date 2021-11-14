# 问题汇总

### 判别式和生成式模型的区别

- 生成式模型估计它们的联合概率分布P(x,y)，关注数据是如何生成的，反映同类数据本身的相似度，不关心到底划分不同类的边界在哪
- 判别式模型估计条件概率分布P(y|x)，关注类别之间的差别
- 生成式模型可以根据贝叶斯公式得到判别式模型，但反过来不行
- 判别式模型主要有：
	- logit R, Linear R
	- SVM, Neural Network
	- KNN
	- Gaussian Process高斯过程
	- CRF条件随机场
	- Boosting
	- CART
	- LDA(linear Discriminant analysis)线性判别分析
- 生成式模型主要有:
	- NB(Naive Bayes), BN(Bayes Network)
	- HMM
	- Mixture Gaussians
	- Sigmoid Belief Network
	- Markov Random Fields
	- DBN深度信念网络
	- Latent Dirichlet Allocation

### 时间序列模型

- AR模型：自回归模型，是一种线性模型，已知N个数据，可由模型推出第N点前面或后面的数据（设推出P点），所以其本质类似于插值
- MA模型：移动平均法模型，使用趋势移动平均法建立直线趋势的预测模型
- ARMA模型：自回归滑动平均模型，模型参量法高分辨率谱分析方法之一，研究平稳随机过程有理谱的典型方法。它比AR模型法与MA模型法有较精确的谱估计及较优良的谱分辨率性能，但参数估算比较繁琐
- GARCH模型：广义ARCH模型，是ARCH模型的拓展， GARCH对误差的 方差进行建模，特别适用于波动性的分析和 预测

**无监督学习**

- 模型
	- 聚类
	- 自编码器auto-encoder
	- 主成分分析PCA
	- GAN
- 算法
	- EM

### GBDT和RF比较

- 参考bagging和boosting的对比
- 都是由多棵树组成和决定结果
- RF可以是分类树，也可以是回归树；GBDT只由回归树组成
- RF可以并行；GBDT只能串行
- RF采用多数投票；GBDT将结果加权累加
- RF对异常值不敏感，GBDT对异常值非常敏感
- RF对数据一视同仁，GBDT是基于权值的弱分类器的集成
- RF减少variance提高性能，GBDT减少bias提高性能

### Dropout测试阶段如何处理

- Dropout在训练时采用是为了减少神经元对部分上层神经元的依赖，类似将多个不同网络结构的模型集成起来，减少过拟合的风险
- 而在测试时，应该用整个训练好的模型，因此不需要dropout
- 如何平衡Dropout在训练和测试时的差异
	- 假设失活概率为p，就是一层中的每个神经元都有p的概率失活，因为实际测试没有dropout，输出层每个神经元的输入和的期望会有量级上的差异
	- 因此在训练时要对输出数据除以（1-p）之后再传给输出层神经元，作为神经元失活的补偿，以使得在训练时和测试时每一层输入有大致相同的期望
- BN和Dropout共同使用时会出现的问题(https://arxiv.org/abs/1801.05134)
	- BN和Dropout单独使用都能减少过拟合并加速训练速度，但如果一起使用可能会得到比单独使用更差的效果
	- 冲突的关键是网络状态切换过程中存在神经方差的（neural variance）不一致行为
	- 试想若有图一中的神经响应X，当网络从训练转为测试时，Dropout 可以通过其随机失活保留率（即p）来缩放响应，并在学习中改变神经元的方差，而 BN 仍然维持 X 的统计滑动方差
	- 这种方差不匹配可能导致数值不稳定，而随着网络越来越深，最终预测的数值偏差可能会累计，从而降低系统的性能
	- 简单起见，作者们将这一现象命名为「方差偏移」
	- 解决方案
		- 在所有 BN 层后使用 Dropout
		- 修改 Dropout 的公式让它对方差并不那么敏感，用高斯Dropout，可以使模型对方差的偏移的敏感度降低，总得来说就是整体方差偏地没有那么厉害了

### 1x1卷积(NiN, Inception->ResNet)

- 升维/降维，跨通道信息交互
	- 1x1卷积在输入中滑动，按照卷积核channel的权重计算输入channel之间的加权求和，输出到filter数量的通道中
	- 输入有多个channel时，1x1卷积对每个像素点在不同的channel上进行了线性的信息整合，且保留了图片的原有平面结构
	- 降维时极大减少了计算量，然后升维恢复网络形态，bottleneck design
- 增加非线性
	- 1x1卷积后接非线性激活函数，同时增加了网络深度
	- 可以理解为输入的一种映射关系，一个针对现有feature map的概括归纳，也可以理解为一次更高维的特征的提取整合
	- 不同的filter(不同的weight和bias)，卷积以后得到不同的feature map，提取不同的特征，得到对应的specialized neuron，可以建立多种映射关系，得到一个更加丰富的概括结论

### softmax

- max：极值，数组中最大的元素
- soft：对应hard，元素的绝对确定性质，soft指代相对以概率为比重的性质
- softmax的含义在于不唯一的确定某一个最大值，而是为每个输出分类的结果都赋予一个概率，表示属于每个类别的可能性
- 一般公式为softmax(zi)=zi/sum(i: zi)，用的更多的是其指数形式pi=softmax(zi)=e^zi/sum(i: e^zi)
- 指数形式优点
	- 指数函数曲线呈递增趋势，最重要的是斜率逐渐增大，也就是说在x轴上一个很小的变化，可以导致y轴上很大的变化，这种函数曲线能够将输出的数值拉开距离
	- 在深度学习中通常使用反向传播求解梯度，进而使用梯度下降进行参数更新，指数函数不容易导致梯度消失，计算方便
- 指数形式缺点
	- zi值非常大时，数值可能会溢出
	- 一般把softmax函数和交叉熵损失函数统一使用，就处理了数值的溢出或不稳定
- 原始指数形式求导
	- 当i=j时，dpj/dzi=(e^zj/sum(j: e^zj))'=(chain role) ((e^zj)'sum(j: e^zj)-e^zj e^zj)/(sum(j: e^zj)^2)=e^zj/sum(j, e^zj)-(e^zj/sum(j: e^zj))^2=pj(1-pj)
	- 当i!=j时，dpj/dzi=(e^zj/sum(j: e^zj))'= (0 sum(j: e^zj)-e^zj e^zi)/(sum(j: e^zj))^2=-(e^zj/sum(j: e^zj))(e^zi/sum(j: e^zj))=-pjpi
- log似然函数（交叉熵）指数形式求导
	- 交叉熵最终公式：-reduce_sum(labels * log(softmax_out))
	- 式子1-log似然函数: loss(i)=-log(pi)=-log(e^zi/sum(i: e^zi))=-(zi-log(sum(i: e^zi)))
	- 式子2-交叉熵：L=-sum(i: yi log(pi))
	- 这两者其实本质是一样的，对于式子1来说，只针对正确类别的对应的输出节点，将这个位置的softmax值最大化，而式子2则是直接衡量真实分布和实际输出的分布之间的距离
	- 交叉熵梯度推导：dL/dzi=-sum(i: yi dlog(pi)/dzi)=-sum(i: yi dlog(pi/dpi) dpi/dzi)=-sum(i: yi 1/pi dpi/dzi)，其中dpi/dzi就是之前原始指数形式的导数
	- 所以得到-yi(1-pi)-sum(j!=i: yj 1/pj (-pjpi))=-yi(1-pi)+sum(j!=i: yjpi)=-yi+yipi+sum(j!=i: yjpi)
	- 提取公共项pi：dL/dzi=pi(yi+sum(j!=i: yj))-yi
	- 因为分类的和sum(i: yi)=1, yi+sum(j!=i: yj)=1
	- 最终简化为dL/dzi=pi-yi

### 梯度消失和梯度爆炸

- gradient vanishing, gradient exploding problem
- 都是因为网络太深，网络权值更新不稳定造成的
- 本质是因为梯度反向传播中的连乘效应
- 解决梯度消失
	- 用ReLU激活函数取代sigmoid（这个主要是梯度消失）
	- LSTM的结构可以改善RNN中的梯度消失问题
	- ResNet的残差结构
- 解决梯度爆炸
	- gradient clipping，设置阈值

### Numpy数组和pytorch tensor对比

- torch.Tensor(...)是类构造函数，使用全局缺省值float，创建了额外的数据副本
- torch.tensor(...)是工厂类函数，根据输入推断数据类型，创建了额外的数据副本，是Pytorch类的实例
- torch.from_numpy(...)是工厂类函数，根据输入推断数据类型，没有创建额外的数据副本，共享数据内存
- torch.as_tensor(...)是工厂类函数，根据输入推断数据类型，没有创建额外的数据副本，共享数据内存
- 后两种做内存优化用as_tensor多，因为它接受任何像Python数据结构这样的数组

### Tensorflow(Google)和Pytorch(Facebook)对比

- 图的创建及调试
	- pytorch 图结构的创建是动态的，即图是运行时创建；pytorch代码更易调试，可以像调试python代码一样利用pdp在任何地方设置断点
	- tensorflow 图结构的创建是静态的，即图首先被"编译"，然后再运行；不易调试，要么从会话请求检查变量，要么学习使用tfdbg调试器
- 灵活性
	- tensorflow：静态计算图，数据参数在CPU与GPU之间迁移麻烦，调试麻烦
	- pytorch：动态计算图，数据参数在CPU与GPU之间迁移十分灵活，调试简便
- 设备管理(内存 显存)
	- tensorflow：不需要手动调整，简单
		- TensorFlow的设备管理非常好用。通常你不需要进行调整，因为默认的设置就很好。例如，TensorFlow会假设你想运行在GPU上（如果有的话）
		- 缺点：默认情况下，它会占用所有的GPU显存。简单的解决办法是指定CUDA_VISIBLE_DEVICES。有时候忘了这一点，GPU在空闲的时候，也会显得很忙
	- pytorch：需要明确启用的设备，启用CUDA时，需要明确把一切移入设备
		- 缺点：代码需要频繁的检查CUDA是否可用，及明确的设备管理，在编写能同时在CPU和GPU上运行的代码时尤其如此
- 总结
	- tensorflow
		- 基于静态图
		- 不易调试
		- 上手需学习额外概念—会话、图、变量范围、占位符
		- 序列化更强大
		- 支持移动和嵌入式部署
		- 数据加载比较复杂
		- 设备管理默认即可
		- 强大的可视化工具TensorBoard
		- 支持分布式执行、大规模分布式训练
	- pytorch
		- 基于动态图
		- 容易理解且易调试
		- 结合NumPy更易上手
		- 序列化的API比较简单
		- 不支持移动和嵌入式部署
		- 数据加载 API 设计得很好
		- 设备管理必须指定
		- 可视化只能调用matplotlib 、seaborn等库
		- 支持分布式执行、暂时不支持分布式训练

### 反卷积(转置卷积Transposed Convolution)

- 出现在FCN图像语义分割中，是upsample方法的一种
- upsample：使大小比原图像小得多的特征图变大，使其大小为原图像大小
- 卷积运算可表示为y = Cx，而卷积的反向传播相当于乘以C^T
- 卷积是多对一的关系，逆卷积是一对多的关系
- 通过反卷积可以可视化卷积的过程，在GAN等领域中有着大量的应用
- 重要特性
	- 通过反卷积并不能还原卷积之前的矩阵，只能从大小上进行还原，反卷积的本质还是卷积，只是在进行卷积之前，会进行一个自动的padding补0，从而使得输出的矩阵与指定输出矩阵的shape相同
	- 在进行反卷积的时候设置的stride并不是指反卷积在进行卷积时候卷积核的移动步长，而是被卷积矩阵填充的padding，在输入矩阵之间有一行和一列0的填充

### 神经网络参数总结

- （大致顺序）初始学习率，优化方法，初始权重，隐藏单元，epoch，网络层数，学习率衰减，激活函数，batch size，BN，正则（dropout l1 l2等）
