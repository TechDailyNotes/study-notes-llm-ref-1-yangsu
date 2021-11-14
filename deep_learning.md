# Deep Learning

### 神经网络简述

- 神经元：以x1,x2,⋅⋅⋅,xK以及截距b为输入值，其输出a=σ(w1a1+...+wKaK+b)，其中σ为激活函数
- 神经网络：神经元的联结，一个神经元的输出可以是另一神经元的输入
- 矩阵表示：每一层的W,a等
- 损失函数：普遍有批量梯度下降法，带有正则项（权重衰减项）的平方误差损失函数
- 反向传播：链式求导法则的应用，通过单个样例的偏导推导整体损失函数的偏导

### 神经网络训练方法

**Gradient Descent, 1951**

- 常用的梯度下降法BGD、SGD和MBGD，不同之处在于对目标函数进行梯度更新时所使用的样本量的多少
- BGD批量梯度下降
	- 在更新每一参数时都使用所有的样本来进行更新
	- 优点：全局最优解；易于并行
	- 缺点：当样本量很大时，训练过程会很慢
- SGD随机梯度下降
	- 在更新每一参数时通过每个样本迭代更新一次
	- 优点：训练速度快
	- 缺点：准确度下降，并不是全局最优；不易于并行
- MSGD小批量梯度下降
	- 前两种的折中，每次更新选择batch size
- 梯度下降的局限
	- 选择一个合适的learning rate很难
		- 学习速率过小，会导致收敛速度很慢
		- 学习速率过大，会阻碍收敛，即在极值点附近振荡
	- 学习速率调整（又称学习速率调度，Learning rate schedules）试图在每次更新过程中，改变学习速率
		- 如模拟退火simulate annealing按照预先设定的调度算法或者当相邻的迭代中目标变化小于一个阈值时候减小学习速率
		- 但是梯度下降的调度和阈值需要预先设置，无法对数据集特征进行自适应
	- 模型所有参数的每次更新都是使用*相同*的学习速率
		- 如果数据很稀疏并且特征出现的次数不同，可能不希望所有的参数以某种相同的幅度进行更新，而是针对很少出现的特征进行一次大幅度更新
	- 在神经网络中常见的极小化highly non-convex error functions的一个关键挑战是避免步入大量的suboptimal local minima
		- Dauphin等人认为实践中的困难来自saddle points而非local minima
		- 这些saddle points（鞍点）经常被一个相等误差的平原包围，导致SGD很难摆脱，因为梯度在所有方向都趋于0

**Momentum**

- 一种启发式算法，形式为 $v_t=γv_{t−1}+η∇\_θJ(θ), θ=θ−v_t$
- 用物理上的动能势能转换来理解，即物体在这一时刻的动能=物体在上一时刻的动能+上一时刻的势能差，由于有阻力和转换时的损失，所以两者都乘以一个系数
- 就像一个小球从坡上向下滚，当前的速度取决于上一时刻的速度和势能的改变量
- 这样在更新参数时，除了考虑梯度，还考虑了上一时刻参数的历史变更幅度
	- 例如参数上一次更新幅度较大，并且梯度也较大，那么在更新时得更加猛烈一些
- Movement=上一时刻参数的变更幅度+梯度

**NAG, 1983**

- Nesterov accelerated gradient
- 以上面小球的例子看，momentum方式下小球完全是盲目被动的方式滚下的
	- 缺点是在邻近最优点附近是控制不住速度的
	- 我们希望小球可以预判后面的“地形”，如果后面地形还是很陡峭，那就继续坚定不移地大胆走下去，不然的话就减缓速度
- 动量的公式更改为 $v_t=γv_{t−1}+η∇\_θJ(θ-γv_{t−1}), θ=θ−v_t$
- 相比动量方式考虑上一时刻的动能和当前点的梯度，NAG考虑上一时刻的梯度和近似下一点的梯度，这使得它可以先往前探探路，然后慎重前进

**Simulated Annealing**

- 学习率退火
- 如果学习率很高，系统的动能就过大，参数向量就会无规律地变动，无法稳定到损失函数更深更窄的部分去
- 如果慢慢减小，可能在很长时间内只能浪费计算资源然后混沌地跳动，实际进展很少
- 如果快速地减少，系统可能过快地失去能量，不能到达原本可以到达的最好位置
- 三种实现方式
	- 随步数衰减：每进行几个周期就根据一些因素降低学习率
		- 通常是每过5个周期就将学习率减少一半，或者每20个周期减少到之前的十分之一
		- 数值的设定严重依赖具体问题和模型的选择
		- 在实践中可能看见这么一种经验做法：使用一个固定的学习率来进行训练的同时观察验证集错误率，每当验证集错误率停止下降，就乘以一个常数（比如0.5）来降低学习率
	- 指数衰减
		- 数学公式是α=α0e−kt，其中α0,k是超参数，t是迭代次数（也可以使用周期作为单位）
	- 1/t衰减
		- 数学公式是α=α0/(1+kt)，其中α0,k是超参数，t是迭代次数

- 下面为自适应学习率方法

**Adagrad, 2011**

- 令第i个参数的第t步更新梯度为$g_{t,i}=∇\_θJ(θ_j)$
- 一般的参数更新方式：$θ_{t+1,i}=θ_{t,i}-ηg_{t,i}$，对
- Adagrad：$θ_{t+1,i}=θ_{t,i}-\frac{ηg_{t,i}}{\sqrt(\sum_{i=0}^t (g^i)^2+ϵ)}$
- 实质上是对学习率形成了一个约束项regularizer$\frac{}{\sqrt(\sum_{i=0}^t (g^i)^2+ϵ)}$
	- 前项是对直至t次迭代的梯度平方和的累加和
	- ϵ是防止分母为0的很小的平滑项
	- 不用平方根操作性能会差很多
- 可以将到累加的梯度平方和放在一个对角矩阵中Gt∈Rd×d中
	- 其中每个对角元素(i,i)是参数θi到时刻t为止所有时刻梯度的平方之和
	- 由于Gt的对角包含着所有参数过去时刻的平方之和，可以通过在Gt和gt执行element-wise matrix vector mulitiplication来向量化我们的操作
	- $θ_{t+1,i}=θ_{t,i}-\frac{η}{\sqrt(G_t+ϵ)}⊙g_t$
- 优点
	- 学习速率自适应于参数
	- 在前期梯度较小的时候，regularizer较大，能够放大梯度
	- 后期gt较大的时候，regularizer较小，能够约束梯度
	- 所以它非常适合处理稀疏sparse数据
	- Dean等人发现Adagrad大大地提高了SGD的robustness并在谷歌的大规模神经网络训练中采用了它进行参数更新，其中包含了在Youtube视频中进行猫脸识别
	- 此外，由于低频词（参数）需要更大幅度的更新，Pennington等人在GloVe word embeddings的训练中也采用了Adagrad
- 缺点
	- 由公式可以看出，仍依赖于人工设置一个全局学习率
	- η设置过大的话，会使得regularizer过于敏感，对梯度的调节太大
	- 中后期，分母上梯度平方的累加将会越来越大，使得梯度为0，训练提前结束

**RMSprop**

- Root Mean Square of the gradients with previous gradients being decayed
- 一个没有公开发表的适应性学习率方法，使用了梯度平方的滑动平均，仍然是基于梯度的大小来对每个权重的学习率进行修改，让Adagrad不过于激进而过早停止学习
- 和Adagrad不同的是，其更新不会让学习率单调变小

**Adadelta, 2012**

- Adagrad的一种扩展，以缓解Adagrad学习速率单调递减的问题
- Adadelta不是对过去所有时刻的梯度平方进行累加，而是递归地定义为过去所有时刻梯度平方的decaying average $E[g^2]\_t$，t时刻的running average仅仅依赖于之前average和当前的梯度
- 于是Adagrad中的Gt被替换为过去时刻梯度平方的decaying average $E[g^2]\_t$，发现学习率公式中的递减（分母）项可以写为Root Mean Square(RMS) of the gradient
- 为了使得参数和学习率更新有相同的hypothetical units，又定义了一个exponentially decaying average取代人为设置的learning rate，直接对更新参数的平方进行操作，而不只是对梯度的平方进行操作
- 最终参数更新形式为 $θ_{t+1}=θ_t-\frac{RMS[Δθ]\_t}{RMS[g]\_t}$, $RMS[Δθ]\_t=\sqrt(E[Δθ^2]\_t+ϵ)$, $RMS[g]\_t=\sqrt(E[g^2]\_t+ϵ)$
- 更新规则中没有了学习速率，所以不用对其进行人为设置

**Adam, 2014**

- Adaptive Moment Estimation
- 除了像Adadelta和RMSprop一样保存去过梯度平方和的exponentially decaying average外，Adam还保存类似momentum一样过去梯度的exponentially decaying average，它看起来像是RMSProp的动量版
- 实现
	- mt=β1⋅mt−1+(1−β)⋅gt
	- vt=β2⋅vt−1+(1−β2)⋅gt^2
	- mt和vt分别是梯度的一阶矩（均值）和二阶距（偏方差）的估计
	- 由于mt和vt由全零的向量来初始化，Adam的作者观察到他们会被偏向0，特别是在initial time steps或decay rates很小的时候（即β1和β2都接近于1），于是他们通过计算bias-corrected一阶矩和二阶矩的估计低消掉偏差：
	- m'=m/(1−β1^t)
	- v'=v/(1-β2^t)
	- 然后用上述项和Adadelta和RMSprop一样进行参数更新，得到更新规则:
	- $θ_{t+1}=θ_t-m'\frac{η}{\sqrt{v'+ϵ}}$
- 推荐默认设置α=0.001,β1=0.9,β2=0.999,ϵ=10−8，实际操作中推荐将Adam作为默认的算法，一般而言跑起来是目前最优的

**二阶方法**

- 基于牛顿法的第二类常用的最优化方法
- $x=x-[Hf(x)]^{-1}∇f(x)$
- Hf(x)是Hessian矩阵，函数二阶偏导数的平方矩阵
- 优点
	- 海森矩阵描述了损失函数的局部曲率，从而使得可以进行更高效的参数更新
	- 具体来说，就是乘以Hessian转置矩阵可以让最优化过程在曲率小的时候大步前进，在曲率大的时候小步前进
	- 在这个公式中没有学习率，相较于一阶方法是一个巨大的优势
- 缺点
	- 很难运用到实际的深度学习应用中，因为计算（以及求逆）Hessian矩阵操作非常耗费时间和空间
	- 假设一个有一百万个参数的神经网络，Hessian矩阵大小就是[1,000,000 x 1,000,000]，占用将近3,725GB的内存。于是，各种各样的拟-牛顿法就被发明出来用于近似转置Hessian矩阵。在这些方法中最流行的是L-BFGS，L-BFGS使用随时间的梯度中的信息来隐式地近似（也就是说整个矩阵从来没有被计算）
	- 然而，即使解决了存储空间的问题，L-BFGS应用的一个巨大劣势是需要对整个训练集进行计算，而整个训练集一般包含几百万的样本。和MSGD不同，让L-BFGS在小批量上运行很需要技巧，同时也是研究热点
- 实践时在深度学习和卷积神经网络中，使用L-BFGS之类的二阶方法并不常见。相反，基于（Nesterov的）动量更新的各种随机梯度下降方法更加常用，因为它们更加简单且容易扩展

### 神经网络防止过拟合方法

**从数据本身**

- 所有的过拟合无非就是训练样本的缺乏和训练参数的增加
- 获得更多数据的方法
	- 从数据源头获取更多数据，但很多情况下，大幅增加数据本身就不容易；另外不清楚获取多少数据才算够
	- 根据当前数据集估计数据分布参数，使用该分布产生更多数据：一般不用，因为估计分布参数的过程也会代入抽样误差
	- 通过一定规则扩充数据，即*数据增强*(Data Augmentation)
- 数据增强
	- 如在物体分类问题里，物体在图像中的位置、姿态、尺度，整体图片明暗度等都不会影响分类结果，就可以通过图像平移、翻转、缩放、切割等手段将数据库成倍扩充
	- Color Jittering：对颜色的数据增强，如图像亮度，饱和度，对比度变化
	- PCA Jittering：首先按照RGB三色通道计算均值和标准差，再在整个训练集上计算协方差矩阵，进行特征分解，得到特征向量和特征值
	- Random Scale：尺度变换
	- Random Crop：采用随机图像差值的方式对图像进行裁剪或缩放，包括scale Jittering的方法(详见VGG, ResNet的使用, todo)或者尺度和长宽比增强变换
	- Horizontal/Vertical Flip：水平垂直变换，Shift平移变换，Rotation/Reflection旋转/仿射变换，Noise高斯噪声，Blur模糊处理
	- Label Shuffle：类别不平衡数据的增广(海康威视ILSVRC2016 report)

**模型限制**

- 限制权值weight decay
	- 常见的正则化如L1 L2，L1较L2能够获得更稀疏的参数，但L1零点不可导
	- 在损失函数中，weight decay(λ)是放在正则项前面的系数，用来调节模型复杂度对损失函数的影响，若λ很大，则复杂的模型损失函数的值也就大
- 训练时间early stopping
	- 提前停止其实是另一种正则化方法
	- 在训练集和验证集上，一次迭代之后计算各自的错误率，当在验证集上的错误率最小，在没开始增大之前停止训练
	- 一般做法是，记录到目前为止最好的validation accuracy，当连续n次Epoch（或者更多次）没达到最佳accuracy时，则可以认为accuracy不再提高
	- 这种策略也称为“No-improvement-in-n”，n即Epoch的次数，可以根据实际情况取，如10、20、30
	- 在初始化网络的时候一般都是初始为较小的权值，训练时间越长，部分网络权值可能越大，如果在合适时间停止训练，就可以将网络的能力限制在一定范围内
- 网络结构：减少网络的层数、神经元个数等均可以限制网络的拟合能力
- 增加噪声
	- 输入加噪声：如加入高斯噪声，训练时减小误差，同时也会对噪声产生的干扰项进行惩罚，达到减小权值的平方，与L2类似的效果
	- 权值加噪声：用0均值的高斯分布作为网络初始化，如Alex Graves 的手写识别RNN(2009)
	- 对网络的响应加噪声：在前向传播过程中让某些神经元的输出变为 binary 或 random，显然这种乱来的做法会打乱网络的训练过程，让训练更慢，但据 Hinton说，在test set上效果有显著提升

**模型结合**

- 简而言之训练多个模型，以每个模型的权重平均输出作为结果
- Bagging和boost（详见之前对两者的分析）
	- 简单理解就是分段函数的概念：用不同的模型拟合不同部分的训练集。以RF为例，就是训练了一堆互不关联的决策树。但由于训练神经网络本身就需要耗费较多自由，所以一般不单独使用神经网络做Bagging
- Dropout
	- 正则是通过在代价函数后面加上正则项来防止模型过拟合的，而在神经网络中，Dropout是通过修改神经网络本身结构来实现的
	- 由Hintion提出，源于Improving neural networks by preventing co-adaptation of feature detectors(通过阻止特征检测器的共同作用来提高神经网络的性能)
	- 以概率p舍弃部分神经元，其他神经元以概率1-p被保留，舍去的神经元输出都被设置为0
	- 在实践中效果很好，因为它在训练阶段阻止了神经元的*共适应*co-adaptation

### Batch Normalization, 2015

- 概念
	- 一个自适应的重新参数化的方法，试图解决训练非常深层模型的困难
	- 见Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
	- 机器学习领域有一个很重要的假设：iid独立同分布假设，就是假设训练数据和测试数据满足相同分布，这是通过训练数据训练出来的模型能够在测试集上获得好的效果的一个基本保证，BN就是在深度神经网络训练过程中使得每一层神经网络的输入保持相同分布的方法
	- 在神经网络训练时遇到收敛速度很慢，或梯度爆炸等无法训练的状况时可以尝试BN来解决

**Internal Covariate Shift**

- 定义：隐层(internal)网络结构中的输入分布总是变化，违背了iid的假设
- 在图像数据中，如果对输入数据先作减均值操作，数据点就不再只分布在第一象限，这是一个随机分界面落入数据分布的概率增加了2^n倍，大大加快学习
- 更进一步的，对数据再进行去相关操作，例如PCA和ZCA白化，数据不再是一个狭长的分布，随机分界面有效的概率就又大大增加，使得数据更加容易区分，这样又会加快训练
- 但计算协方差的特征值太耗时也太耗空间，一般最多只用到z-score处理，即每一维减去自身均值，再除以自身标准差，这样能使数据点在每维上具有相似的宽度，可以起到增大数据分布范围，进而使更多随机分界面有意义的作用

**直观解释**

- 深层神经网络在做非线性变换前的激活输入值随着网络深度加深或者在训练过程中，其分布逐渐发生偏移或者变动
	- 收敛慢一般是整体分布逐渐往非线性函数的取值区间的上下限两端靠近（对于Sigmoid函数来说，意味着激活输入值WU+B是大的负值或者正值）
	- 这导致梯度消失，这是训练深层神经网络收敛慢的本质原因
	- BN把逐渐向取值区间极限饱和区靠拢的输入分布，强制拉回到均值为0方差为1的比较标准的正态分布，使得非线性变换函数的输入值落入对输入比较敏感的区域
	- 因为梯度一直都能保持比较大的状态，所以对神经网络的参数调整效率比较高，也就是收敛地快
- 但如果都通过BN，就跟把非线性函数替换成线性函数了
	- 这就会使得多层线性网络跟一层线性网络等价
	- 比如在使用sigmoid激活函数的时候，如果把数据限制到零均值单位方差，那么相当于只使用了激活函数中近似线性的部分，这显然会降低模型的表达能力
- 为了保证非线性，BN对变换满足均值为0方差为1的x又进行了scale加上shift操作(y=scale\*x+shift)
	- 每个神经元增加scale和shift参数，这两个参数是通过训练学习到的
	- 通过scale和shift把这个值从标准正态分布左移或者由移一点并长胖一点或者变瘦一点，每个实例挪动的程度不一样，等价于非线性函数的值从正中心周围的线性区往非线性区动，让因训练所需而“刻意”加入的BN能够有可能还原最初的输入
	- 核心思想是想找到一个线性和非线性的较好平衡点，既能享受非线性的较强表达能力的好处，又避免太靠非线性区两头使得网络收敛速度太慢
- BN对sigmoid激活函数的提升非常明显，解决了sigmoid过饱和的问题
	- 但sigmoid在分类问题上确实没有ReLU好用，因为sigmoid的中间部分太“线性”，不像ReLU一个很大的转折，在拟合复杂非线性函数的时候没那么高效

**算法**

- 每个隐层又加上了一层BN操作层，它位于X=WY+B激活值获得之后，非线性函数变换之前
- 对mini-batch SGD，BN操作就是对于隐层内每个神经元的激活值，进行如下变换
	- x' = (x - E[x]) / sqrt(Var[x])
- 在理想情况下均值和方差是针对整个数据集的，但显然这不现实，因此用一个Batch的均值和方差作为对整个数据集均值和方差的估计
- gain：把值往后续要进行的非线性变换的线性区拉动，增大梯度，增强反向传播信息流行性，加快训练收敛速度
- lose：为了防止网络表达能力下降，每个神经元增加scale和shift，这俩个参数是通过训练来学习的，用来对x'反变换，这其实是变换的反操作
	- y' = p_{scale} x' + p_{shift}

**作用**

- 可以使用很高的学习率
	- 如果每层的scale不一致，其需要的学习率不一样，同一层不同维度的scale也需要不同大小的学习率，通常需要使用最小的那个学习率才能保证损失函数有效下降
- 移除或使用较低的dropout
	- dropout用来防止过拟合，而导致过拟合的位置往往在数据边界处，如果初始化权重就已经落在数据内部，过拟合就可以得到一定的缓解
	- 论文中最后的模型分别使用10%、5%和0%的dropout训练模型，与之前的40%~50%相比，大大提高了训练速度
- 降低L2权重衰减系数
	- 边界处的局部最优往往有几维的权重（斜率）较大，使用L2衰减可以缓解这一问题，现在用BN就可以把这个值降低了，论文中降低为原来的5倍
- 取消LRN层
	- 由于使用了一种Normalization，再使用LRN就没那么必要了，而且LRN实际上也不怎么work
- 减少图像扭曲的使用
	- 由于现在训练epoch数降低，所以要对输入数据少做一些扭曲，让神经网络多看看真实的数据

### CNN

**经典网络**

- LeNet-5
	- input (hand-written digits, 32 x 32 x 1) -> 6 of 5x5 conv (s=1, p=0) -> 28 x 28 x 6 -> avgPool(f=2, s=2) -> sigmoid -> 14 x 14 x 6 -> 16 of 5x5 conv (s=1, p=0) -> 10 x 10 x16 -> avgPool(f=2, s=2) -> sigmoid -> 5 x 5 x 16 -> FC(120) -> FC(84) -> sth before softmax -> 10 possible yHat
	- 0.06 Mil 参数
	- 还没有max pooling，padding，或softmax，ReLu
	- 激活函数是sigmoid/tanh
	- avgpooling-sigmoid时提到了一种graph transformer(现在几乎不用了)
- AlexNet
	- input (227 x 227 x 3) -> 11x11 conv (s=4) -> 55 x 55 x 96 -> 3x3 maxPool(s=2) -> 27 x 27 x 96 -> 5x5 conv -> 27 x 27 x 256 -> 3x3 maxPool(s=2) -> 13 x 13 x 256 -> 3x3 conv -> 13 x 13 x 384 -> 3x3 conv -> 13 x 13 x 384 -> 3x3 conv -> 13 x 13 x 256 -> 3x3 maxPool(s=2) -> 6 x 6 x 256 -> FC(4096) -> DC(4096) -> softmax(1000)
	- 60 Mil 参数
	- 包含更多的隐层和单元，使用了ReLu激活函数
	- 使用了复杂的方法把模型拆分到两个GPU上做计算
	- 使用了局部响应归一化层(Local Response Normalization, LRN层)，把不同channel进行了normalize，但被证明用处不大
- VGG-16
	- conv = 3x3, s=1, same
	- maxPool = 2x2, s=2
	- input (224 x 224 x 3) -> do 64 of conv 2 times -> 224 x 224 x 64 -> maxPool -> 112 x 112 x 64 -> do 128 of conv 2 times -> 112 x 112 x 128 -> maxPool -> 56 x 56 x 128 -> do 256 of conv 3 times -> 56 x 56 x 256 -> maxPool -> 28 x 28 x 256 -> do 512 of conv 3 times -> 28 x 28 x 512 -> maxPool -> 14 x 14 x 512 -> do 512 of conv 3 times -> 14 x 14 x 512 -> maxPool -> 7 x 7 x 512 -> FC(4096) -> FC(4096) -> softmax(1000)
	- 138 Mil 参数
	- 结构很重复，多但不复杂
	- VGG-19：更多参数，但performance没有提升很多
	- 比较有规律的地方：图像缩小的比例和通道增加的比例有关联

**Inception(GoogLeNet)**

- 概念
	- 把各种不同的卷积和池化操作全部试一遍并concatenate在一起，让网络自己决定过滤器的组合
	- input (28 x 28 x 192)
	- 0-padding使前两个维度对上28 x 28, 第三个维度人为设置
	- input -> 1x1 conv -> 0-padding -> 28 x 28 x 64
	- input -> (1x1 conv) -> 3x3 conv -> 0-padding -> 28 x 28 x 128
	- input -> (1x1 conv) -> 5x5 conv -> 0-padding -> 28 x 28 x 32
	- input -> 3x3 maxPool -> 0-padding -> (1x1 conv) -> 28 x 28 x 32
	- input -> ... -> concat -> 28 x 28 x 256
	- 这整个流程叫一个Inception Block
	- 8-10个Inception Block组成了Inception Network
- 问题：计算时间成本高昂，解决：*1x1 conv* (bottleneck layer)
	- 原本：28 x 28 x 192 -> 5x5 conv -> 28 x 28 x 32
	- 计算次数为 5x5x192 x 28x28x32 = 120 Mil
	- 改进：28 x 28 x 192 -> 1x1 conv -> 28 x 28 x 16 -> 5x5 conv -> 28 x 28 x 32
	- 计算次数为 192 x 28x28x16 + 5x5x16 x 28x28x32 = 12.4 Mil
- 延申
	- 论文中在某些Inception Block的input后面还加入了一些side branches
	- 把隐藏层的一部分拿出来，经过几个FC后作softmax，并参与最终预测
	- 这样做从某种意义上实现了regularizing，防止了过拟合(参照ResNet的做法)

**ResNet**

- 概念
	- “Plain Block”：一个神经元经过两层网络的一般形式（main path）
		- input a_l -> MLP -> z_{l+1} -> ReLU(z_{l+1}) -> a_{l+1} -> MLP -> z_{l+2} -> ReLU(z_{l+2}) -> a_{l+2}
	- Residual Block：在训练中途增加输入fast forward（short cut/skip connection）
		- input a_l -> MLP -> z_{l+1} -> ReLU(z_{l+1}) -> a_{l+1} -> MLP -> z_{l+2} -> ReLU(z_{l+2} *+ a_l*) -> a_{l+2} 
		- 这整个流程叫一个res block，多个res block相连就形成了ResNet
	- 一般网络的训练误差在网络过于deep时会go back up，但ResNet却没有这个问题（即使网络达到100+的深度）
- 为何有用
	- ResNet的重点在于g(z_{l+2} + a_l)，g是激活函数
	- 如果我们使用ReLu，经过前面network的激活，input a_l非负
	- 如果z_{l+2}=0, 那么g(z_{l+2} + a_l) = g(a_l) = (因为a_l非负) a_l
	- 由此可见Res Block很容易学习identity function
	- 这意味着对比我们不加ResNet（也就是identity）的时候，加了ResNet起码不会使得网络变得更差，也就是说更深的网络不会hurt performance
	- 那么当z_{l+2}!=0时，网络的性能就有可能提高
- 延申
	- 如果a_l和a_{l+2}的维度不一样，假设为m和n，我们需要增加
		- z_{l+2} + W_s a_l
		- W_s 的维度是 nxm，可以是一个正常参与训练的矩阵
		- 或者直接0 padding a_l的剩余维度

**应用**

- 边缘检测(Edge Detection)
	- vertical edge 3x3卷积核:[[1 1 1],[0 0 0],[-1 -1 -1]]
	- 会将图像像素差异很大的部分提取出来，就实现了边缘检测

### RNN

- 应用引入
	- slot filling(ticket booking system)
	- sequence data, spreading & sharing
	- 需要“记忆力”，对上下文的理解，“记忆力”即指RNN
- 概念
	- 输入矩阵W_{ax}, 包含所有y_k
	- 隐层矩阵W_{aa}, 包含所有sharing info vector a_k
	- 输出矩阵W_{ya}, 包含所有x_k
- 向前传播
	- W_a = [W_{aa} | W_{ax}]
	- a_t = g_1(W_{aa} a_{t-1} + W_{ax} x_t + b_a) = g_1(W_a [a_{t-1}, x_t]^T + b_a), g = tanh/ReLu
	- y_t = g_2(W_{ya} a_t + b_y)
- 损失函数（交叉熵）
	- loss L(y, y') = sum(t : L_t(y_t, y'\_t))
	- L_t(y_t, y'\_t) = - y_t log(y'\_t) - (1-y_t) log(1-y'\_t)
- 不同结构的RNN
	- one-to-one
	- many-to-many: 一般形式，输入和输出同时进行
		- 长度相同
		- 长度不同：机器翻译，全部输入之后(encoder)输出所有结果(decoder)
	- many-to-one: 全部输入之后输出单一结果
	- one-to-many: 生成模型(sequence generation)，单一输入后输出所有结果
- 语言模型(language modelling)
	- 可用于生成模型
	- tokenize, BOS, EOS (end of sentence), UNW (unknown word), 把这些加入dictionary并one-hot represent
	- 把从0->t的输入放入RNN并预测下一个word
	- 可以在训练好之后把第t个生成的输入放入RNN并sample下一个word
	- 其他
		- character-level语言模型，每个letter是一个输入
		- 计算更昂贵，使用在专有问题上
- 发展
	- Elman Network：上层input中hidden layer的值会被保存并参与下层input hidden layer的计算
	- Jordan Network：上层output的值保存到下层input hidden layer的计算

**GRU**

- 解决RNN中长句段的长期依赖引起梯度消失的问题，无法预测或记忆
- 一个简化的LSTM，只有两个gate
- simplified版本
	- memory cell：记忆模块 c_t = a_t
	- c'\_t = tanh(W_c [a_{t-1}, x_t]^T + b_c)
	- gate_u = lambda(W_u [a_{t-1}, x_t]^T + b_u)
	- c_t = gate_u * c'\_t + (1 - gate_u) * c_{t-1}
	- forget gate_u用来决定是否把之前的值放入时间t的输入
- visualization
	- 上次输出c_{t-1}=a_{t-1}与这次输入x_t权重相加 = 这次总输入矩阵
	- 经过tanh得到c'\_t
	- 通过gate_u决定c'\_t舍弃和保留的部分，输出c_t=a_t
	- 或者把a_t作softmax同时输出y_t
- standard版本
	- gate_r = lambda(W_r [a_{t-1}, x_t]^T + b_r)
	- c'\_t = tanh(W_c [gate_r * a_{t-1}, x_t]^T + b_c)
	- input(relevance) gate_r 表示c'\_t和c_{t-1}的相关性
	- 实验表明这两个gate是最有效的

**LSTM, 1997**

- simplified版本(simple RNN)
	- memory storage：记忆模块，也就是LSTM的一个neuron
	- input/update gate：选择是否把输入放入记忆, gate_u
	- forget gate：选择是否把记忆忘记或作format, gate_f
	- output gate：选择是否把记忆输出, gate_o
	- c'\_t, gate_u 同simplified GRU
	- gate_f = lambda(W_f [a_{t-1}, x_t]^T + b_f)
	- gate_o = lambda(W_o [a_{t-1}, x_t]^T + b_o)
	- c_t = gate_u * c'\_t + gate_f * c_{t-1}
	- a_t = gate_o * tanh(c_t)
	- 加上初始输入，每个神经元一共4个input，所以(完全)LSTM的网络会有4倍的参数量
	- 把neuron所有维度的4个input合成vector：z(input vector), z^i, z^f, z^o
	- 于是一次在位置（时间）t的训练就是y^t=LSTM^t(z, z^i, z^f, z^o)
- visualization
	- 与GRU相同的
		- 上次输出a_{t-1}与这次输入x_t权重相加 = 这次总输入矩阵
		- 经过tanh得到c'\_t
	- 通过gate_u决定c'\_t保留的部分，输出c_t的一部分(u_t)
	- 通过gate_f决定上次c_{t-1}保留的部分，输出c_t的另一部分(f_t)
	- 两部分相加得到c_t
	- c_t通过gate_o决定其保留的部分，输出a_t
	- 或者把a_t作softmax同时输出y_t
- standard版本(multiple-layer LSTM)
	- peephole:总输入矩阵还会增加c_{t-1}

**Bidirectional RNN (BRNN)**

- 解决了之前网络中只利用forward info的问题，双向记忆
- 每个RNN block加入另一个block，但反向连接，所有x_{t+1}=f(x_t)的部分变为X_{t-1}=f'(x_t), where f/f'代表整个变换
- y_t = g(W_y[a_t(forward), a_t(backward)]^T + b_y)
- 在NLP有完整句子输入时BRNN最为常用
- 缺点：需要*整个*句子的info，在语言识别中就需要读完整个句子

**Deep RNNs**

- 每个input堆叠多层RNN block(a_{x,y})
- column: x_k -> a_{1,k} -> a_{2,k} -> ... -> a_{k,n} -> y_k
- row: a_{k,0} -> a_{k,1} -> ... -> a_{k,n}
- 因为其庞大的计算量，n=3时网络已经可以称之为deep

### GNN

- 任务
	- 分类：分子突变检测(Graph 2009)
	- 生成：开发新型药物的化学分子(GraphVAE 2018)，todo (MolGAN)
	- semi-supervised learning半监督，部分无标注信息的数据
	- 表示学习：Graph InfoMax
- 概念
	- 每一个输入(entity)都有一些(attribute)，entity之间有复杂的结构和关联(structure, relation)
	- 1.卷积神经网络的推广：spatial-based convolution, 基于不同的聚类(arggregation)方法
		- sum: NN4G
		- mean: DCNN, DGC, GraphSAGE
		- weighted sum: MoNET, GAT*主要*, GIN
		- LSTM & max pooling: GraphSAGE
	- 2.信号处理的卷积：把图转为傅里叶域(Fourier domain) spectral-based convolution
		- ChebNet -> GCN*主要* -> HyperGCN
- benchmark数据集
	- SuperPixel MNIST：图像分类
	- ZINC molecule：预测分子溶解度
	- SBM：graph pattern recognition和semi-supervised graph clustering
	- TSP：边缘分类, traveling salesman problem
	- CORA：citation network
	- TU-MUTAG：todo
- library：Deep Graph Library

**Spatial-based GNN**

- 概念
	- aggreagate：用neighbor feature update下一层的hidden state
	- readout：把所有nodes的feature集合起来代表整个graph
- 发展
	- NN4G(NN for Graphs, 2009)
		- 把每个node通过一个embedding matrix得到feature
		- 然后通过neigbor feature的aggregation得到每一层的hidden layer
		- 每层的feature相加(sum的由来)得到整个graph(readout)
	- DCNN(Diffusion CNN, 2015)
		- 把每个node与和它相距为k的node的距离的平均值作为权重学习feature的vector(h_{node}^k)
		- 所有的node的这个feature是一个相距为k的所有feature的一个矩阵H^k
		- 所有k的可能排列成了一个多重矩阵网络，某一个node的训练就是通过h_{node}^{1...k}乘以一个训练矩阵W的输出
	- DGC(Diffusion Graph Convolution)
		- 把DCNN的每个H^k直接加起来训练而不是组成全连接网络
	- MoNET(Mixture Model Network, 2016)
		- 把feature计算的方式增加权重而不是直接加和或者取平均
		- 定义了node之间*距离*的计算方式
	- GraphSAGE(SAmple and aggreGatE, 2017)
		- 使用了将node以不同顺序输入LSTM，邻居加和/求平均，和将feature作max-pooling的不同方法学习feature信息
		- 与以前的模型比较，使用了mini-batch neighbor sampling
		- 使用的是inductive learning，可以work在both indutive和transductive setting上
	- GAT(Graph Attention Network, 2017)
		- 对某一层各个node的feature计算其之间的energy e, 表示一个node对另一个的重要程度
	- GIN(Graph Isomorphism Network, 2019)
		- 提供了GNN某些feature learning的方法work的原因和证明
		- 结论：update的方式应该遵循把邻居feature*相加*(而不是取平均或用pooling)后加上自己feature(可以额外增加一个极小值epilson的权重)后放入网络中训练

**Spectral-based GNN**

- 概念
	- 图本身的convolution因为不在欧氏空间无法进行，但可以通过fourier transform把图转换到傅里叶域，在fourier domain里的convolution相当于multiplication，完成后做inverse傅里叶变换回原domain
- 信号与系统 (to do more)
	- 信号可以被看作N维空间的一个向量，是由一组basis经过线性组合*合成*的
		- 信号A=sum(a_k v_k)
	- 要知道信号的每一个component a_k的大小，就需要用*分析*
		- a_j=A v_j=sum(a_k v_k) v_j 因为v_i v_j是正交的
	- (时域time domain basis)对于一个周期性的信号，可以把它展开成一个由正余弦构成的傅里叶级数Fourier series
		- x(t)=sum(a_k H_k(t))
		- H_k(t): j-th harmonic components
		- 这样就有方法算出a_k的大小(时域卷积公式)
	- (频域frequency domain basis)
		- x(t)=1/(2pi)'{X(jw)e^{jwt}}dw
		- 其中e^{jwt}就代表了basis, X(jw)就是系数a_j
		- 要找到X(jw)的值，也就是a_j，要用fourier transform
		- X(jw)='{x(t)e^{-jwt}}dt 是inner product(分析),frequency domain的傅里叶变换
	- 将信号(函数)在时域的正交基orthogonal basis和频域的正交基相互变换
- 推导：Spectral Graph Theory
	- 定义无向图G=(V, E), 邻接矩阵A, 度数矩阵(对角矩阵，diagonal项是当前node的邻居数)D, 信号函数(表示一个node的信号, 信号可以定义成任何统一的表示量)f：V -> R^N
	- 图拉普拉斯变换 Graph Laplacian (L = D - A)
		- GL矩阵L被定义为度数矩阵减去邻接矩阵
		- (无向图)L对称symmetric, positive semidefinite
		- L可以被谱分解(spectral decomposition)为U V U^T
		- V=diag(v_0, ..., v_{N-1})特征值(eigenvalue)
		- U=[u_0, ..., u_{N-1}] orthonormal的特征向量(eigenvector)
		- v_l是频率(scalar)，u_l是对应v_l的basis(vector)
		- L = U V U^T = [u_0, ..., u_{N-1}] diag(v_0, ..., v_{N-1}) [u_0, ..., u_{N-1}]^T
	- Vertex domain signal
		- f(node)是一个节点的信号量，所有node就可以写成一个f向量
		- 同时可以写出其邻接矩阵A和度数矩阵D, 算出L, 继而算出U和V
		- 于是就得到了每个频率v_l(代表每个node)的basis u_l
	- Interpreting vertex frequency
		- 我们把拉普拉斯矩阵和信号f相乘会如何?
		- Lf = (D-A)f = Df - Af 是一个向量
		- 每一个单位L_k f = D_k f - A_k f
		- 也就是一个节点k，它的度数向量乘以所有节点的信号量，然后减去它的邻接向量乘以所有节点的信号量
		- 度数向量只有它自己的那一column不为0(因为是diag), 所以D_k f就是它有几个neighbor乘以他自己的信号量
		- 邻接向量就是他所有neighbor的信号量的和
		- 所以，L_k f代表节点k和它旁边节点的能量差异
	- 结论
		- 为了表示能量差异的公平性，把它平方：f^T L f
		- 结果简化后是$1/2\sum_{v_i\in V}\sum_{v_j \in V}w_{i,j}(f(v_i)-f(v_j))^2$，其中w_{i,j}代表A_{i,j}，也就是用邻接矩阵来表示(初始)学习权重
		- 简单地说，就是节点之间的信号能量差("power" of signal variation between nodes), 或叫图信号的smoothness
		- Discrete time Fourier basis 告诉我们信号频率和信号变化量的关系，频率越大，两点间信号的变化量越大，于是Spectral Graph Theory就可以用来量化信号的频率差异大小
		- DZ component：同理，谱分解的结果也代表在频率v_l变高后，u_l basis向量(相当于能量)变化的差异变剧烈，在傅里叶变换里的sin/cos函数波动就更大
- 实现
	- 对于输入x，把x理解为上文的信号，我们要学习信号component V的大小
	- Graph Fourier Transform of signal x(spectral domain, 分析)
		- x'=U^Tx(vertext domain, 合成)
		- U^T_k x 代表了V_k的大小
	- Inverse Graph Fourier Transform of signal x
		- simply：x = Ux'，从傅里叶反向变换回原域
	- Filtering
		- 就是用来学习V的转换矩阵，表示为g(V)，我们要在spectral domain得到 y' = g(V)x'
		- 所以转换回来的 y = Uy' = Ug(V)U^Tx = g(UVU^T)x = g(L)x
	- 问题：L中的V的学习过程是O(N)的复杂度，且g(L)非localize(见下)
	- g(L)的选择
		- 如果g(L)=L, y=Lx, 图只学习到距离为1的临近矩阵的信号信息
		- 如果g(L)=l^2, 图会学习到最多距离为2的neighbor的信号信息
		- 在CNN中，我们类似的学习附近pixel的信息
		- 如果g(L)=cos(L)=I-L^2/2!+L^4/4!...，图会学习到无限延展递减的信号，但这样就不是*附近*(localize)的信息了
- 发展
	- ChebNet(2016)
		- 解决了non-localized的问题
		- 使用多项式参数化 g(L or V)：g(L or V)=sum_{k=i->K}(h_k L or V^k), 学习h_k，于是变成K-Localized
		- y=Ug(V)U^Tx=U(sum_{k=i->K}(h_k L^k))U^Tx
		- 但时间复杂度更高了，O(N^2)
		- 解决：切比雪夫多项式Chebyshev polynomial，一个在L上循环的多项式
		- T_0(x)=1, T_1(x)=x, T_2(x)=2xT_{k-1}(x)-T_{k-2}(x), x\in[-1,1]
		- 此时V相当于要learn的x，假设在某次epoch后有V->V',那么在Cheb ploy中V'=2V/max(V)-I
		- 把g(V)=sum(h_k V^k)转换成一个g(V')=sum(h_k' T_k(V'))，这个转换矩阵需要比原先的g(V)好算的多
		- 注意g(L')x=h_0'T_0(L')x+h_1'T_1(L')x+...+h_K'T_K(L')x
		- 根据Cheb ploy的上述性质，此式可写为h_0'x_0'+...+h_K'x_K'=[x_0' x_1' ... x_K'][h_0' h_1' ... h_K']，其中x_k'=T_k(L')x
		- 而x_K是可以通过L递归计算的，时间复杂度最终变为O(KE)
	- GCN(2018)
		- 用ChebNet并且把K设为1，使用normalized Laplacian
		- K=1相当于只关注邻近点
		- normalized Laplacian的性质是L=I-D^{-1/2}AD^{-1/2}, 而且max(V)大约为2
		- 于是L'=2L/max(V)-I=L-I
		- 为了更加减少参数提升性能，定义h=h_0'=-h_1'
		- 得到y=g(L')x=h_0'x+h_1'L'x=h_0'x+h_1'(L-I)x=h(I+D^{-1/2}AD^{-1/2})x
		- 使用一种renormalization trick，y=h(D'^{-1/2}A'D'^{-1/2})x
		- 最后学习的唯一变量 h_v = f(1/|V|sum(u, Wx_u+b))
		- 简要意图就是把输入x通过一种变换后把它自己和所有邻近矩阵加和取平均，然后经过激活函数，学习feature h并迭代
- 延申 (todo)
	- GCN面临严重的深度收敛问题over-smoothing, 有DropEdge的方法改善
	- GCN更深层效果反而不太好，亟待解决
	- HyperGCN
	- Graph Generation
		- VAE based model
		- GAN based model
		- AR based model
	- GNN for NLP
		- Semantic Roles Labeling
		- Event Dectection
		- Document Time Stamping
		- Name Entity Recognition
		- Relation Extraction
		- Knowledge Graph

### Attention

- seq2seq: 模型决定生成几个label
- sequence labeling: 每个seq对应一个label
	- 上下文信息

**self-attention**

- 广义的Transformer，*结合*上层所有input的信息输出下层对应的output
	- 上层input(query, key)的相关性attention score: dot-product, additive
	- 做一个query和所有key的score之后归一如softmax, 得到所有alpha
	- 每一个key乘以一个W矩阵得到v(相当于加权操作), 乘以所有alpha后加和得到b, 为那个query的最终output
	- 总结
		- input矩阵I的每一个input a分为(q, k, v), Q=W^qI, K=W^kI, V=W^vI
		- alpha矩阵A=K^TQ, normalize后得到A', A'被叫做Attention Matrix
		- output矩阵O=VA'
- multi-head self-attention
	- 不同形式的相关性，多个W^q, W^k, W^v
- 没有位置信息?
	- 解决：positional encoding，输入a加一个e(positional vector)代表位置, hand crafted -> trainable(sinusoidal, position embedding, FLOATER, RNN)
- 应用
	- 语音辨识：Truncated self-attention, 使用部分句子
	- 图像处理：self-attention GAN, DEtection Transformer(DETR)
	- 图论：Attention Matrix就是邻接矩阵，直接设置不用训练了->是一种GNN
- 对比CNN
	- CNN只考虑感知域的信息，且人为设置，是一种简化版的self-attention
	- self-attention的考虑范围大小是自动学习的，类似于复杂化的CNN
	- 但更flexible的模型需要更大量的数据，不然容易overfit， CNN在小样本量时效果就比较好
- 对比RNN
	- RNN不能parallel并行，距离远的输入影响小
	- 对self-attention来说没有距离的概念(无positional encoding时)
