# CV

- 计算机视觉现状
	- Little Data (more hand-engineering, "hacks" <- transfer learning helps) - [ object detection - image recognition - speech recognition ] - Lots of Data (less hand-engineering)
	- two sources of knowledge: Labeled data, hand engineered features/network architecture/other components
	- tips for doing well on benchmarks/competitions
		- ensembling
		- multi-crop at test time (10-crop)

### 目标检测(Object Detection)

**目标定位(Object Localization)**

- 图像分类 -> 图像定位 -> 图像检测
- bounding box (4 #s): midpoint(bx, by), height(bh), width(bw)
- class label (1 to n+1), n+1 = background (no object)
- the output: y=[p_c, bx, by, bh, bw, c_1, c_2, c_3]^T
	- p_c：有任何object的可能性
	- c_k：class k
	- 如果p_c = 0, 其他参数dont care
	- b_i：必须在0和1之间，需要sigmoid和ReLu激活
	- 损失函数：可以都是square loss，或者对p_c作logistic reg loss，对c_k作log likelihood loss

**特征点检测(Landmark Detection)**

- 实现计算机图形特效的key之一
- 某一个特征点的位置(l_x, l_y)
- people-pose detection 人物姿态检测，连续的特征点相连形成posture

**two-stage算法**

- 概述
	- 将物体识别和物体定位分为两个步骤，分别完成
	- 识别错误率和漏识别率低，但速度较慢，不能实时检测
- R-CNN (2014)
	- 用候选区域方法(region proposal method)创建目标检测的感兴趣区域(region of interest, ROI)
		- 在选择性搜索（selective search，SS）中，首先将每个像素作为一组
		- 然后，计算每一组的纹理，并将两个最接近的组结合起来
		- 为了避免单个区域吞噬其他区域，首先对较小的组进行分组
		- 继续合并区域，直到所有区域都结合在一起
		- R-CNN 利用候选区域方法创建了约 2000 个 ROI
		- 这些区域被转换为固定大小的图像，并分别馈送到卷积神经网络中（将原始图像根据ROI切割、reshape再送进NN学习）
		- 该网络架构后面会跟几个全连接层，以实现目标分类并提炼边界框
	- 候选区域方法有非常高的计算复杂度，为了加速这个过程，通常会使用计算量较少的候选区域选择方法构建 ROI，并在后面使用全连接层进一步提炼边界框
	- 问题
		- 候选框由传统的selective search算法完成，速度慢
		- 对2000个候选框均需要做物体识别，也就是需要做2000次卷积网络计算，R-CNN很多卷积运算是重复的
- Fast R-CNN (2015)
	- 思路
		- R-CNN的很多候选区域是彼此重叠的，因此训练和推断速度非常慢
		- CNN中的特征图以一种密集的方式表征空间特征，能否直接使用特征图代替原图来检测目标？
	- 优化
		- 提出ROI pooling的池化结构，解决了候选框子图必须将图像裁剪缩放到相同尺寸大小的问题
			- 由于CNN网络的输入图像尺寸必须是固定的某一个大小（否则全连接时没法计算），R-CNN中对大小形状不同的候选框，进行了裁剪和缩放，使得他们达到相同的尺寸
			- 这个操作既浪费时间，又容易导致图像信息丢失和形变
			- fast R-CNN在全连接层之前插入了ROI pooling层，不需要对图像进行裁剪
			- 如果最终我们要生成MxN的图片，先将特征图水平和竖直分为M和N份，每一份取最大值，输出MxN的特征图，就实现了固定尺寸的图片输出
	- 算法
		- 使用CNN先提取特征图，而不是对每个子图进行卷积层特征提取
		- 抛弃了selective search，使用RPN(候选区域网络，region proposal network)层生成候选框，并利用softmax判断候选框是前景还是背景，从中选取前景候选框（因为物体一般在前景中），并利用bounding box regression调整候选框的位置，从而得到特征子图，称为proposals
		- 使用 ROI 池化将特征图块转换为固定大小，并送到全连接层
		- 利用ROI层输出的特征图proposal，判断proposal的类别，同时再次对bounding box进行regression从而得到精确的形状和位置
	- 优势
		- 包含特征提取器、分类器和边界框回归器在内的整个网络能通过多任务损失函数进行端到端的训练，这种多任务损失结合了分类损失和定位损失的方法，大大提升了模型准确度

**one-stage算法**

- Sliding Window Detection
	- 滑动窗口尝试每一块儿region看是否有target
	- 然后增大窗口的大小（stride步幅），再重新尝试，迭代此操作
	- 问题：计算成本极高，步幅太大还会hurt performance
	- 在CNN里的解决
		- 首先把FC层换成卷积层，比如：
			- 5 x 5 x 16 -> FC(400) -> FC(400) -> softmax(4)
			- 5 x 5 x 16 -> 400 of 5x5 conv -> 1 x 1 x 400 -> 400 of 1x1 conv -> 1 x 1 x 400 -> 4 of 1x1 conv -> 1 x 1 x 4
		- (OverFeat)把滑动窗口的周围也放入输入，可以证明最终层的每个部分就是每个滑动窗口的结果，比如：
			- 14 x 14 x 3 -> 5x5 conv -> 10 x 10 x 16 -> 2x2 maxPool -> 5 x 5 x 16 -> 5x5 FC -> 1 x 1 x 400 -> 1x1 FC -> 1 x 1 x 400 -> 1x1 FC -> 1 x 1 x 4
			- 28 x 28 x 3 -> 5x5 conv -> 24 x 24 x 16 -> 2x2 maxPool -> 12 x 12 x 16 -> 5x5 FC -> 8 x 8 x 400 -> 1x1 FC -> 8 x 8 x 400 -> 1x1 FC -> 8 x 8 x 4
			- 每个1x1x4在一起就代表了64种滑动窗口的输出
		- 所以可以直接把整个图片共享计算
	- 额外问题：bounding box的位置不够准确，且可能不是正方形，解决：YOLO
- YOLO (you only look once, 2015)
	- y如果有8维（1维p_c, 4维b, 3维物体种类）
	- 滑动窗口的个数如果为3x3, 就是长宽一共分成3x3个窗口
	- output就是 3 x 3 x 8
	- 整个方法是单次卷积实现，运行特别快，因为不用重复滑动
	- 而且位置和大小可以被清晰地标注出来
	- 注意：y的p_c是根据中心点是否有物体决定的
- SSD (Single Shot MultiBox Detector)
	- 多尺寸feature map
		- 每一个卷积层，都会输出不同大小的感受野
		- 可以克服yolo对于宽高比不常见的物体识别准确率较低的问题
	- 多个anchors
	- 采用了数据增强
		- 生成与目标物体真实box间IOU为0.1 0.3 0.5 0.7 0.9的patch，随机选取这些patch参与训练，并对他们进行随机水平翻转等操作
	- 基础卷积网络采用mobileNet，适合在终端上部署和运行
- YOLO V2
	- 网络采用DarkNet-19
	- 去掉全连接层
	- 卷积层后加入BN
	- 多个anchors
	- pass through layer
	- 高分辨率输入training
	- multi-scale training
- YOLO V3
	- 网络采用DarkNet-53
	- 添加了特征金字塔(FPN), 取代了特征提取器(如R-CNN的CNN), 可以更好地检测小目标
- mask R-CNN, A-Fast-RCNN, Light-Head R-CNN, R-SSD, RON...(todo)

**评价指标**

- 交并比(Intersection over Union, IoU)
	- IoU = size of [预测的] / size of [实际的]
	- 一般 IoU >= 0.5 被认为correct (human convention, or 0.6/0.7)
	- 衡量两个边界的重合比例

**延申**

- 非最大值抑制(Non-max suppression, NMS)
	- 确保每个对象只被检测到一次而不是很多次
	- 窗口很小的时候好几个窗口可能都会觉得物体的中点在自己里面
	- 查看p_c c_k最高的bounding box并作高亮标记，如果别的预测box和这个的IoU很高(>= 0.5)，他们就会被抑制变暗
	- 重复这个步骤直到所有box都被高亮或变暗，然后只保留高亮的结果
- Anchor Boxes
	- 用来检测多个不同的物体
	- 先定义K个Anchor Box的形状(中心点，长宽)，代表K种想预测的物体类别
	- new y=[y1, y2, ..., yK]，或者说如果y有8维，滑动窗口19 x 19，5个Anchor Box，new y的维度是19 x 19 x 5 x 8
	- 预测时，查看bounding box的IoU和哪一种Anchor Box更高
	- 优势：可以更有针对性的对不同形状的物体进行分辨
	- 劣势：如果出现新的物体类别，或者两种物体Anchor Box的Shape相似，这个方法会陷入僵局
	- 如何选择Anchor Box形状：可以human标注，或使用k-means聚类物体类别
	- 
