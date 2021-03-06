---
layout: default
---

# **预学习(pre-training)的作用**

#### &nbsp;&nbsp;&nbsp;&nbsp;本篇博文主要讨论的是：为什么预学习能够比随机初始化参数的网络能有更好的效果？预学习在整个网络结构中起到了什么作用个？

#### &nbsp;&nbsp;&nbsp;&nbsp;首先我们再回顾一下深度学习存在的问题：

- <h4>参数太多，当参数数量多于样本数量时会造成过拟合；</h4>
- <h4>优化模型是非凸优化，存在无数个最小值；</h4>
- <h4>梯度离散问题，当反向调整参数时，随着层数的增加参数越难调动。</h4>

#### &nbsp;&nbsp;&nbsp;&nbsp;在深度网络中搜索参数是非常困难的，因为优化模型的非凸性，会涉及到很多局部最小值。Erhan(2009b)通过实验证明，对于上千个随机初始化的参数，梯度下降法每次迭代得到的最小值都不同，在结构多于2或3层的时候，得到的效果很差。这也说明了为什么在很长一段时间，深度结构没有引起注意。

#### &nbsp;&nbsp;&nbsp;&nbsp;2006年Hinton提出了training Deep Belief Networks，利用无监督方法逐层预学习(<a href="../Layer-wise/index.html">逐层预学习简介</a>)，然后用监督方法精调(fine-tuning)。将无监督预学习作为参数调整的一个阶段，后面利用梯度方法进行精调，这是第一次提到预学习方法。

#### &nbsp;&nbsp;&nbsp;&nbsp;Bengio在文献[1]中利用大量的实验探究了预学习在深度结构中的优势。

#### &nbsp;&nbsp;&nbsp;&nbsp;1. 文献中利用去噪自编码构建了深度结构。首先来看一下，随机初始化的网络和预学习之后的网络在测试误差上的表现效果。
<div style="text-align: center">
<table>
<tr>
<td>
<img src="../images/pre-1.jpg">
</td>
<td>
<img src="../images/pre-2.jpg">
</td>
</tr>
</table>
</div>


#### &nbsp;&nbsp;&nbsp;&nbsp;从这幅图上可以看到，从网络的1-4层，没有预学习的网络测试误差随着层数的增大误差在逐渐增大，而带有预学习的测试误差是逐步下降的。没有预学习的网络没有画出第五层的误差，是因为没有办法有效的训练第五层。从这个实验结果可以看到，预学习在深度网络中的优势是很明显的。下图是一个测试误差的直方图：
<div style="text-align: center">
<img src="../images/pre-3.jpg">
</div>

#### &nbsp;&nbsp;&nbsp;&nbsp;这幅图表示测试误差的分布直方图(手写体图像库)， 可以看到随着网络层数的增大，带有预学习的网络鲁棒性更强，而参数随机初始化的网络，随着网络层数的增大，误差的均值和方差也越来越大，这是因为随机初始化参数的深度网随着网络层数的增大，达到局部最小解的概率也增大。而对于图1中的预学习的网络第五层误差增大，说明了，虽然预学习能够在一定程度上优化网络的训练，但并不能完全使网络训练达到最好(这里我的理解是，预学习是使的网络的训练靠近最好，但是影响网络达到最好的因素不是预学习)。

#### &nbsp;&nbsp;&nbsp;&nbsp;2. 从上面的实验结果中我们可以看到预学习的优势，那么这种优势体现在什么地方？从网络的结构中我们可以知道，深度网络是由多层非线性结构组成的非凸模型，这将会涉及到许多个局部最小解。而在模型参数求解过程中利用的是梯度下降法，梯度下降法的特点是，无论从哪个初始点开始，遇到局部最小解就会停止下降。从这一观点出发，预学习是将参数规划到梯度下降法能够下降到更深的参数空间，这是从优化角度讲预学习能够帮助优化参数。而从另一种角度来讲，预学习也是模型的一种正则化，虽然这种正则化并没有明显的正则项体现在目标函数中。

#### &nbsp;&nbsp;&nbsp;&nbsp;这里我们提一些正则化。在训练模型的时候，通常是最小化训练样本的误差，当训练误差足够小的时候，可能我们训练出来的模型很大程度上依赖于训练样本集，或者可能仅仅使用与样本集，这会造成模型的泛化能力很差。那为了避免这种情况，需要对模型进行正则化，简单来说就是为了在训练误差和泛化能力之间找一个平衡，最好能达到模型既有我们认可的泛化能力又有足够小的训练样本误差。另一方面，也可以说正则化是对训练样本的一个先验约束。

#### &nbsp;&nbsp;&nbsp;&nbsp;返回来看预学习的正则作用，文献中也做了相应的实验，即在相同测试误差的条件下，观察使用预学习和没有使用预学习的网络在测试误差上的情况，如下图：
![](../images/pre-4.jpg)
![](../images/pre-5.jpg)

#### &nbsp;&nbsp;&nbsp;&nbsp;上图画的是test cost(Negative Log Likelihood)和train cost(Negative Log Likelihood)的网络分别为1,2,3层的网络。从图上看到，在训练误差相同的情况下，预学习的测试误差要小于随机初始化的测试误差。而1层网络图上可以看到，预学习的训练误差大于随机初始化网络的训练误差，而测试误差相差不大，而在2层和3层的网络中，预学习的训练误差都小于随机初始化的训练误差，而测试误差也小很多。这又可以出另一个问题，在浅层网络中，预学习效果并不理想，这是因为浅层网络训练比深层网络更为容易，而在深层网络中，预学习的效果体现的比较明显。这是因为随着层数的增加，越是靠近输入层，在反向传播调整参数的时候，梯度信息越来越不明显。<font color="red">这也说明一点，在深度网络中，利用预学习学习前面几层(靠近输入层)比训练后面几层(靠近输出层的)更有意义</font>。

#### &nbsp;&nbsp;&nbsp;&nbsp;3. 下面要看的是，隐层节点的个数对预学习是否有影响。这个在文章中也做了实验：
<div style="text-align: center">
<img src="../images/pre-7.jpg">
<img src="../images/pre-8.jpg">
</div>

#### 在这个图上，每层的节点数分别为25,50,100,200,400,800(手写体库上)，分别在有1个隐层，2个隐层，和3个隐层的网络上实验。从图上可以看到，在一层网络中，节点要在将近300个一上，预学习的测试误差才能小于随机初始化的网络误差，而随着网络层数的增大，到三层网络时，节点个数的影响稍微变小了一点。<font color="red">这说明，预学习在浅层和节点个数少的情况下并不适用</font>。

#### &nbsp;&nbsp;&nbsp;&nbsp;4. 下面一个点是说，前面已经提到了，预学习将参数规划到了使得梯度下降到比较深的参数空间，那么如果从预学习画定的这个空间中随机初始化参数，是不是也能到达和预学习同样的效果？文献里面，对这一问题也进行了实验：
<div style="text-align: center">
<img src="../images/pre-6.jpg">
</div>

#### 上述表格第一列表示两个网络，一个是一层一个是两层网络，第二列表示参数是随机初始化的，第二列表示在预学习划定的参数变动范围和分布上随机初始化参数，第三列表示预学习得到参数，表格里面的数据为测试误差。从这个表格上可以看到，即使在预学习得到的参数范围和分布上随机初始化得到的结果比随机初始化得到的要好一些，但依然没有预学习得到的结果好。<font color="red">这说明预学习并不是仅仅提供一个参数初始化范围和分布</font>(具体还有什么，文章里没有提~, 然后文献[4]中hinton又提出了好的初始化加合理动量能得到好的效果(待进一步研究一下)。


#### **Reference**

#### [1] Erhan D, Manzagol P A, Bengio Y, et al. The difficulty of training deep architectures and the effect of unsupervised pre-training[C]//International Conference on Artificial Intelligence and Statistics. 2009: 153-160.

#### [2] Erhan D, Bengio Y, Courville A, et al. Why does unsupervised pre-training help deep learning?[J]. The Journal of Machine Learning Research, 2010, 11: 625-660.

#### [3] Bengio Y, Lamblin P, Popovici D, et al. Greedy layer-wise training of deep networks[J]. Advances in neural information processing systems, 2007, 19: 153.

#### [4]Sutskever I, Martens J, Dahl G, et al. On the importance of initialization and momentum in deep learning[C]//Proceedings of the 30th International Conference on Machine Learning (ICML-13). 2013: 1139-1147.
