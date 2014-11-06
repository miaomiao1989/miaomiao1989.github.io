---
layout: post
---
<table align="left">
    <h4 style="text-indent: 2em;"><a href= "../index.html">Deep-Learning</a></h4>
</table>

#**CNN预学习-PSD**

####这博文主要总结一下“Learning convolutional feature hierarchies for visual recognition”这篇论文。

####&nbsp;&nbsp;&nbsp;&nbsp;传统的CNN模型，是一种有监督模型，需要随机初始化卷积滤波，再利用反向传播方法对滤波进行调整，由前面的博文中我们也提到过，当网络结构很复杂，网络层数变多时，靠近输入层的参数是很难调整的，这就需要无监督方法预学习卷积滤波。

####&nbsp;&nbsp;&nbsp;&nbsp;在进行特征提取的过程中，我们既希望特征能够包含足够多的信息，又希望能够降低计算复杂程度，无疑稀疏特征在一定程度上满足了我们的这种要求。在前面博文中介绍CDBN的文章中，我们也提到了，在经过卷积得到的特征维数是很大的，为了降低计算复杂度，我们希望能够保留激活最大的特征，因此希望特征map是稀疏的。在这里，是同样的出发点，即：希望卷积得到的特征map具有稀疏性。

####&nbsp;&nbsp;&nbsp;&nbsp;上文提到了稀疏性，我们首先先来简单的回顾一下稀疏编码(sparse coding)。稀疏编码的主要思想是，对于给定的输入样本向量$$x\in R^{n\times 1}$$在字典$$D\in R^{n\times m}(n<m)$$下寻找一组稀疏向量$$z\in R^{m\times 1}$$:
![](../images/PSD-1.jpg)

####&nbsp;&nbsp;&nbsp;&nbsp;由稀疏编码的这一思想，可以看到，比较符合我们上面需要求解稀疏特征映射的问题，但是在直接应用稀疏编码到卷积情况下会存在一些问题。例如，传统的稀疏编码的输入$$x$$为图像patch，且相互之间的独立的，他们对应的稀疏系数都是独立进行求解的，并且在应用于整个图像时，输入patch之间是有重叠的，这造成了大量的冗余；当在训练样本上训练得到字典$$D$$和稀疏系数$$z$$之后，对于新的测试样本，需要重新求解优化模型，利用字典$$D$$求解系数$$z$$，而这一求解过程通常需要迭代求解，是非常耗时的。

####对已第一个问题，可以用卷积滤波对整个图像做卷积来减少特征冗余，由此产生了稀疏卷积编码，这和前面博文讲到的Deconvolution network反卷积网络非常相似(<a href= "../DC/index.html">反卷积网络简介</a>)，可做参考。稀疏卷积编码的目标函数为：
![](../images/PSD-2.jpg)

####其中$$x\in R^{w\times h}$$是输入图像，$$D_{k}\in R^{s\times s}$$是卷积滤波，$$z_{k}\in R^{(w+s-1)\times (h+s-1)}$$是2D特征map，$$*$$表示矩阵卷积，$$K$$表示下一层与$$x$$共有$$K$$个连接。

####为了避免在测试图像上迭代求解，在论文中提出了一个回归预测器和字典同时训练，这一回归器的作用就是，在新的数据需要求解$$z$$的时候不需要再迭代求解，而是直接用回归器预测$$z$$的值。这一回归器模型可以是传统的前馈神经网络，也可以是卷积神经网络。文章中的模型为：
![](../images/PSD-3.jpg)

####其中$$W^{k}$$为编码卷积滤波，大小为$$s\times s$$. 这一模型实现了编码和解码过程，从本质上讲，是自编码的结构。

###Reference

####[1] Kavukcuoglu K, Sermanet P, Boureau Y L, et al. Learning convolutional feature hierarchies for visual recognition[C]//Advances in neural information processing systems. 2010: 1090-1098.

####[2] Kavukcuoglu K, Ranzato M A, LeCun Y. Fast inference in sparse coding algorithms with applications to object recognition[J]. arXiv preprint arXiv:1010.3467, 2010.


