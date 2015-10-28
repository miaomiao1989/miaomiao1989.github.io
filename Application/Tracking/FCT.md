---
layout: default
---

#**FCT**

####&nbsp;&nbsp;&nbsp;&nbsp;这里主要讨论的是文献[1]compressive tracking(CT)的优化算法[2] Fast comressive tracking(FCT).

####&nbsp;&nbsp;&nbsp;&nbsp;CT是一种简单高效地基于压缩感知的跟踪算法。首先利用符合压缩感知RIP条件的随机感知矩对多尺度图像特征进行降维，然后在降维后的特征上采用简单的朴素贝叶斯分类器进行分类。该跟踪算法非常简单，但是实验结果很鲁棒，速度大概能到达40帧/秒。和一般的模式分类架构一样：先提取图像的特征，再通过分类器对其分类，不同在于这里特征提取采用压缩感知，分类器采用朴素贝叶斯。然后每帧通过在线学习更新分类器。

- ###**压缩感知**

#### &nbsp;&nbsp;&nbsp;&nbsp;首先我们来简单了解一下压缩感知理论。<font color = "red">简单地说，压缩感知理论指出：只要信号是可压缩的或在某个变换域是稀疏的，那么就可以用一个与变换基不相关的观测矩阵将变换所得高维信号投影到一个低维空间上，然后通过求解一个优化问题就可以从这些少量的投影中以高概率重构出原信号，可以证明这样的投影包含了重构信号的足够信息。</font>
在该理论框架下，采样速率不再取决于信号的带宽，而在很大程度上取决于两个基本准则：<font color = "red">稀疏性和非相关性，或者稀疏性和等距约束性</font>。







####**Reference**

[1] Kaihua Zhang, **Real-time compressive tracking**, CVPR 2012.

[2] Kaihua Zhang, **Fast compressive tracking**, TPAMI 2014.