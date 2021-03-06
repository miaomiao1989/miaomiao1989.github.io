---
layout: default
---

# **Struck**

#### &nbsp;&nbsp;&nbsp;&nbsp;这里主要讨论的是《Struck:Structured Output Tracking with Kernels》[1]的跟踪算法，struck跟踪算法与传统的tracking-by-detection算法的不同之处，可以用如下图表示：

<div style="text-align: center">
<img src="../Images/struck1.png">
</div>

#### 上图中的右边显示了传统的tracking-by-detection跟踪方法，将跟踪问题转化为二分类问题，训练在线的分类器，对采样的样本进行二分类，然而，为了达到更新分类器的目的，通常需要将一些预估计的目标位置作为已知类别的训练样本，这些分类样本并不一定与实际目标一致(比如说跟踪错误发生的情况会导致输入给分类器训练的正负样本有错误)。另一方面，分类器的在线更新作为中间步骤，我们无法控制输入的哪些样本对分类器的训练是有正的作用，而哪些样本会起到负的作用。因此难以实现最佳的分类效果。 而struck跟踪方法为上图中的左图显示，省略了中间分类器二分类的过程，从而避免了由传统二分类分类器在跟踪过程中带来的误差。换句话说，在样本的采集过程中的类标由$$(x,+1), (x,-1)$$变成了$$(x,y)$$，其中的$$y$$不再指示样本的正负性，而是指向了样本的坐标位置。可以简单的理解为将问题由分类问题转换成了回归问题。

#### &nbsp;&nbsp;&nbsp;&nbsp;struck跟踪算法使用的是在线结构化输出SVM算法实现跟踪的。通过学习一个预测函数$$f:X-> Y$$来直接预测两帧之间的目标位置变化。而训练样本对应的输出空间不再是二值类标+1，-1，而是所有的变换$$y$$组成的空间。为了在structured output SVM中学习到函数$$f$$，文中定义了一个离散函数:

<div style="text-align: center">
<img src="../Images/struck2.png">
</div>

#### 函数$$g$$度量了训练样本对$$(x,y)$$的相容性，并计算出这些样本中的最高的匹配度。文章中限制了函数$$g$$为线性函数$$g(x,y) = <w, \Phi(x,y)>$$, 其中$$\Phi(x,y)$$为核函数。从一系列样本$$\{(x_{1}, y_{1}),\cdots,(x_{n}, y_{n})\}$$通过如下凸优化方法学习得到函数$$g$$:

<div style="text-align: center">
<img src="../Images/struck3.png">
</div>

#### 通过优化方法选择出作为支持向量的样本，而随着长时间的跟踪，所选出的支持向量样本会越来越多，到导致跟踪越来越慢，为了解决这一现象，文中提到了限制支持向量的个数。假设说需要限制的支持向量的个数为100个，当支持向量的个数超过100个之后需要在现有的支持向量集中剔除一部分不需要的。而文中提出的删除方法是<font color="red">删除对权值贡献最小的的支持向量(即：导致权w变化最小的，</font>$$\|\Delta W\|$$<font color = "red">最小).</font>考虑的是对目标的贡献，而不是考虑的支持向量内部之间的关系。

#### &nbsp;&nbsp;&nbsp;&nbsp;这里简单讲一下实施过程：

 1. 训练过程：在目标框周围一定范围内采集一定数量的样本，将样本的坐标(x,y,w,h)——>yv表示结构化输出值，将采集的样本提取192维harr特征——>X, 构成训练图像对{(X，yv)}，用SVM进行训练，选择支持向量；
 2. 测试过程中：在预测目标框周围进行密集采样(逐像素滑动)采集样本，处理方式和1相同，将所有的图像对$$\{(X_{i}, yv_{i})\}$$计算$$g(X_{i},yv_{i}) = <w, \Phi(X_{i},yv_{i})>$$的score分数，选择所有采样得到的样本中score分数最高的样本对应的变换yv值作为最后的目标位置输出。

