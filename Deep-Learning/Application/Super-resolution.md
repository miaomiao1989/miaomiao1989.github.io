---
layout: default
---

# **Super-resolution using deep-learning**

#### &nbsp;&nbsp;&nbsp;&nbsp;超分辨率图像(Super-resolution, SR)重建又称为超采样，是从一张或多张低分辨率图像中恢复在图像退化过程中丢失的高频信息，其本质上是逆问题。常用的SR方法主要有基于插值方法,基于学习的方法和基于重建的方法。基于插值的方法方法简单，运行速度快，但是对放大因子过大时，由于缺少的信息较多，会导致边缘过于光滑。通常基于重建的方法多用于多帧图像的重建，即从低分辨率图像序列中利用低频信息之间的互补来重构高频图像。而基于学习的方法是利用外部图像库建立图像块字典，利用图像patch之间的相似性来重建高分辨率图像，这类方法不受放大因子大小的影响，因为是利用外部高频信息填补的确实信息。虽然这种方法效果较好，但是训练较为复杂，而且多依赖于训练好的字典库。

#### &nbsp;&nbsp;&nbsp;&nbsp;在文献[1]中作者提出了利用CNN实现SR，即避免了图像patch的提取也避免了字典库的训练，而是直接对整图进行重构(相对于之前基于学习的方法都是通过patch块，逐块重构再重合部分求均值实现)。这篇文章中提出的方法思路是比较容易理解的。


- 首先，对低分率图像进行bicubic插值运算，可以得到和预期的高分辨率图像有相同尺寸的初始图像$$Y$$，将其归一化到0-1之间，作为CNN网络的输入。

- 用CNN的第一层权值和偏置值$$W_{1}, B_{1}$$对初始图像$$Y$$进行卷积，并作用非线性激活函数， ReLU(ReLU = max(0, x)):

<div style="text-align: center">
<img src="../images/SR-1.jpg">
</div>


#### 其中$$W_{1}$$的维数为$$c\times f_{1}\times f_{1} \times n_{1}$$, $$n_{1}$$为滤波个数;

- 用CNN的第二层权值和偏置值$$W_{2}, B_{2}$$对第一层得到的$$F_{1}(Y)$$进行卷积，并作用非线性激活函数， ReLU(ReLU=max(0, x)):

<div style="text-align: center">
<img src="../images/SR-3.jpg">
</div>


- 用CNN的第三层权值和偏置值$$W_{3}, B_{2}$$对第二层得到的$$F_{2}(Y)$$进行卷积，不作用非线性激活函数。

<div style="text-align: center">
<img src="../images/SR-4.jpg">
</div>


- 将最后得到的数据重新归一化到0-255之间，即可得到待重构的高分辨率图像，具体过程如下图所示:

<div style="text-align: center">
<img src="../images/SR-2.jpg">
</div>


#### 另外文章中还提出了，可以将基于稀疏表示的高分辨率图像重建，看成是卷积网络的一种。



#### **Reference**

#### [1]. Image Super-Resolution Using Deep Convolutional Networks. Chao Dong, Chen Change Loy, Kaiming He, and Xiaoou Tang, TPAMI, 2014.
