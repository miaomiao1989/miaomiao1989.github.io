---
layout: default
---

# **SRCNN**

#### 这里主要讨论的是文献[1]利用深度学习进行超分辨率重建的方法。整体的思想方法比较简单，网络结构也并不复杂。结构图像如下图所示：

<div style="text-align: center; height: 500px">
<img src="../Images/SRCNN.jpg">
</div>

#### 在整个处理过程中需要保持图像的尺寸不发生变化(输入的图像需要对给出的低分辨率图像首先进行bicubic插值，这样就可以达到和目标高分辨率图像相同的尺寸，这样在处理过程中可以不用考虑尺寸上的差异)，因此在做卷积处理的时候，需要进行和原尺寸大小相同的卷积处理。在前向 传播中多个map形成一个map的时候进行的是全连接处理(即多个map的和相加得到后面的一个map)。在反向传播过程中，使用的误差函数为Mean squared error (MSE）:

$$
L = \frac{1}{n}\sum_{i=1}^{n}\|F(Y)-X_{i}\|
$$

即为重构得到的高频图像与原始高频图像之间的误差。为了方便网络的处理，在初始输入的过程中需要对图像进行归一化(这里可以直接归一化到0-1之间即可)。

#### Refrence

[1] Kaiming He. **Image super-resolution using deep convolutional networks**, TPAMI 2015.