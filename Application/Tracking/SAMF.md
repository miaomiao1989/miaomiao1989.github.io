---
layout: default
---

# **SAMF**

#### 这里主要讨论的是文献[1]中的方法，其是对KCF[2]的改进，因为KCF方法无法处理尺度问题，这里主要是修改其能够进行尺度变化。

### **1. 回归跟踪**

#### &nbsp;&nbsp;&nbsp;&nbsp;当前实现跟踪的方法基本都是tracking-by-detection，将跟踪问题转换成二分类或回归问题。具体过程如下：


#### &nbsp;&nbsp;&nbsp;&nbsp;**训练过程：** 在训练过程中在目标框周围以窗口滑动的方式进行采样，可以得到一定数量的正负样本，用这些正负样本进行分类器训练，训练二分类分类器或回归器，采样方式如下图所示：

<div style="text-align: center">
<img src="../Images/SAMF1.png">
</div>

#### &nbsp;&nbsp;&nbsp;&nbsp;而在下一帧的测试过程中，在上一帧的目标框周围采样样本，用训练好的费雷器对采样的样本进行筛选，选出最相似的样本作为当前针的预测结果。测试过程中的采样如下图所示：

<div style= "text-align:center">
<img src = "../Images/SAMF2.png">
</div>

#### &nbsp;&nbsp;&nbsp;&nbsp;假设我们通过上述采样得到了一系列样本$$\{(x_{1}, y{1}), (x_{2}, y_{2}, \cdots, (x_{n}, y_{n}))\}$$,其中$$x$$为采样得到的样本，$$y$$是样本对应的响应值。则可以通过回归建立样本和响应值之间的对应关系<font color = "red">(公式太多部分，直接贴了笔记^_^)</font>

<div style= "text-align:center">
<img src = "../Images/SAMF32.jpg">
</div>
<div style= "text-align:center">
<img src = "../Images/SAMF4.jpg">
</div>





#### &nbsp;&nbsp;&nbsp;&nbsp;由上述过程可知，在测试样本中采样测试样本越是密集的采样得到的结果会更精确，而同时，越密集的采样会越耗时。




#### **Reference**

[1] A scale adaptive kernel correlation filter tracker with feature integration, 2014.

[2] High-speed tracking with kernelized correlation filters, TPAMI 2015.
