---
layout: default
---

# **DSST**

#### &nbsp;&nbsp;&nbsp;&nbsp;这篇主要讨论的是Martin Danelljan 对MOSST[1]的相关滤波()跟踪做了改进，并增加了多尺度跟踪，改进效果很显著，在2014年的VOT(visual object tracking)上，其测试效果是第一的。文章名为Accurate Scale Estimation for Robust Visual Tracking[2]，其代码为DSST。

#### &nbsp;&nbsp;&nbsp;&nbsp;*注：*本文部分参考[3]

#### &nbsp;&nbsp;&nbsp;&nbsp;MOSSE[1]在求解滤波器时，其输入项是图像本身（灰度图），也就是图像的灰度特征。对于灰度特征，其特征较为简单，不能很好的描述目标的纹理、边缘等形状信息，因此DSST的作者将灰度特征替换为在跟踪和识别领域较为常用的HOG特征。

### **1. 相关滤波器**
在上一篇<a href = "KCF/index.htlm">KCF</a>中我们对相关滤波(correlation filters)做了简单的介绍，可以知道相关滤波计算速度较于传统的方法速度较快，是因为相关滤波在傅里叶域里将卷积运算变换成了按元素的内积运算。

#### &nbsp;&nbsp;&nbsp;&nbsp;相关滤波器能够用于有效定位图像中的显著特征。对于构造一个用于检测图像中的一种特殊类型目标的滤波器问题，理想的滤波器期望是在相关输出值中，目标位置产生强峰值而其他位置为0。该滤波器可以由$$n$$幅训练图像$$\{f_{1},f_{2},\cdots, f_{n}\}$$几何来构造，训练图像实例包含感兴趣目 标及背景。为了训练获得最佳的相关滤波器， 首先根据训练图像建立对应的期望输出图$$\{g_{1}, g_{2}, \cdots, g_{n}\}$$。建立的期望输出可定义为目标位置是峰值而背景位置近似为0，可以采用二维高斯函数来定义:

$$
g_{i}(x, y) = exp\{-[(x-x_{i}^{2}) + (y-y_{i})^{2}]/ \sigma^{2}
$$

#### 其中$$(x_{i}, y_{i})$$为训练图像的目标真实坐标，$$\sigma$$为高斯参数，用以调节输入的尖锐程度。对于预付图像样本，相关滤波器训练的任务是求解一个滤波器$$h_{i}$$满足一下关系：$$g_{i}=f_{i}\otimes h_{i}$$，其中$$\otimes$$是相关操作。在傅里叶域下，等价于$$G_{i} = F_{i}H_{i}^{*}$$，其中$$*$$是复共轭，$$H_{i}$$是滤波器$$h_{i}$$对应的傅里叶操作，称为模板。这样，对于单一的训练样本，能够得到的精确的滤波器结果$$H_{i}^{*} = G_{i}/F_{i}$$, $$H_{i}$$能够完全精确的奖$$f_{i}$$变换到$$g_{i}$$，在傅里叶域下，求解可以等价于：

$$
H_{i}^{*} = \frac{G_{i}F_{i}^{*}}{F_{i}F_{i}^{*}}
$$

### **2. DSST跟踪**

#### &nbsp;&nbsp;&nbsp;&nbsp;DSST跟踪算法使用了相关滤波器。DSST将跟踪分为两个部分，位置变化(translation)和尺度变化(scale estimation)。在跟踪的实现过程中，作者定义了两个correlation filter，一个滤波器(translation filter)专门用于确定新的目标所处的位置，另一个滤波器(scale filter)专门用于尺度评估。

- #### **位置变化(translation)**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**1. 训练过程**

#### &nbsp;&nbsp;&nbsp;&nbsp;训练过程即为学习相关滤波器$$H$$的过程。

#### &nbsp;&nbsp;&nbsp;&nbsp;假如图像块(当前帧目标周围采样)提取出$$d$$个特征(例如HOG), 令$$f=[f_{1},f_{2}, \cdots, f_{d}]$$表示图像块特征($$f^{\ell}, 0<\ell<d$$)，训练过程即为寻找最优的相关滤波$$h=[h_{1}, h_{2}, \cdots, h_{d}]$$，每维特征相应的滤波器记为$$h^{\ell}, 0<\ell<d$$，目标函数为：

$$
\epsilon = \|\overset{d}{\underset{\ell = 1}{\sum}}h^{\ell} \star f^{\ell}-g \|^{2} + \lambda \overset{d}{\underset{\ell=1}{\sum}}\|h^{\ell}\|^{2}
$$

这里$$g$$为训练样特征$$f$$相对应的输出值(由上一节可以知道，可以用二维高斯函数定义，越靠近中心点越接近于1，越远离中心点越接近于0)

#### &nbsp;&nbsp;&nbsp;&nbsp;则上述目标函数的解为：

$$
H^{\ell} = \frac{\overline{G}F^{\ell}}{\sum_{k=1}^{d}\overline{F^{k}}F^{k}+\lambda}
$$

这里$$F^{\ell}$$为将特征转换到傅里叶域之后的值，$$\overline{F}, \overline{G}$$表示复共轭，$$G$$表示目标输出$$g$$转换到傅里叶域的值。<font color = "red">将上式中的分子，分母分别记为</font>$$A_{t}，B_{t}$$<font color = "red">(t表示第t帧)</font>。


**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2. 检测过程**
  
#### &nbsp;&nbsp;&nbsp;&nbsp;假设我们在第$$t$$帧训练好了相关滤波器$$H^{\ell}$$，在第$$t+1$$帧进行检测时，只需要计算最大的响应值：

$$
y = \mathscr{F}^{-1}
\left\{
\frac{\sum_{\ell=1}^{d}\overline{A_{t}^{\ell}}Z^{\ell}}
{B_{t}+\lambda}
\right\}
$$

其中$$Z^{\ell}$$为第$$t+1$$帧的第$$\ell$$个特征在傅里叶域的值。只需要寻找$$y$$中最大的值即为t+1帧目标所在位置。然后更新分子分母如下：

$$
\begin{array}{cc}
A_{t+1}^{\ell} &=& (1-\eta)A_{t}^{\ell} + \eta\overline{G}_{t+1}F_{t+1}^{\ell}\\
B_{t+1} &=& （1-\eta)B_{t} + \eta\sum_{k=1}^{d} \overline{F^{k}_{t+1}}F_{t+1}^{k}
\end{array}
$$

- **尺度变化(scale estimation)**

#### &nbsp;&nbsp;&nbsp;&nbsp;尺度变化是和位置变化分开的，是在检测好了位置之后，再进行的尺度估计。在进行尺度估计的时候，假设我们预先设定有33个尺度，中间的第13个尺度的尺度因子为1， 13-1逐渐减小， 13-33逐渐增大。所谓的尺度估计就是从这33个尺度中找对一个最相近的尺度。原理和上面位置变换的相关滤波是一样的。假设初始图像的跟踪框为[50,50],假设第一个尺度为1.5，
则跟踪框应该为[50*1.5, 50*1.5] = [75, 75]，则在原始图像上以跟踪框的中心点为中心，裁取[75,75]的图像块，强行将图像块缩放到初始化的跟踪框大小[50,50]，再提取hog特征，并将特征拉成列向量，构成800x33维的矩阵，其中800为特征维数，33为尺度个数。

#### &nbsp;&nbsp;&nbsp;&nbsp;在选定的尺寸中，假设我们从33个尺寸中选好了1.02，再进行下一帧的位置检测的过程中，裁取尺度上的图像块，强行缩放到初始化的图像块大小，再进行预测，这样做的目标的为了保证在处理过程中
始终能够保持图像大小尺寸的一致性，能使滤波在各种尺度上都能够使用。

#### &nbsp;&nbsp;&nbsp;&nbsp;假设我们就虚线设定有33个尺度，对这33个尺度建立输出$$gs$$值，一共33个(以指数方式生成)。训练过程中学习相关滤波$$HS$$, 将图像在不同33个尺度上提取的所有的特征拉成列向量，作为该尺度下的特征$$fs = [fs_{1}, fs_{2}, \cdots, fs_{33}]$$, 将$$gs,fs$$均转换到傅里叶域得到$$Gs, Fs$$，则相关滤波$$Hs$$:

$$
Hs = \frac{\overline{Gs}Fs}{\sum\overline{Fs}{Fs}+\lambda}
$$

#### 之后的寻找最大相应的计算方法和更新分子分母的方法和位置变化的计算方式是相同的，在33个尺度中选择相应最大的一个作为当前帧的尺度。<font color="red">（位置变化检测和尺度变化检测所不同的仅仅是特征的构造，和目标输出的构造不同，其他的计算方法均相同）</font>


#### **Reference**

[1] D. S. Bolme,  **Visual object tracking using adaptive correlation filters**, CVPR 2010.

[2] M. Danelljan, **Accurate scale estimation for robust visual tracking**, BWVA 2014.

[3] <http://blog.csdn.net/autocyz/article/details/48651013>
