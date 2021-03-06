---
layout: default
---

# **mean-shift**

&emsp;这里主要讨论的是文献[1]中的基于meanshift的跟踪方法。经典的meanshift跟踪方法不能处理尺度变化问题(fixed scale)，因为没有尺度估计，这样会导致跟踪目标在存在尺度变化的时候将背景也会计算进去，从而会导致跟踪的失败。这里文献[1]不仅考虑了尺度变化，在计算下一帧的位置变化的时候将背景的直方图也考虑进去了，进一步提高了跟踪的准确度。

**1. 经典的mean-shift跟踪算法**

&emsp;mean-shift算法是一种无参数方法，是一种无模板更新的跟踪方法，由于其计算简便，速度快，而被广泛应用于跟踪中。mean-shift跟踪的本质是计算目标框直方图和候选框直方图的最小距离(相似性)。由于直方图的计算不涉及到搜索框的空间结构，因此，mean-shift跟踪方法适用于可变形和较为清晰的目标跟踪。

&emsp;在经典的mean-shift跟踪算法中，目标是用含有m个bin的核运算之后的直方图进行表示的：

$$\hat{q}=\{\hat{q}_{u}\}_{u=1...m}\ \ \ \sum_{u=1}^{m}\hat{q}_{u}=1$$

&emsp;其中$$u$$表示直方图中的m个bin中的某个。

&emsp;在待检测图像帧中的候选目标框的直方图表示为(以坐标$$y$$为中心)：

$$\hat{p}(y)=\{\hat{p}_{u}(y)\}_{u=1...m}\ \ \ \sum_{u=1}^{m}\hat{p}_{u}=1$$

&emsp;<font color="red">目标框:</font>假设$$\{x_{i}^{*}\}_{i=1...n}$$表示目标图像中以原点为中心的像素的坐标，运用核函数$$k(x)，b:R^{2}\rightarrow 1...m$$将位于坐标位置$$x_{i}^{*}$$的像素值映射到相应的第u个bin的直方图中。目标的概率直方图$$u\in \{1,\dots,m\}$$计算方式为：

$$\hat{q}_{u}=C\overset{n}{\underset{i=1}{\sum}}k(\|x_{i}^{*}\|^{2})\delta[b(x_{i}^{*}-u)]$$

其中$$\delta$$是克罗内克函数，直方图映射值$$b(x_{i}^{*})==u$$时值为1，否则值为0。直观意思就是，仅仅只在像素值映射到的相应的bin上进行相加操作。而$$C$$是归一化因子，目的是使得$$\sum_{u=1}^{m}\hat{q}_{u}=1$$。

&emsp;<font color="red">候选目标框：</font>假设在当前图像帧上，$$\{x_{i}\}_{i=1...n_{h}}$$表示以$$y$$为中心点的目标候选框的像素位置坐标，$$n_{h}$$为目标候选框中的像素个数。用相同的核函数和尺度参数$$h$$，目标候选框的概率表示为：

$$
\hat{p}_{u}(y)=C_{h}\sum_{i=1}^{n_{h}}k(\|\frac{y-x_{i}}{h}\|^{2})\delta[b(x_{i})-u]
$$

其中$$C_{h}$$为归一化因子。而概率分布$$\hat{q},\hat{p}$$之间的距离(相似性)是由Hellinger distance来测量的：

$$H(\hat{p}(y), \hat{q})=\sqrt{1-\rho[\hat{p}(y),\hat{q}]}$$

$$\rho[\hat{p}(y),\hat{q}]=\sum_{i=1}^{m}\sqrt{\hat{q}_{u}(y)\hat{q}_{u}}$$

&emsp; <font color="red">而上述距离</font>$$H$$<font color="red">最小的那个中心位置</font>$$y$$<font color="red">即是我们寻找的当前图像帧的目标应该所在的中心点位置。因为寻找目标位置转化为求解最小最小距离H。而使用梯度下降法从初始位置中心点坐标</font>$$\hat{y}_{0}$$<font color="red">开始搜索新的坐标位置等价于mean-shift方法。迭代地从初始位置移动到新的中心点位置计算为：</font>

$$
\hat{y}_{1}=\frac{\sum_{i=1}^{n_{h}}x_{i}w_{i}g(\|\frac{\hat{y}_{0}-x_{i}}{h}\|^{2})}{\sum_{i=1}^{n_{h}}w_{i}g(\|\frac{\hat{y}_{0}-x_{i}}{h}\|^{2})}
$$

&emsp;其中$$g(x)=-k'(x)$$是核函数的导数，$$\hat{y}_{0}$$为初始位置(可以为上一帧的目标位置，也可以为在当前帧搜索过程中的上一次迭代的目标位置作为下一次迭代的初始位置)，$$x_{i}$$为候选目标框中的像素的坐标位置，$$n_{h}$$为候选目标框中的像素个数，$$h$$为尺度参数(在第一帧初始化过程中，尺度参数为1)，$$w_{i}$$为权值：

$$
w_{i}=\sum_{u=1}^{m}\sqrt{\frac{\hat{q}_{u}}{\hat{p}_{u}(\hat{y}_{0})}}\delta[b(x_{i})-u]
$$

&emsp;简单来说针对每个位置$$i$$的权值$$w_{i}$$即为该位置的像素所在的直方图的bin的比值(目标直方图和目标候选直方图)，是根据初始位置$$\hat{y}_{0}$$计算得到的。

&emsp;<font color = "red">迭代执行</font>$$\hat{y}_{1}$$和$$w_{i}$$<font color="red">直到收敛(即</font>$$\hat{y}_{1},\hat{y}_{0}$$<font color="red">之间的差距足够小)或是达到指定最大迭代次数，即为最后寻找到的结果。</font>

**2. 带尺度mean-shift(考虑入背景直方图)**








**Reference**

[1] Tomas Vojir, **Robust scale-adaptive mean-shift for tracking**, CVPR2013