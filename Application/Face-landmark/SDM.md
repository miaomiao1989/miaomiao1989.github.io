---
layout: default
---
#**Supervised descent method**

####&emsp;&emsp;这里主要整理的是文献[1]中的人脸特征点检测方法。这篇文章是从牛顿下降法的观点来求解特征点检测问题。

####emsp;&emsp;给定一张含有$$m$$个像素点的图像$$d\in R^{m\times 1}$$，图像中的$$p$$个landmark表示为$$d(x)\in R^{p\times 1}$$，单个landmark表示为$$x^{*}$$.$$h$$表示为sift非线性特征提取方法，$$h(d(x))\in R^{128\times 1}$$表示提取到的sift特征。由此，人脸特征点检测方法可以表示为如下公式：

 <div style="text-align: center">
 <img src="../Images/sdm1.jpg">
 </div>

####其中$$\Phi_{*} = h(d(X_{*}))$$表示以landmark为中心的sift特征。

####&emsp;&emsp;SDM的目标是学习一系列梯度下降方向和尺度因子用来更新$$x_{k+1}=x_{k} +\Delta x_{k}$$，从$$x_{0}$$收敛到$$x_{*}$$。将上式中的$$f(x+\Delta x)$$看成是二阶可导，并用泰勒展开进行展开：

<div style="text-align: center">
<img src="../Images/sdm2.jpg">
</div>

####其中$$J_{f}(x_{0})和H_{x_{0}}$$分别为Jacobian和Hessian矩阵。


####**Reference**

####[1] Supervised descent method and its applocations to face alignment, CVPR2013.