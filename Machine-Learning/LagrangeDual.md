---
layout: default
---
#**拉格朗日对偶**#

####拉格朗日对偶性是解决带约束的最优化问题的方法，在实际应用中，通过拉格朗日对偶原理将原始问题转换成对偶问题，将原来不容易解决的问题转化为一个容易解决的问题，如支持向量机。

####**原始问题**

####假设$$f(x),g(x)$$是定义在$$R^{n}$$上的连续可微函数，原始问题如下所示：
<div style="text-align: center">
<img src="../images/LD-1.jpg">
</div>

####引进广义拉格朗日函数
<div style="text-align: center">
<img src="../images/LD-2.jpg">
</div>

####那么原始问题等价于如下问题
<div style="text-align: center">
<img src="../images/LD-3.jpg">
</div>

####即：
<div style="text-align: center">
<img src="../images/LD-12.jpg">
</div>

####这是因为如果约束条件不满足，即$$g(x)>0$$，那么那么总可以找到一个$$\lambda$$，使得$$L(x,\lambda)>f(x)$$,即
<div style="text-align: center">
<img src="../images/LD-13.jpg">
</div>

####我们定义原始问题的最优值为原始问题的值。
<div style="text-align: center">
<img src="../images/LD-14.jpg">
</div>

####**对偶问题**

####将原始问题极小极大顺序互换后的极大极小问题称为原始问题的对偶问题,如下所示
<div style="text-align: center">
<img src="../images/LD-15.jpg">
</div>

####定义对偶问题的最优值为对偶问题的值。
<div style="text-align: center">
<img src="../images/LD-16.jpg">
</div>

####**原始问题和对偶问题的关系**

####假设函数$$f(x)$$是凸函数，并且不等式$$g(x)$$是严格可行的，则$$x^{*},\lambda^{*}$$分别是原始问题和对偶问题的解的充分必要条件是以下的Karush-Kuhn-Tucker(KKT)条件成立：
<div style="text-align: center">
<img src="../images/LD-17.jpg">
</div>

####**拉格朗日对偶法学习字典**

####假设稀疏系数已知的条件下，字典学习的基本公式为
<div style="text-align: center">
<img src="../images/LD-6.jpg">
</div>

####这是一个带有二次约束的最小二乘优化问题，通常的梯度下降法也可以求解，也可以用拉格朗日对偶法进行求解。首先将上式变为拉格朗日约束问题：
<div style="text-align: center">
<img src="../images/LD-7.jpg">
</div>

####其中$$\lambda_{j}>0$$是对偶变量。最小化字典$$B$$, 得到拉格朗日对偶公式：
<div style="text-align: center">
<img src="../images/LD-8.jpg">
<img src="../images/LD-9.jpg">
</div>

####计算梯度和Hession为：
<div style="text-align: center">
<img src="../images/LD-10.jpg">
</div>

####其中$$e_{i}\in R^{n}$$是第$$i$$个单位向量。利用牛顿法或共轭梯度法求得上述拉格朗日对偶问题的解，即字典的更新为：
<div style="text-align: center">
<img src="../images/LD-11.jpg">
</div>

###Reference

####[1] Lee H, Battle A, Raina R, et al. Efficient sparse coding algorithms[C]//Advances in neural information processing systems. 2006: 801-808.



