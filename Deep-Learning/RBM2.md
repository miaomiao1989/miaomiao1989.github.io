---
layer：default
---
#**RBM2**

####Restricted Boltzmann Machine (RBM) 根据可视层节点和隐层节点的类型，可以分为不同的模型([1],[2])。 而可视层节点的分布通常是依赖于数据集的。例如，比较常用的是binary可视层节点，通常是指伪的二值分布，即数据分布峰值在0和1处，常见的例如手写体字体。而real-value数据，例如自然图像，其分布通常表现为高斯分布模型(单峰值)或混合高斯模型(多峰值)--这里通常是将数据归一化到0-1之间，观看其直方图分布。

####**Bernoulli-Bernoulli RBM**

####Bernoulli-Bernoulli RBM是指可视层节点和隐层节点都服从二值分布。假设输入层为$$v$$，隐层为$$h$$，则联合概率分布为:
<div style="text-align: center">
<img src="../images/RBM2-1.jpg">
</div>

####其中能量函数为：
<div style="text-align: center">
<img src="../images/RBM2-2.jpg">
</div>

####$$v_{i},h_{j}$$为第$$i$$个可视层节点和第$$j$$个隐层节点，$$a_{i},b_{j}，w_{ij}$$为可视层和隐层的偏置和两层之间的权值。参数求解的目标函数为：
$$-\log p(v,h)$$

####由这一目标函数可以推导出参数更新公式：
$$w_{ij}=w_{ij}+\alpha(<v_{i}h_{j}>_{data}-<v_{i}h_{j}>_{recon})$$

$$b_{i}=b_{i}+\alpha(<v_{i}>_{data}-<v_{i}>_{recon})$$

$$a_{j}=a_{j}+\alpha(<h_{j}>_{data}-<j_{j}>_{recon})$$

####
- <h4>Gaussian-Bernoulli RBM是指可视层节点服从高斯分布，隐层节点服从二值分布。多隐层RBM通常是第一层为Gaussian-Bernoulli RBM，后面几层为stack Bernoulli-Bernoulli RBM 构成的。</h4>
- <h4></h4>


####**Reference**
####[1] Yamashita T, Tanaka M, Yoshida E. To be Bernoulli or to be Gaussian, for a Restricted Boltzmann Machine[J]. To be Bernoulli or to be Gaussian, for a Restricted Boltzmann Machine, 2014: 1520-1525.

####[2] 