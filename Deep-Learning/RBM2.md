---
layout: default
---

#**RBM2**

###**1. RBM节点类型**

####&nbsp;&nbsp;&nbsp;&nbsp;Restricted Boltzmann Machine (RBM) 根据可视层节点和隐层节点的类型，可以分为不同的模型([1],[2])。 而可视层节点的分布通常是依赖于数据集的。例如，比较常用的是binary可视层节点，通常是指伪的二值分布，即数据分布峰值在0和1处，常见的例如手写体字体。而real-value数据，例如自然图像，其分布通常表现为高斯分布模型(单峰值)或混合高斯模型(多峰值)--这里通常是将数据归一化到0-1之间，观看其直方图分布。

####**Bernoulli-Bernoulli RBM**

####&nbsp;&nbsp;&nbsp;&nbsp;Bernoulli-Bernoulli RBM是指可视层节点和隐层节点都服从二值分布。假设输入层为$$v$$，隐层为$$h$$，则联合概率分布为:
<div style="text-align: center">
<img src="../images/RBM2-1.jpg">
</div>

####其中能量函数为：
<div style="text-align: center">
<img src="../images/RBM2-2.jpg">
</div>

####&nbsp;&nbsp;&nbsp;&nbsp;$$v_{i},h_{j}$$为第$$i$$个可视层节点和第$$j$$个隐层节点，$$a_{i},b_{j}，w_{ij}$$为可视层和隐层的偏置和两层之间的权值。参数求解的目标函数为：

$$\min -\log p(v,h)$$

####&nbsp;&nbsp;&nbsp;&nbsp;由这一目标函数可以推导出参数更新公式：

$$w_{ij}=w_{ij}+\alpha(<v_{i}h_{j}>_{data}-<v_{i}h_{j}>_{recon})$$

$$b_{i}=b_{i}+\alpha(<v_{i}>_{data}-<v_{i}>_{recon})$$

$$a_{j}=a_{j}+\alpha(<h_{j}>_{data}-<j_{j}>_{recon})$$

####&nbsp;&nbsp;&nbsp;&nbsp;Bernoulli-Bernoulli的条件概率为：

$$p(h_{j}=1|v)=sigmoid(b_{j}+\underset{i}{\sum}v_{i}w_{ij})$$

$$p(v_{i}=1|h)=sigmoid(a_{i}+\underset{j}{\sum}h_{j}w_{ij})$$

####&nbsp;&nbsp;&nbsp;&nbsp;<font color='red'>因为在Bernoulli-Bernoulli RBM中，大部分的数据是分布在0,1处，因此实际上是并不能很准确的表达有用信息的，正是因为这一原因，如果将自然图像的实值数据归一化到[0,1]之间，用Bernoulli-Bernoulli RBM来刻画是不合理的(参见文献[2]，[3]第30页)。</font>

####**Gaussian-Bernoulli RBM**

####&nbsp;&nbsp;&nbsp;&nbsp;Gaussian-Bernoulli RBM是指可视层节点服从高斯分布，隐层节点服从二值分布。多隐层RBM通常是第一层为Gaussian-Bernoulli RBM，后面几层为stack Bernoulli-Bernoulli RBM 构成的。能量函数为：
<div style="text-align: center">
<img src="../images/RBM2-3.jpg">
</div>

####&nbsp;&nbsp;&nbsp;&nbsp;其中$$\sigma_{i}$$为输入数据$$v$$每个节点的标准方差。<font color="red">为了便于计算, 这里通常我们会首先对数据进行归一化处理，使得输入层的每个节点满足均值为0，方差为1的高斯分布。</font>参数更新公式为：

$$w_{ij}=w_{ij}+\alpha(<\frac{1}{\sigma_{i}^{2}}v_{i}h_{j}>_{data}-<\frac{1}{\sigma_{i}^{2}}v_{i}h_{j}>_{recon})$$

$$b_{i}=b_{i}+\alpha(<\frac{1}{\sigma_{i}^{2}}v_{i}>_{data}-<\frac{1}{\sigma_{i}^{2}}v_{i}>_{recon})$$

$$a_{j}=a_{j}+\alpha(<h_{j}>_{data}-<j_{j}>_{recon})$$

####&nbsp;&nbsp;&nbsp;&nbsp;Gaussian-Bernoulli RBM条件概率为(参见文献[3])：

$$p(h_{j}=1|v)=sigmoid(b_{j}+\underset{i}{\sum}\frac{1}{\sigma_{i}}v_{i}w_{ij})$$

$$p(v_{i}=1|h)=\mathcal{N}(a_{i}+\underset{j}{\sum}h_{j}w_{ij}, \sigma_{i}^{2})$$

####&nbsp;&nbsp;&nbsp;&nbsp;这里实现过程是计算：$$p(v_{i}=1|h)=a_{i}+\underset{j}{\sum}h_{j}w_{ij}$$+高斯分布的随机数。

####&nbsp;&nbsp;&nbsp;&nbsp;Gaussian-Bernoulli RBM相比较于Bernoulli-Bernoulli RBM是难以学习的，因为Gaussian输入的数据没有上下界限，不像Bernoulli分布数据介于[0,1]之间，而对于RBM来说，得到的隐层节点和可视层节点分布于[-1,1]之间是比较合理的， 也正是因为之一原因，Bernoulli-Bernoulli RBM更为稳定(参见文献[2])。

####**Gaussian-Gaussian RBM**

####&nbsp;&nbsp;&nbsp;&nbsp;Gaussian-Gaussian RBM是指输入可视层和隐层都是连续的实值高斯分布数据。这一模型结构虽然很强大，但是由于输入层和隐层都是高斯分布，这使得模型受更多数据噪声的影响，模型变得更加不稳定，训练比较困难(参见文献[3]第30页，[2]第14页)。能量函数为：

<div style="text-align: center">
<img src="../images/RBM2-4.jpg">
</div>

###**1. 判别RBM(Discriminative RBM)**

####在通常使用RBM训练时，通常是利用RBM进行无监督学习，将得到的特征输入分类器，再进行分类器的学习。在这个特征学习的过程中，没有用到训练样本的类标信息，使得学习到的特征并不能够最大程度上体现类间的信息。因此，文献[4-5]提出的判别RBM直接将分类层的训练在RBM训练的过程中一起训练，不再单独训练分类器。即判别RBM的最后输出直接就是类标，而不再是特征向量。

####判别RBM主要分为三个模型，生成模型，判别模型和混合模型(参见文献[4-5])。

####**生成模型(generative model)**

####生成模型又称为无监督模型，是指在可视层到隐层之间与类标无关，基本模型如下：

<div style="text-align: center">
<img src="../images/RBM2-6.jpg">
</div>

####这个过程是由隐层$$h$$根据可视层$$x$$和输出层$$y$$共同求得，再有隐层$$h$$重构可视层$$x'$$和输出层$$y'$$,$$b,c,d$$分别为可视层，隐层和分类输出层的偏置，具体过程为：

####**假设可视层$$x$$为Bernoulli分布数据，隐层$$h$$也是Bernoulli分布**，则能量函数为：

<div style="text-align: center">
<img src="../images/RBM2-7.jpg">
</div>

####其中$$e_{y}=(1_{i=y})_{i=1}^{C}$$表示类标$$y$$中属于第$$C$$类的部分为1其余为0.

####注：相比较于原始的RBM，能量函数多了最后两项，即添加了输出层的能量。

####概率分布为：
<div style="text-align: center">
<img src="../images/RBM2-8.jpg">
</div>

####目标函数为：

$$\min -\underset{i}{\sum}\log p(y_{i}, x_{i})$$

####隐层的条件概率为：
<div style="text-align: center">
<img src="../images/RBM2-9.jpg">
</div>

####由此可以看出，隐层的条件概率和可视层$$x$$，输出层类标$$y$$都有关系。

####反向重构过程中的条件概率为：
<div style="text-align: center">
<img src="../images/RBM2-10.jpg">
</div>

####从上述的条件概率分布中可以看出，在训练过程中，可视层$$x$$和输出层类标$$y$$之间没有直接的联系，这也是该模型成为无监督生成模型的原因。

####参数更新：
<div style="text-align: center">
<img src="../images/RBM2-11.jpg">
</div>

$$\Delta W_{ij}=<x_{i}h_{j}>_{model}-<x_{i}h_{j}>_{reconstruct}$$

$$\Delta b_{i}=<x_{i}>_{model}-<x_{i}>_{reconstruct}$$

$$\Delta c_{j}=<h_{j}>_{model}-<h_{j}>_{reconstruct}$$

$$\Delta U_{jk}=<h_{j}(e_{y})_{k}>_{model}-<h_{j}(e_{y})_{k}>_{reconstruc}$$

$$\Delta d_{k}=<(e_{y})_{k}>_{model}-<(e_{y}）_{k}>_{reconstruct}$$

####**若输入的可视层为均值为0，方差为1的Gaussian分布**，则能量函数，条件概率为：

$$E(y,x,h)=\frac{\|x-b\|_{2}^{2}}{2}-h^{T}Wx-c^{T}h-d^{T}e_{y}-h^{T}Ue_{y}$$

####重构过程中的条件概率为：

$$p(x|h)=\mathcal{N}(b_{i}+\underset{j}{\sum}W_{ji}h_{j})$$

####<font color='red'>其余的，如$p(h|y,x),p(y|h)$,参数更新和Bernoulli分布的相同的计算公式。</font>

####**判别模型(discriminative model)**

####判别模型是直接最小化条件概率：

$$\min -\underset{i}{\sum}\log p(y_{i}|x_{i})$$


####在这个过程中不需要单独计算隐层节点，也没有重构过程，而是利用条件概率$$p(y|x)$$直接更新参数。

####条件概率计算为：

$$\begin{eqnarray}
p(y|x)&=&\frac{p(x,y)}{p(x)}\\
&=&\frac{\underset{h_{1}\in \{0,1\}}{\sum}\cdots\underset{h_{H}\in\{0,1\}}{\sum}exp(-E(x,y,h))}{\underset{y\in\{1,\cdots,C\}}{\sum}\underset{h_{1}\in \{0,1\}}{\sum}\cdots\underset{h_{H}\in\{0,1\}}{\sum}exp(-E(x,y,h))}
\end{eqnarray}$$

####可视层$$x$$为Bernoulli分布数据，隐层$$h$$也是Bernoulli分布，则能量函数为：

<div style="text-align: center">
<img src="../images/RBM2-7.jpg">
</div>

####因此：

$$
p(y|x)=\frac{e^{b^{T}x} e^{d_{y}+{\underset{j}{\sum}}\log(1+e^{c_{j}+U_{jy}+\underset{i}{\sum}W_{ji}x_{i}})}}{e^{b^{T}x} \underset{y^{*}\in\{1,\cdots,C\}}{\sum}e^{d_{y^{*}}+\underset{j}{\sum}\log(1+e^{c_{j}+U_{jy^{*}}+\underset{i}{\sum}W_{ji}x_{i} }) }}
$$

<div style="text-align: center">
<img src="../images/RBM2-12.jpg">
</div>


####**Reference**

####[1] Yamashita T, Tanaka M, Yoshida E. ***To be Bernoulli or to be Gaussian, for a Restricted Boltzmann Machine[J]***. To be Bernoulli or to be Gaussian, for a Restricted Boltzmann Machine, 2014: 1520-1525.

####[2] Hinton G. ***A practical guide to training restricted Boltzmann machines[J]***. Momentum, 2010, 9(1): 926.

####[3] ***Learning Natural Image Statistics with Gaussian-Binary Restricted Boltzmann Machines***, 2012.

####[4] Larochelle H, Bengio Y. ***Classification using discriminative restricted Boltzmann machines[C]***//Proceedings of the 25th international conference on Machine learning. ACM, 2008: 536-543.

####[5] Hugo Larochelle, Michael Mandel, Razvan Pascanu, Yoshua Bengio. ***Learning Algorithms for the Classification Restricted Boltzmann Machine***, 2012.