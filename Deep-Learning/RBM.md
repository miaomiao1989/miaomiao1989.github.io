---
layout: default
---

#**RBM**

####**1. RBM基本思想**

####&nbsp;&nbsp;&nbsp;&nbsp;Restricted Boltzmann machines（RBM）(Li Deng[1])是一种特殊的马尔科夫随机场，其含有一个随机隐层(通常为伯努利分布)和一个随机可视层(通常为高斯或伯努利分布)。RBM可以看成是双边图，可视层和隐层之间是全连接。

####&nbsp;&nbsp;&nbsp;&nbsp;在RBM中，隐层和可视层之间的联合分布$$p(v,h;\theta)$$可由能量函数$$E(v,h;\theta)$$定义为：
<div style="text-align: center">
<img src="../images/RBM-1.jpg">
</div>


####其中$$Z=\sum_{v}\sum_{h}exp(-E(v,h;\theta))$$是归一化因子。对于一个伯努利(可视层)--伯努利(隐层)的RBM，能量函数可以定义为：
<div style="text-align: center">
<img src="../images/RBM-2.jpg">
</div>

####其中$$w_{ij}$$为隐层和可视层之间的权值，$$b_{i},a_{j}$$为偏置值，$$I,J$$为隐层和可视层节点的个数。则，条件概率可以表示为：
<div style="text-align: center">
<img src="../images/RBM-3.jpg">
</div>

####其中$$\sigma (x)=1/(1+exp(-x))$$.

####&nbsp;&nbsp;&nbsp;&nbsp;同理，对于高斯(可视层)--伯努利(隐层)RBM，能量函数可以表示为：
<div style="text-align: center">
<img src="../images/RBM-4.jpg">
</div>

####&nbsp;&nbsp;&nbsp;&nbsp;相应的，条件概率分布可以表示为：
<div style="text-align: center">
<img src="../images/RBM-5.jpg">
</div>

####其中，$$v_{i}$$为服从均值为$$\sum_{j=1}^{J}(w_{ij}h_{j}+b_{i})$$方差为1的高斯分布实值。<font color='red'>高斯-伯努利RBM能够将实值随机变量转换成伯努利分布随机变化，后续可以利用伯努利-伯努利RBM继续处理。</font>

####&nbsp;&nbsp;&nbsp;&nbsp;对于高斯-伯努利，伯努利-伯努利的RBM其，参数更新是相同的，都是最小化下面目标函数进行参数更新：

$$\min -\log(p(v,j;\theta))$$

####&nbsp;&nbsp;&nbsp;&nbsp;参数更新为(Andrew [2])：

$$w_{ij}=w_{ij}+\alpha(<v_{i}h_{j}>_{data}-<v_{i}h_{j}>_{recon})$$

$$b_{i}=b_{i}+\alpha(<v_{i}>_{data}-<v_{i}>_{recon})$$

$$a_{j}=a_{j}+\alpha(<h_{j}>_{data}-<j_{j}>_{recon})$$

####其中$$<>_{data}$$为实际得到的数据，$$<>_{recon}$$为重构得到的数据。

####**2. RBM具体实现过程**

####**逐层学习**

####&nbsp;&nbsp;&nbsp;&nbsp;根据 hinton 2006[3] 文章中的RBM网络构建，假设构建带有4个隐层的全连接深度RBMs网络，每层都是一个RBM。各层的节点个数分别为：输入层节点数784，第一层隐层节点数1000，第二层节点数500，第三层节点数250，第四层节点数30。

####&nbsp;&nbsp;&nbsp;&nbsp;如果输入为伯努利分布分布可视节点，则经过sigmoid函数变化，得到第一层隐层的输出$$h^{1}\in R^{1000\times 1}$$，其中的连接权值$$W^{1}\in R^{784\times 1000}$$。之后对得到的$$h^{1}$$进行随机采样(通过与随机数的比较，将值置为0或1)，然后利用这一采样后的二值状态通过sigmoid函数对输入层进行重构，得到重构后的输入层节点$$v'$$，再通过重构的$$v'$$经过sigmoid函数重构隐层节点$$(h^{1})'$$。利用$$v,v',h^{1},(h^{1})'$$通过前面提到的公式更新权值和偏置值参数。

####&nbsp;&nbsp;&nbsp;&nbsp;<font color='red'>Note:上面提到了在隐层到输入可视层的重构之前，需要进行一步随机采样，这是2002年hinton[4]提出的一种对Gibbs采样的一种加速方法称为对比散度(Contrastive Divergence, CD)。在RBM中进行Gibbs采样的目的是通过k步Gibbs采样从原始样本中采样得到符合RBM定义的分布的随机样本。这种计算在可视层节点较多的情形下计算量是非常大的。因此，hinton提出了CD方法，仅需要一步采样就能得到足够好的近似。</font>

####&nbsp;&nbsp;&nbsp;&nbsp;由上面训练得到了第一层隐层的输出值$$h^{1}$$，将这一值作为下一层的输入，用相同的方法进行后面三层的训练，得到$$v-h^{1}-h^{2}-h^{3}-h^{4}$$，连接权值分别为$$W_{1}\in R^{784\times 1000}, W_{2}\in R^{1000\times 500}, W_{3}\in R^{500\times 250}, W_{4}\in R^{250\times 30}$$。

####**展开与精调**

####&nbsp;&nbsp;&nbsp;&nbsp;Hinton 2006[3]中提到，当完成了RBM的逐层学习之后，得到了每层的权值，需要将网络展开对权值进行精调(fine-tuning)。将网络进行展开为：
<div style="text-align: center">
<img src="../images/RBM-6.jpg">
</div>

####&nbsp;&nbsp;&nbsp;&nbsp;作为自编码，不实现分类情况下，将所有训练好的权重和偏置拉成一个列向量，记为向量$$w\in R^{n}，n$$为网络中所有的参数个数之和。在对权调整的过程中，是将$$w$$作为优化问题的初始值，通过优化问题迭代求解$$w$$，优化目标函数为：

$$\min -\frac{1}{N}\underset{i}{\sum}\underset{j}{\sum}(V.*\log(V'))+(1-V).*(1-\log(V'))_{ij}$$

####其中$$X,X',N$$分别表示输入训练样本，输入重构样本，和样本个数，$$.*$$表示对应元素相乘。

####&nbsp;&nbsp;&nbsp;&nbsp;在hiton(2006)的文章中称上述目标函数为交叉熵误差，作为fine-tuning的误差函数，即:

$$\min -\underset{i}{\sum}V_{i}\log(V'_{i})-\underset{i}{\sum}(1-V_{i})\log(1-V'_{i})$$

####&nbsp;&nbsp;&nbsp;&nbsp;这一目标函数是利用共轭梯度法(<a href="http://blog.sciencenet.cn/blog-54276-569356.html">共轭梯度法</a>)求解下降方向，用线性搜索法求解每次下降的迭代步长。

####如果对应输出元素用于分类的情况下，在输入层后面添加一层和类别个数相同和节点的输出层，权值首先随机初始化$$W_{class}$$。在fine-tuning的过程中，首先调整最后一层的权值。假设输出的分类结果为$$C$$，真实的类别结果为$$C'$$，则调整最后一层权值的目标函数为：

$$\min -\sum_{i}C_{i}-C'_{i}$$

####方法依然是共轭梯度法。再将调整好的了最后一层的权值和前面RBM学习好的权值联合成一个列向量，用上面作为自编码的相同的方法进行调整权值。

####**权值可视化结果**

####在手写体图片上，图片大小为28x28，训练图像为10000张，网络结果含有三个隐层~，节点分别为1000,500,250,权值可视化结果如下：
<div style="text-align: center">
<img src="../images/layer1_diff.jpg">
</div>
<center><h4>第1个隐层权值可视化</h4></center>

<div style="text-align: center">
<img src="../images/layer2-diff.jpg">
</div>
<center><h4>第2个隐层权值可视化</h4></center>

<div style="text-align: center">
<img src="../images/layer3-diff.jpg">
</div>
<center><h4>第3个隐层权值可视化</h4></center>

<!--####在非全脸(不带头发和额头)女性人脸图像上，图像大小为48x48，训练图像为600张，网络结果为含有四个隐层，节点分别为1000，500,250,30，迭代10次，各层的可视化权值为：
<div style="text-align: center">
<img src="../images/layer1-600.jpg" style="width:1000; height=1000px;"/>
</div>
<center><h4>第1个隐层权值可视化</h4></center>
<div style="text-align: center">
<img src="../images/layer2-600.jpg">
</div>
<center><h4>第2个隐层权值可视化</h4></center>
<div style="text-align: center">
<img src="../images/layer3-600.jpg">
</div>
<center><h4>第3个隐层权值可视化</h4></center>
<div style="text-align: center">
<img src="../images/layer4-600.jpg">
</div>
<center><h4>第4个隐层权值可视化</h4></center>-->

####<font color='red'在图像为48x48的全脸女性(带头发和额头)人脸图像上</font>，训练样本为10000，网络结构同上，依然是四个隐层，分别为1000，500,250,30，迭代200次，可视化权值结果为：
<div style="text-align: center">
<img src="../images/layer1-10000-200.jpg" style="width:1000; height=1000px;"/>
</div>
<center><h4>第1个隐层权值可视化</h4></center>
<div style="text-align: center">
<img src="../images/layer2-10000-200.jpg">
</div>
<center><h4>第2个隐层权值可视化</h4></center>
<div style="text-align: center">
<img src="../images/layer3-10000-200.jpg">
</div>
<center><h4>第3个隐层权值可视化</h4></center>
<div style="text-align: center">
<img src="../images/layer4-10000-200.jpg">
</div>
<center><h4>第4个隐层权值可视化</h4></center>

####<font color='red'在图像为48x48的非全脸(不带头发和额头)女性图像上</font>，训练样本为10000，网络结构依然为四个隐层，分别为1000，500,250,30，迭代200次，可视化权值为：
<div style="text-align: center">
<img src="../images/female/layer1-10000-female.jpg" style="width:1000; height=1000px;"/>
</div>
<center><h4>第1个隐层权值可视化</h4></center>
<div style="text-align: center">
<img src="../images/female/layer2-10000-female.jpg">
</div>
<center><h4>第2个隐层权值可视化</h4></center>
<div style="text-align: center">
<img src="../images/female/layer3-10000-female.jpg">
</div>
<center><h4>第3个隐层权值可视化</h4></center>
<div style="text-align: center">
<img src="../images/female/layer4-10000-female.jpg">
</div>
<center><h4>第4个隐层权值可视化</h4></center>

####<font color='red'>在图像为48x48的非全脸(不带头发和额头)男性图像上</font>，训练样本为10000，网络结构依然为四个隐层，分别为1000，500,250,30，迭代200次，可视化权值为：
<div style="text-align: center">
<img src="../images/male/layer1-10000-male.jpg" style="width:1000; height=1000px;"/>
</div>
<center><h4>第1个隐层权值可视化</h4></center>
<div style="text-align: center">
<img src="../images/male/layer2-10000-male.jpg">
</div>
<center><h4>第2个隐层权值可视化</h4></center>
<div style="text-align: center">
<img src="../images/male/layer3-10000-male.jpg">
</div>
<center><h4>第3个隐层权值可视化</h4></center>
<div style="text-align: center">
<img src="../images/male/layer4-10000-male.jpg">
</div>
<center><h4>第4个隐层权值可视化</h4></center>

####我的感觉是，相同网络结构和参数下，男性第四层提到的特征脸部情况要比女性清晰~我个人的猜测是女性第四层五官学习的不清析，可能是由于女性头发的干扰~


####**Reference**

####[1] Deep Learning methods and applications, Li Deng and Dong Yu, 2014.

####[2] Sparse deep belief net model for visual area V2, Honglak Lee, Andrew Y.Ng, 2008.

####[3] Reducing the dimensionality of data with neural networks, Hinton, 2006.

####[4] Training products of experts by minimizing contrastive divergence, Hinton, 2002.
