---
layout: default
---

# **FCT**

#### &nbsp;&nbsp;&nbsp;&nbsp;这里主要讨论的是文献[1]compressive tracking(CT)的优化算法[2] Fast comressive tracking(FCT).

#### &nbsp;&nbsp;&nbsp;&nbsp;CT是一种简单高效地基于压缩感知的跟踪算法。首先利用符合压缩感知RIP条件的随机感知矩对多尺度图像特征进行降维，然后在降维后的特征上采用简单的朴素贝叶斯分类器进行分类。该跟踪算法非常简单，但是实验结果很鲁棒，速度大概能到达40帧/秒。和一般的模式分类架构一样：先提取图像的特征，再通过分类器对其分类，不同在于这里特征提取采用压缩感知，分类器采用朴素贝叶斯。然后每帧通过在线学习更新分类器。

- ### **压缩感知**

#### &nbsp;&nbsp;&nbsp;&nbsp;首先我们来简单了解一下压缩感知理论。<font color = "red">简单地说，压缩感知理论指出：只要信号是可压缩的或在某个变换域是稀疏的，那么就可以用一个与变换基不相关的观测矩阵将变换所得高维信号投影到一个低维空间上，然后通过求解一个优化问题就可以从这些少量的投影中以高概率重构出原信号，可以证明这样的投影包含了重构信号的足够信息。</font>在该理论框架下，采样速率不再取决于信号的带宽，而在很大程度上取决于两个基本准则：<font color = "red">稀疏性和非相关性，或者稀疏性和等距约束性</font>。

#### &nbsp;&nbsp;&nbsp;&nbsp;通俗些讲，压缩感知的意思就是，如果$$N$$维高维信号$$x$$在某个变换域是稀疏冗余的，也就是可以找到一组正交基$$\Psi$$，使得信号$$x$$在$$\Psi$$是k-稀疏的，即可以表示为：

$$
x = \Psi \alpha
$$ 

$$\alpha$$是k-稀疏向量(非零元素个数小于k)

如果能找到一个与正交基不相干的观测矩阵(测量矩阵)$$\Phi$$,能将高维信号$$x$$映射到低维$$M(M<<N)$$维观测信号$$y$$,

$$
\begin{array}{cc}
y & = & \Phi x\\
&=& \Phi \Psi \alpha
\end{array}

$$

则原始信号$$x$$即可以用低维观测信号$$y$$以少量的测量完美重构($$y$$中包含了重构$$x$$的足够的信息)：

$$\hat{x} = \Psi (\Phi \Psi)^{-1}y$$

#### &nbsp;&nbsp;&nbsp;&nbsp;自然界存在的真实信号一般不是绝对稀疏的，而是在某个变换域下近似稀疏，即为可压缩信号。或者说从理论上讲任何信号都具有可压缩性，只要能找到其相应的稀疏表示空间，就可以有效地进行压缩采样。信号的稀疏性或可压缩性是压缩感知的重要前提和理论基础。<font color = "red">稀疏表示的意义：只有信号是K稀疏的（且K<M<<N），才有可能在观测M个观测值时，从K个较大的系数重建原始长度为N的信号。也就是当信号有稀疏展开时，可以丢掉小系数而不会失真。</font>

#### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**观测矩阵**

#### &nbsp;&nbsp;&nbsp;&nbsp;观测矩阵(也称测量矩阵)$$\Phi\in R^{M\times N}(M<<N)$$是用来对$$N$$维的原信号进行观测得到$$M$$维的观测向量$$Y$$，然后可以利用最优化方法从观测值$$Y$$中高概率重构$$X$$。也就是说原信号$$X$$投影到这个观测矩阵(观测基)上得到新的信号表示$$Y$$。

#### &nbsp;&nbsp;&nbsp;&nbsp;为了保证能够从观测值准确重构信号，其需要满足一定的限制：观测基矩阵与稀疏基矩阵的乘积满足RIP性质(Restricted isometry property有限等距性质)。这个性质保证了观测矩阵不会把两个不同的K稀疏信号映射到同一个集合中（保证原空间到稀疏空间的一一映射关系），这就要求从观测矩阵中抽取的每$$M$$个列向量构成的矩阵是非奇异的。<font color = "red">这条测量矩阵的性质是下面压缩跟踪的基础</font>.



- ### **压缩跟踪 compressive tracking**

#### &nbsp;&nbsp;&nbsp;&nbsp;和一般的检测跟踪方法相同，压缩跟踪也是在训练过程中在目标框周围以窗口滑动的方式进行采样，分成正负样本，对所有正负样本进行特征提取，这里利用的是压缩特征，对正负样本的特征用朴素贝叶斯训练分类器。在跟踪过程中，在上一帧的目标位置周围窗口滑动采样样本，提取压缩特征，并输入到训练好的分类器中进行分类，取相似度最高的样本位置作为跟踪的结果。然后重复在目标框周围采样正负样本，重新提取压缩特征，并更新分类器。

#### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **1. 压缩特征提取**

#### &nbsp;&nbsp;&nbsp;&nbsp;这里使用的是上面讲述的压缩感知用观测矩阵将高维信号$$x$$投影到低维信号$$y$$：

$$y = \Phi x$$

#### 这里的观测矩阵$$\Phi$$需要满足RIP条件，作者在论文中选用的是稀疏随机高斯矩阵作为观测矩阵$$\Phi = R, r_{ij}~N(0,1)$$

$$
r_{ij} = \sqrt{s}\times
\Bigg\{
\begin{array}{cc}
&1&\text{ with probability }\  \frac{1}{2s}\\
&0&\text{ with probability }\  1-\frac{1}{s}\\
&-1&\text{ with probatility }\  \frac{1}{2s}
\end{array}
$$

#### &nbsp;&nbsp;&nbsp;&nbsp;作者将在目标框附近采样得到的正负样本看作高维信号$$x$$(若目标框大小为50*50，则维数$$x$$为2500维)，用稀疏随机测量矩阵$$\Phi = R$$映射成低维向量$$y$$(文章中维数为100维)，<font color = "red">从而可以大大降低维数的同时又保留了原始图像样本信号的主要信息。</font>
   

#### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **2. 朴素贝叶斯分类器**

#### &nbsp;&nbsp;&nbsp;&nbsp;对于任意的样本$$x \in R^{n}$$, 其映射到低维表示为$$v=(v_{1}, v_{2}, \cdots, v_{m})^{T}\in R^{m},m<<n$$。假设向量$$v$$中的每个元素是独立分布的，则用朴素贝叶斯(naive Bayes classifier)分类器进行分类:

$$
\begin{array}{cc}
H(v) &=& \log\left(\frac{\Pi^{n}_{i=1}p(v_{i}|y=1)p(y=1)}{\Pi^{n}_{i=1}p(v_{i}|y=0)p(y=0)}\right)\\
&=&\sum^{n}_{i=1}\log\left(\frac{p(v_{i}|y=1)}{p(v_{i}|y=0)} \right)
\end{array}
$$

其中$$y\in\{0,1\}$$为二值样本类标，$$p(y=1)=p(y=0)$$是均匀先验(uniform prior)。选择$$H(v)$$中最大的采样样本的位置作为检查的结果目标框的位置。Diaconis在文献[3]中指出当高维向量随机投影到低维向量时也是高斯分布的。因此，在分类器$$H(v)$$中的条件概率也假设为是满足高斯分布的：

$$
p(v_{i}|y=1) \backsim N（\mu_{i}^{1}, \sigma_{i}^{1})\\
p(v_{i}|y=0) \backsim N(\mu_{i}^{0}, \sigma_{i}^{0})
$$

然后更新$$\mu_{i}^{1}, \mu_{i}^{0}, \sigma_{i}^{1}, \sigma_{i}^{0}$$:

$$
\mu_{i}^{1} \leftarrow \lambda\mu_{i}^{1} + (1-\lambda)\mu^{1}\\
\sigma^{1}_{i} \leftarrow \sqrt{\lambda(\sigma^{1}_{i})+(1-\lambda)(\sigma^{1})^{2}+\lambda(1-\lambda)(\mu^{1}_{i}-\mu^{1}_{i}-\mu^{1})^{2}}
$$

其中有：

$$
\mu^{1} = \frac{1}{n}\sum^{n-1}_{k=0|y=1}v_{i}(k)\\
\sigma^{1} = \sqrt(\frac{1}{n}\sum^{n-1}_{k=0|y=1}(v_{i}(k)-\mu^{1})^{2})
$$

$$\mu^{1}, \sigma^{1}$$为当前帧正样本的均值和方差，$$\mu^{0}, \sigma^{0}$$为当前帧负样本的均值和方差。
                                                                                                                                            

#### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **3. 具体实施过程**

#### &nbsp;&nbsp;&nbsp;&nbsp;具体算法的流程图如下图所示：

<div style="text-align: center; height: 500px">
<img src="../Images/FCT1.png">
</div>

#### &nbsp;&nbsp;&nbsp;&nbsp;上图中的第一行为训练过程，在第$$t$$，假设我们已经知道了目标的精确位置，在目标周围分别采样得到正负样本，分别对正负样本进行特征提取(可以是灰度值，也可以是其他高维特征。这里还加入了多尺度样本，即对一张样本进行不同尺度采样，将多个尺度上的特征联合在一起，形成改样本的特征)，然后将高维特征映射到低维特征，形成压缩特征。用正负样本低维压缩特征进行分类器训练。

#### &nbsp;&nbsp;&nbsp;&nbsp;上图上的第二行为检测跟踪过程，在第$$t+1$$帧，在上一帧检测结果的目标框周围进行采样，将采样得到的样本用分类器进行分类，选出最相似的样本的位置作为检测的目标框的位置。


#### **Reference**

[1] Kaihua Zhang, **Real-time compressive tracking**, CVPR 2012.

[2] Kaihua Zhang, **Fast compressive tracking**, TPAMI 2014.

[3] Diaconis P.**Asymptotices of graphical projection pursuit**, 1984.
