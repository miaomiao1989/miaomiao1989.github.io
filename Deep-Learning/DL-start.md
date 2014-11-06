---
layout: post
---
<table align="left">
    <h4 style="text-indent: 2em;"><a href= "../index.html">Deep-Learning</a></h4>
</table>

#**深度学习开篇**

####机器学习的两大热潮分别为浅层学习(Shallow Learning) 和深度学习(Deep Learning). 
  
####浅层学习： 
  
<h4>&nbsp;&nbsp;&nbsp;&nbsp;20世纪80年代末期，用 于人工神经网络的反向传播算法（也叫Back Propagation算法或者BP算法）的发明，给机器学习带来了希望，掀起了基于统计模型的机器学习热潮。这个热潮一直持续到今天。人们发现，利用 BP算法可以让一个人工神经网络模型从大量训练样本中学习统计规律，从而对未知事件做预测。这种基于统计的机器学习方法比起过去基于人工规则的系统，在很多方面显出优越性。这个时候的人工神经网络，虽也被称作多层感知机（Multi-layer Perceptron），但实际是种只含有一层隐层节点的浅层模型。
20世纪90年代，各种各样的浅层机器学习模型相继 被提出，例如支撑向量机（SVM，Support Vector Machines）、 Boosting、最大熵方法（如LR，Logistic Regression）、高斯混合模型(GMM)、条件随机场(CRF)等。这些模型的结构基本上可以看成带有一层隐层节点（如SVM、 Boosting），或没有隐层节点（如LR）。这些模型无论是在理论分析还是应用中都获得了巨大的成功。相比之下，由于理论分析的难度大，训练方法又需 要很多经验和技巧，这个时期浅层人工神经网络反而相对沉寂。</h4> 
   
####深度学习：   
<h4>&nbsp;&nbsp;&nbsp;&nbsp;2006年，加拿大多伦多大学教授、机器学习领域的 泰斗Geoffrey Hinton和他的学生RuslanSalakhutdinov在《科学》上发表了一篇文章[1]，开启了深度学习在学术界和工业界的浪潮。这篇文章有两 个主要观点：1）多隐层的人工神经网络具有优异的特征学习能力，学习得到的特征对数据有更本质的刻画，从而有利于可视化或分类；2）深度神经网络在训练上 的难度，可以通过“逐层初始化”（layer-wise pre-training）来有效克服，在这篇文章中，逐层初始化是通过无监督学习实现的。</h4>

<h4>&nbsp;&nbsp;&nbsp;&nbsp;深度学习的实质，是通过构建具有很多隐层的机器学习模型和海量的训练数据，来学习更有用的特征，从而最终提升分类 或预测的准确性。因此，“深度模型”是手段，“特征学习”是目的。区别于传统的浅层学习，深度学习的不同在于：1）强调了模型结构的深度，通常有5层、6 层，甚至10多层的隐层节点；2）明确突出了特征学习的重要性，也就是说，通过逐层特征变换，将样本在原空间的特征表示变换到一个新特征空间，从而使分类 或预测更加容易。与人工规则构造特征的方法相比，利用大数据来学习特征，更能够刻画数据的丰富内在信息。</h4>

<h4>&nbsp;&nbsp;&nbsp;&nbsp;从2006年以来，深度学习受到了越来越多的关注， 真正震撼大家的可能是两件事。一个是在2012年的ImageNet画像识别大赛上，Geoffrey Hinton [2]带领学生利用Deep Learning取得了极好的成绩(2012, 85%；2011, 74%；2010, 72%)。另外一件事是Microsoft通过与Geoffrey Hinton合作，利用Deep Learning在语音识别系统中取得了巨大的成功[3]。</h4>

<h4>&nbsp;&nbsp;&nbsp;&nbsp;Deep learning与传统的神经网络之间有相同的地方也有很多不同。二者的相同在于deep learning采用了神经网络相似的分层结构，系统由包括输入层、隐层（多层）、输出层组成的多层网络，只有相邻层节点之间有连接，同一层以及跨层节点 之间相互无连接，每一层可以看作是一个logistic regression模型； 这种分层结构，是比较接近人类大脑的结构的。</h4>   
![Alt text](../images/DL-start.jpg) 

####<center>图1 单隐层神经网络模型和深度学习模型</center>
<h4>&nbsp;&nbsp;&nbsp;&nbsp;而为了克服神经网络训练中的问题，DL采用了与神经网络很不同的训练机制。传统神经网络中，采用的是back propagation的方式进行，简单来讲就是采用迭代的算法来训练整个网络，随机设定初值，计算当前网络的输出，然后根据当前输出和label之间的 差去改变前面各层的参数，直到收敛（整体是一个梯度下降法）。而deep learning整体上是一个layer-wise的训练机制。这样做的原因是因为，如果采用back propagation的机制，对于一个deep network（7层以上），残差传播到最前面的层已经变得太小，出现所谓的gradient diffusion (梯度扩散)。

&nbsp;&nbsp;&nbsp;&nbsp;2006年，hinton[1][4]提出了在非监督数据上建立多层神经网络的一个有效方法，简单的说，分为两步，一是每次训练一层网络，二是调优，使原始表示x向上生成的高级表示r和该高级表示r向下生成的x'尽可能一致。方法是：首先逐层构建单层神经元，这样每次都是训练一个单层网络。当所有层训练完后，Hinton使用wake-sleep算法进行调优。</h4>  
 
####具体训练过程为：

<h4>&nbsp;&nbsp;&nbsp;&nbsp;1）使用自下上升非监督学习（就是从底层开始，一层一层的往顶层训练）：

&nbsp;&nbsp;&nbsp;&nbsp;采用无标定数据（有标定数据也可）分层训练各层参数，这一步可以看作是一个无监督训练过程，是和传统神经网络区别最大的部分（这个过程可以看作是feature learning过程）：具体的，先用无标定数据训练第一层，训练时先学习第一层的参数（这一层可以看作是得到一个使得输出和输入差别最小的三层神经网络的隐层），由于模型 capacity的限制以及稀疏性约束，使得得到的模型能够学习到数据本身的结构，从而得到比输入更具有表示能力的特征；在学习得到第n-1层后，将n- 1层的输出作为第n层的输入，训练第n层，由此分别得到各层的参数；

&nbsp;&nbsp;&nbsp;&nbsp;2）自顶向下的监督学习（就是通过带标签的数据去训练，误差自顶向下传输，对网络进行微调）：

&nbsp;&nbsp;&nbsp;&nbsp;基于第一步得到的各层参数进一步fine-tune整个多层模型的参数，这一步是一个有监督训练过程；第一步类似神经网络的随机初始化初值过程，由于DL 的第一步不是随机初始化，而是通过学习输入数据的结构得到的，因而这个初值更接近全局最优，从而能够取得更好的效果；所以deep learning效果好很大程度上归功于第一步的feature learning过程。</h4> 
  
###Reference  

####[1] Hinton G E, Salakhutdinov R R. Reducing the dimensionality of data with neural networks[J]. Science, 2006, 313(5786): 504-507.

####[2] Krizhevsky A, Sutskever I, Hinton G E. Imagenet classification with deep convolutional neural networks[C]//Advances in neural information processing systems. 2012: 1097-1105.  

####[3] Hinton G, Deng L, Yu D, et al. Deep neural networks for acoustic modeling in speech recognition: The shared views of four research groups[J]. Signal Processing Magazine, IEEE, 2012, 29(6): 82-97.

####[4] Hinton G, Osindero S, Teh Y W. A fast learning algorithm for deep belief nets[J]. Neural computation, 2006, 18(7): 1527-1554.