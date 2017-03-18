---
layout: default
---

# **MEEM**

#### &emsp;这里讨论的的是文献[1]中的MEEM跟踪算法。这篇跟踪算法旨在解决的问题是：**在online-tracking问题中，若是目标出现遮挡，越出界面，跟错等问题时，实时更新的跟踪模板会随着错误的样本的更新模型的update会出现错误的更新，从而导致错误的积累越来越多，会出现跟踪框相对目标出现偏移状况(model shift)。**而防止这一现象的出现有两个方面的做法，一种是避免bad update，即每次更新模型之前需要检测审查此次更新是不是有效的(文献[2-3]),但是这类方法通常是基于目标运动和外形变化平滑的假设，在一些复杂的场景下审查会失效，还是无法避免错误的模型更新。另一种是在bad update发生之后，修正跟踪器。而这里讨论的MEEM方法就是这种思想。

#### &emsp;MEEM由两部分组成，一个是基础跟踪器，采用的是online svm二分类跟踪器，采用的通常的二分类跟踪，即在目标框周围以扫描框的形式来对样本进行二分类，在选出的正样本中选出置信度最高的正样本作为跟踪框的目标输出； 另一个是multi-experts framework用来修正bad update发生之后的跟踪器。

#### &emsp;multi-experts的主要思想是，将在第t帧之前的多个tracker保存为experts组成专家组，在第t帧评估tracker的时候，将之前的多个tracker和当前帧训练的tracker进行比较，比较方法为熵正规化，选出最优的tracker。若最优的tracker为当前帧的，则直接保留当前帧的tracker，若最优的tracker不是当前帧的，而是之前的某一个，则说明模型进行了bad upadte，这时候需要将tracker用之前的最优tracker替代(意思是将tracker退回到了没有bad update的时候)。这种思想的核心是tracker的评估，即熵正规化loss函数来评估，当前tracker相对于之前的tracker是不是进行了bad update。

#### &emsp;损失函数：在监督学习中，损失函数计算的是预测值和真实值标签label之间的误差。而在online tracking中是没有真实值标签的，属于是无监督学习，因此损失函数采用的是文献[4]中的熵正规化来作为损失函数。条件熵函数测量的是类之间的重合度。下图展示了作者使用上函数的思想。

<div style="text-align: center; height: 550px">
<img src="../Images/MEEM1.png">
</div>

#### &emsp;上面图中(a)显示了不同帧数的tracker的跟踪结果(绿色框),在目标经过了遮挡后在第#374帧tracker预测的绿色框出现了偏移。而下图(b)是保留了在#250,#300,#350,#374帧的tracker组成的experts。其中红色和绿色分别表示对第#374帧中的红色框部分和绿色框部分进行的置信度。在选择这四个tracker的时候，自然的选择的是第#250帧的tracker作为当前#374帧的tracker而舍弃当前帧的tracker。这是因为#374帧的tracker对绿色框错误位置的置信度要高于真实框红色的置信度，这是明显跟踪器更新出现了偏差。而第#350帧的tracker对于红色和绿色框的置信度分不开，这并不是一个好的tracker应该有的结果，也放弃。而#300帧和#250帧的结果相同，说明在这50帧中tracker的更新没有出现较大的变化，因此保留最靠前的#250帧的tracker替换当前#374帧的tracker。

- #### **选择合适的tracker(experts)**

#### &emsp;假设在第t时刻的跟踪器为$$S_{t}$$，则$$E=\{S_{t_{1}}, S_{t_{2}},\cdots\}$$组成expert ensumble专家组。在t时刻从E中选择最佳的expert作为当前t时刻的跟踪器。假设训练样本$$L=\{(x_{i}, z_{i})\}$$，其中$$x_{i}$$表示样本，而$$z_{i}$$表示样本集的可能的类标，其中包含有$$x_{i}$$的真实类标。[4]中采用的损失函数为最大化log posterior概率

<div style="text-align: center">
<img src="../Images/MEEM2.png">
</div>

#### 其中第一项为log似然估计，第二项为训练样本和可能类标下的类标条件熵函数。添加熵函数正则化的目的是为了选择区分性比较大的tracker，意思就是在其他条件相同条件下比较两个tracker选择的时候，优先选择不同类别间置信度区分较大的tracker，说明tracker的区分判别能力比较强。上式可以转化为如下的最小化问题：

<div style="text-align: center">
<img src="../Images/MEEM3.png">
</div>

#### &emsp;似然估计可以定义为：

<div style="text-align: center">
<img src="../Images/MEEM4.png">
</div>

#### &emsp;熵函数可以定义为：

<div style="text-align: center">
<img src="../Images/MEEM5.png">
</div>


#### **Reference**

#### [1] MEEM: Robust tracking via multiple experts using entropy minimization, ECCV2014.

#### [2] Robust object tracking via sparsity-based collaborative model, CVPR2014.

#### [3] Mininum error bounded efficient l1 tracker with occlusion detection, CVPR2011.

#### [4] Semi-supervised learning by entropy minimization, NIPS2005.
