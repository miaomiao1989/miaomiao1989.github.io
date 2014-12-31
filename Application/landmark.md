---
layout: default
---
#**形状回归人脸特征点检测**

####&emsp;&emsp;脸关键点定位是在人脸检测基础上，进一步定位人脸的眼睛眉毛鼻子嘴巴轮廓等。 主要思想就利用关键点附近的信息以及各个关键点之间的相互关系来定位。方法大致分为两类，一个是基于模型的方法，一个是基于回归的方法。这里主要介绍的第二种方法。

####&emsp;&emsp;基于模型的方法以cootes在1995年提出的asm方法(参见文献[1])最为代表，asm方法将数十个脸部特征点的纹理和位置关系约束一起考虑来进行计算出一个参数模型。从局部特征中检测到所求的关键点，但是这种方法对噪声非常敏感。

####&emsp;&emsp;基于回归的方法是ASM相关改进的另外一个方向，就是对形状模型本身的改进。这里主要介绍的是文献[2]中的回归方法(Face alignment by explicit shape regression, ESR). 文献[4，5]也都是在这一方法的基础的进行的改进。

####&emsp;&emsp;ESR 使用的是一个cascade regression的框架，是P. D ollár在CVPR2010的Cascaded Pose Regression（参见文献[4]）中提出的一种方法，用来解决对齐问题。ESR方法在此基础上做了几个扩展，使更为适合做人脸关键点定位。

####**Face alignment by explicit shape regression, ESR**

####**1. Two-level boosted regression**

####&emsp;&emsp;ESR(文献[2])使用了一个两级的boosted regressor。作者是使用了第一级10级，第二层500级。在这个二级结构中，第一级中每个节点都是500个弱分类器的级联，也就是一个第二层的regressor(文献[3](rcpr)中使用的是100个第一层，50个第二层regressor)。在第二层regressor中，特征是保持不变的，而在第一层中，特征是变化的。所以，这事实上是一个两层的结构, 并不是一层的结构。在第一层，每一个节点的输出都是上一个节点的输入。都是在上一级估计的关键点上在取的特征。如下图

<div style="text-align: center">
<img src="../Images/mark1.jpg">
</div>

####&emsp;&emsp;其中$$S^{0}$$为初始化形状，$$R^{t}$$为第$$t$$个第一层回归器，$$S^{t}$$为经过第$$t$$个第一层回归器后得到的形状。

####&emsp;&emsp;注：文献[2，3]中的ESR,RCPR第一层回归器都使用的是最小二乘，即预测值和真实值的误差，用来和设定阈值进行比较，判断是否进行下次回归。

####&emsp;&emsp;**第二层回归器**ESR用的是Fern(一种二叉树)(注：Fern和普通二叉树不同的是，Fern不用存储根节点，只存储叶子节点，即对于输入的每个样本最终都要到达叶子节点中的一个，不在中间根节点停止。而普通二叉树需要用样本来构建根节点)。Fern是N个特征和阈值的组合，将训练样本划分为2的F次幂个bins。每一个bin对应一个输出，

<div style="text-align: center">
<img src="../Images/mark2.jpg">
</div>

####&emsp;&emsp;这里通俗的理解就是能够到达该bin的所有样本的平均值，作为当前bin的输出。

####**2. Feature**

####&emsp;&emsp;前面在第一节中提到过，在两层回归器的应用的特征是不同的。第一层中应用的是shape indexd feature。CPR (文献[4])中提出了 pose indexd feature，这种特征保持了对形状的不变性，从而增加了算法的鲁棒性。在ESR(文献[2])方法中，将这种feature变成了shape indexd feature，所谓的shape index feature，就是根据关键点的位置和一个偏移量，取得该位置的像素值，然后计算两个这样的像素的差值，从而得到了形状索引特征。该方法中采用的是局部坐标而非全局坐标系，极大的增强了特征的鲁棒性。如下图所示：(具体选取方法在后面会详细讲)。

<div style="text-align: center">
<img src="../Images/mark3.jpg">
</div>

####&emsp;&emsp;其中第一行为相对于整幅图像的坐标，第二行为相对于单个特征点的特征选取的坐标，可以看到相对特征点来说对于整个人脸的移动，偏置等比相对于整幅图像要更鲁棒。

####&emsp;&emsp;第二层训练Fern利用的是Correlation-based feature。即从第一层中的shape index 中选择其中相关性最强的几个特征(rcpr(文献[3])中选择的是5个)。假设说我们需要训练50个Fern，那么我们从第一层的shape index feature中选择50组相关性最强个5个特征，每5个特征用来训练一个Fern。这里这5个特征的选择方法，文献[2]中称为“Correlation-based feature selection”，选择方法简单来说是将目标$$y$$投影到随机向量上得到$$y_{proj}$$，然后计算$$y_{proj}$$和每个特征之间的相关系数，选择相关系数最大的前k个构成相关性特征去学习Fern.

####&emsp;&emsp;关于Fern的选择是，对每个样本都利用上述方法提取5个相关性特征，然后随机取5个阈值，利用每个相关性特征和阈值的大小比较，使得每个样本都能达到Fern的叶子节点(Fern是二叉树的一种，但只保存叶子节点，假设Fern的深度为5，则应该有叶子节点32个，这里利用阈值使得每个样本都达到32个叶子节点中的一个)， 然后这一Fern的32个叶子节点的输出为所有到达该叶子节点的训练样本的特征点形状的平均值。以此类推，可以训练到其他的Fern。

####**3. 具体实施过程**

####**训练过程**
* <h4> 初始化训练样本的形状： 这里首先要对形状进行归一化，因为我们现有的ground truth坐标都是相对于整幅图像而言的，在计算的过程中比较复杂，这里首先要将这些ground truth相对于人脸框中心和人脸框宽度进行归一化，这种处理也是为了能够对人脸框大小，图像大小比较鲁棒。然后对人脸框位置添加随机扰动，ground truth相对于扰动后的人脸的归一化信息作为训练样本的初始形状。为了提高泛化能力，文献[2]中使用多个初始化(实验中是20个)初始化形状。</h4>
* <h4><b>For t=1 : T</b>(第一层回归器个数回归器个数，实验中为100)<br>
  - **计算目标误差:**<br> 计算第t次第一层回归器预测结果和ground truth的误差，作为目标值输入第二层回归器。<br>
  - **计算shape index feature:** <br>
  &emsp;&emsp;文献[2]中提出的是利用与训练集上的<font color='red'>平均形状</font>的特征点最近的点，即：假如需要选择400个shape index feature，随机产生400个2维随机数，作为相对于人脸框大小归一化后的与人脸框中心点的相对坐标$$x,y$$，用这400个坐标点与平均形状的每个特征点计算距离，取距离最小的特征点和距离差，作为这400个shape index feature的坐标索引， 用这一索引去取<font color='red'>训练集图像</font>原始图像上对应位置的像素值作为训练样本的shape index featrue。<br>
  &emsp;&emsp;文献[3]在上面基础上提出了利用和训练集平均形状两个特征点相关的点作为shape index featrue。即：假设需要400个shape index feature，首先需要随机产生400对特征点组合，和400个随机数，用每对特征点的坐标可以计算一条直线，用随机数作为x轴坐标，在这条直线上取点，取到的点作为索引，在训练集原始图像上的取相应的灰度值作为shape index feature。这种方法可以用两个特征点约束shape index feature，而且不需要利用训练集上的平均形状。<br>
  - **训练回归器Fern**<br>
  从上一步计算出了shape index feature， 这里可以利用这些特征进行Fern的训练。<br>
  **For k=1 : K**(第二层回归器Fern个数，实验中为50)<br>
  &emsp; <b>1.</b> 更新回归误差;<br>
  &emsp; <b>2.</b> 选择Correlation-based feature：(实验中选择5维，决定着Fern的深度)<br>
  &emsp;&emsp;计算方法如下：<br>
  ![](../Images/mark4.jpg)<br>
  &emsp;&emsp;其中$$Y_{proj}$$为回归目标在随机向量上的投影，$$\rho_{m}, \rho_{n}$$为shape index feature中的一个，$$cov$$为协方差计算。<br>
   &emsp; <b>3.</b> 训练Fern;<br>
   &emsp;随机产生Correlation-based feature个数相同个阈值，用来和Correlation-based feature特征进行比较，使得样本能够到达Fern的某个叶子节点。(这里阈值的个数和Fern的深度由Correlation-based feature的个数决定，例如实验中为5，则Fern的深度为5，叶子节点也称bin的个数为$$2^{5}$$个，相应的阈值个数也应该为5.)利用相应阈值和训练集每个样本相应的Correlation-based feature进行比较，直到最后一个，样本能够到达Fern的32个bin中的一个，待所有训练样本完成，当前Fern的每个bin的输出为所有到达该bin的训练集目标值的平均值，到此，一个Fern的训练完成；<br>
   &emsp;<b>4. </b>输出训练集的预测值。<br>
    **end**(第二层回归器Fern)<br>
   - **计算预测值与真实值误差；**<br>
* <h4><b>end</b>(第一层回归器)</h4>

####**测试过程**
* 初始化：文献[2][3]中都是用的多个初始化(实验中为5)，即测试图像的人脸框进行随机扰动，在训练集中随机抽取5个样本的形状，分别对测试图像的人脸框进行归一化，将归一化后的形状作为测试图像的初始化形状。这5个初始化形状是同时进行计算的(备忘[matlab下18个点5个初始化灰度图测试时间为0.04s])，最后的预测结果是多个预测形状的平均值。
* <h4><b>For t=1 : T</b>(第一层回归器)</h4>
  - 利用训练好的模型的特征索引，提取shape index feature;
  - <b>For k=1 : K</b>（第二层回归器）<br>
  &emsp; <b>1.</b> 利用训练模型里的特征索引，提取Correlation-based feature；<br>
  &emsp; <b>2.</b>利用Correlation-based feature和训练模型里的阈值进行比较，使测试样本到达Fern的32个bin中的一个，到达的bin中值作为当前$$k$$步的测试样本的输出；<br>
  &emsp; **3.** 更新目标值。
  - <b>end</b>
* <b>end</b>
* 若是有多个测试样本的初始化，判断预测结果和平均预测结果的误差，若误差大于阈值，则要以当前预测结果为初始值重新进行测试。
<!--（会出现这种情况，通常是人脸框的位置不太合适，当人脸框较大，基本能包含需要检测的所有特征点是，一般不会执行二次测试。在实验过程中利用了两个人脸框检测进行测试，一种是matlab检测，框比较小，大致是嘴唇下一点，不包含下巴，眉毛上一点，在这种情况下测试过程中是会有部分测试样本会进行二次检测。第二种是利用特征点来估计人脸框，使得人脸框能够包含所有的要检测的特征点，大致是下巴，眉毛上一点，左右耳朵，在这种情况下测试图像是不执行二次测试的)。-->

####**Robust face landmark estimation under occlusion， rcpr**

####文献[3]中rcpr在上述方法的基础之上，提出了遮挡信息，即在样本的特征点形状向量后面还有一维信息为是否遮挡，不遮挡为0，遮挡为1。文章中提出的这一遮挡信息有两个用处，一个是在测试过程中用来推测测试样本中的每个特征点是否是遮挡的，另一个作用是在训练过程中用遮挡信息作为权值，在每个第二层回归器中，用同样的回归目标训练3个Fern，用这3个Fern的加权平均值作为该层第二层回归器的输出。

####在上一节中讲到训练过程中外层回归器用的是100个，内层回归器是50个。而在含有遮挡信息时，内层回归器是15*3个， 15是内层回归器的个数，3是每层内层回归器有平行的三个Fern，用加权平均作为每个15层内层回归器的输出值。
![](../Images/mark5.jpg)

####&emsp;&emsp;上图(a)是遮挡信息的示例，在训练过程中，将提取到每个样本的shape index feature的每一个归一化到[0,1]之间，然后标记每个feature落在上述图(b)中的哪个group，同理将预测到的训练样本的特征点也相同方法处理，记录落在哪个group中。然后计算每个训练样本预测的的特征点落在的gounp中和它相关的shape index feature的个数，将这个个数归一化后作为此特征点预测结果的权值(比如假设共有4个特征点，分别落在上图(b)的1,2,3,4group内，和他们相关的shape index featrue落在相应的1,2,3,4group中的个数为10,20，30,40个，则权值为1-(10/100, 20/100, 30/100, 40/100))。

####&emsp;&emsp;在测试过程中，在经过训练好的模型得到预测形状的同时也会得到遮挡信息，将这一遮挡信息和阈值进行比较，判断为0还是为1，即是遮挡还是未遮挡

####**Face Alignment at 3000 FPS via Regressing, 3000FPS**

####&emsp;&emsp;上面讲到的两种方法(参考文献[2-3])在回归的过程中都是将整个shape同时进行两层回归器回归的，而文献[4]在上面方法的基础上，提出了先local后global的方法，即先对单个特征点用回归森林进行回归，然后将local回归结果用global结合起来。

 


####**Refernce**

####[1]Cootes T F, Taylor C J, Cooper D H, et al. **Active shape models-their training and application**[J]. Computer vision and image understanding, 1995, 61(1): 38-59.

####[2] Cao X, Wei Y, Wen F, et al. **Face alignment by explicit shape regression**[J]. International Journal of Computer Vision, 2014, 107(2): 177-190.

####[3] Burgos-Artizzu X P, Perona P, Dollár P. **Robust face landmark estimation under occlusion**[C]//Computer Vision (ICCV), 2013 IEEE International Conference on. IEEE, 2013: 1513-1520.

####[4] Dollár P, Welinder P, Perona P. **Cascaded pose regression**[C]//Computer Vision and Pattern Recognition (CVPR), 2010 IEEE Conference on. IEEE, 2010: 1078-1085.

####[5] Ren S, Cao X, Wei Y, et al. **Face Alignment at 3000 FPS via Regressing Local Binary Features**[J].