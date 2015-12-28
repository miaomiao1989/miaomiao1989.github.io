---
layout: default
---

#**CMT**

####&nbsp;&nbsp;&nbsp;&nbsp;这里主要讨论的是文件[1],[2]提出的CMT特征点跟踪算法。基本思想是根据特征点来跟踪目标，包括特征点跟踪和检测两个部分，并将两部分的结果进行和并，达到跟踪结果和检测结果相互修正的目的。主要过程为：

- <h4><b>初始化</b></h4>

####&nbsp;&nbsp;&nbsp;&nbsp;和TLD的思想一样，初始化的过程主要是初始化检测器，对第一帧图像进行如下处理：
  
####&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1) 首先用BRISK算法来检查全图中的特征点；

####&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2) 根据给定的目标框选取在框内的特征点为目标特征点，剩余的特征点为背景特征点；

####&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3) 计算所有目标特征点两两之间的距离和角度；


- <h4><b>跟踪特征点</b></h4>

####&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1）用光流算法跟踪上一帧的目标特征点；

####&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2) 计算当前帧跟踪的特征点和上一帧的特征点的之间的像素位移差，删除掉位移较大的异常点，剩余的特征点作为跟踪得到的特征点；

####&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3) 利用跟踪得到的特征点估计跟踪目标的尺度和旋转角度(这里计算尺度和旋转角度都是以第一帧为基准的)：

#### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;计算尺度为： 用当前跟踪得到的特征点两两之间的距离和相应的初始帧的相应特征点之间的距离比值的平均值作为估计得到的当前帧的尺度。

#### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;计算旋转角度为： 用当前跟踪得到的特征点两两之间的角度减去初始帧相应特征点之间的角度得到的平均值作为估计得到的当前帧的旋转角度。

####&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4) 计算跟踪特征点到中心点的voting： 

$$h(a,m) = a-s*Rr_{m}$$

####&nbsp;&nbsp;&nbsp;&nbsp;其中$$a,m,s,R,r$$分别表示为特征点的坐标，特征点在初始特征点集中的序号，尺度因子，旋转矩阵和初始特征点相对于目标框的中心的点的相对坐标。

####&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;5) 删除异常点：计算voting两两之间的距离，并对距离进行聚类，取最大的类别作为内点，其他的点视为异常点，删除异常点。

####&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;6) 至此跟踪完成。若是在跟踪成功的点的个数少于阈值，则认为目标在当前帧的是不可见的。无目标框返回。若是跟踪成功的特征点的个数大于阈值则取所有跟踪的特征点的中心点$$\mu$$.

- <h4><b>检测特征点</b></h4>
 
####&nbsp;&nbsp;&nbsp;&nbsp;利用BRISK在当前帧图像整图进行特征点检测，将检测到的特征点和初始帧的所有特征点进行匹配。取能和初始帧的目标特征点匹配成功的特征点。
- <h4><b>合并跟踪和检测结果</b></h4> 

####&nbsp;&nbsp;&nbsp;&nbsp;将跟踪得到的特征点和检测得到的特征点集进行合并(作为下一帧跟踪的目标特征点)，输出最终的目标框：

####&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1）若是跟踪特征点为0，则认为目标不可见，当前帧无目标框返回，则用检测得到的特征点更新下一帧的目标特征点；

####&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2) 若有跟踪特征点，而检测特征点个数为0，则返回跟踪特征点为下一帧的目标特征点；

####&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3) 若是跟踪特征点和检测特征点同时存在，则将跟踪特征点和检测特征点进行合并成下一帧的目标特征点。

####&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4) 输出目标框为跟踪的特征点中心$$\mu$$和初始帧的目标框的四个点相对于目标框的中心点的相对坐标的和

$$\text{top_left} = \mu + s*R(\text{init_top_left}-\text{init_center})$$

####其中top_left为当前帧预测目标框的左上角点的坐标，s为当前帧的尺度因子，R为当前帧的旋转矩阵，init_top_left，init_center分别为初始帧的左上角的坐标和中心点。


####Reference

####[1] Consensus-based matching and tracking of keypoints for object tracking, WACV 2014.

####[2] Clustering of static-adaptive comrrspondences for deformatble object tracking, CVPR 2015.