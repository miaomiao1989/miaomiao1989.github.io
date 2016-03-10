---
layout: default
---
# **CNN人脸检测**

#### &nbsp;&nbsp;&nbsp;&nbsp;人脸检测是人脸处理的一个重要步骤，因为要为后续的人脸特征点检测和人脸识别提供人脸区域。这里主要讨论的是用卷积神经网络(convolutional neural network)进行人脸检测。

#### &nbsp;&nbsp;&nbsp;&nbsp;这里首先讨论Garcia 2004年在PAMI上发表的文献[1]。

<div style="text-align: center">
<img src="../Images/CFF-1.jpg">
</div>

#### &nbsp;&nbsp;&nbsp;&nbsp;首先，如上图所示。利用各二万张人脸图像样本(32x36)和非人脸样本(32x36)首先训练二分类CNN，是人脸图像分类结果为1，非人脸图像分类结果为-1，其中人脸图像是经过眼睛对齐的。

<div style="text-align: center">
<img src="../Images/CFF-2.jpg">
</div>

#### &nbsp;&nbsp;&nbsp;&nbsp;第二步，将待检测的整幅图像按一定的比例进行缩放，形成图像金字塔，对金字塔中的没幅图像输入到CNN中，利用训练好的滤波对其进行卷积和降采样，得最后的map（如上图所示）。对最终的map特征图形进行window扫描，并进行二分类，分出是否是人脸，对是人脸的window中心点进行标记,这对应的是下图中的第2步。得到了每层图像金字塔上的人脸候选点，利用正并错消法对候选点进行删减，依据是若是人脸图像区域，在一定邻域内出现的候选窗口即候选点个数应该很多，而误检的候选点邻域内出现的个数比较少。具体是统计候选点一定邻域内的候选点个数，小于一定阈值的认为是误检点，对其进行删除。对删减后的金字塔映射图像(即下图中的2)投影到元素尺度的图像上得到下图中的(3)，将(3)中的候选点映射到原图上的人脸检测窗口，得到(4)，对(4)中初步检测到的人脸进行再次筛选，连通区域内的候选点的邻域内的候选点个数小于阈值的认为是误检区域，进行删除，得到最终的检测结果。

<div style="text-align: center">
<img src="../Images/CFF-3.jpg">
</div>


#### **Refrence** 

#### [1] Convolutional face finder: a neural architecture for fast and robust face detection, 2004.
 
