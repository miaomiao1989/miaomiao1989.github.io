---
layout: default
---
# **Dropout**

#### **本文来自 [Dropout]: <http://blog.csdn.net/stdcoutzyx/article/details/49022443> **

#### 开篇明义，dropout是指在深度学习网络的训练过程中，对于神经网络单元，按照一定的概率将其暂时从网络中丢弃。注意是暂时，对于随机梯度下降来说，由于是随机丢弃，故而每一个mini-batch都在训练不同的网络。dropout是CNN中防止过拟合提高效果的一个大杀器，但对于其为何有效，却众说纷纭。在下读到两篇代表性的论文，代表两种不同的观点.

#### 
