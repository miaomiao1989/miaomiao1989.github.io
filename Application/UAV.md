---
layout: default
---
####&nbsp;&nbsp;&nbsp;&nbsp;这里总结的是参考文献[1]中的无人机上安装前置摄像头实现目标跟踪。下图展示了整个过程图：

<div style="height:650px;text-align: center">
<img src="../images/uva1.jpg">
</div>

####&nbsp;&nbsp;&nbsp;&nbsp;外部设备： 上图第一行表示外部设备，有跟踪目标(行人), 无人机(AR Drone 2.0), 和PC(wifi和无人机相连)。

####&nbsp;&nbsp;&nbsp;&nbsp;PC端算法模块： 主要有两个模块，一个是OpenTLD目标跟踪算法，一个是IBVS(Image based visual servoing)无人机控制模块。

####&nbsp;&nbsp;&nbsp;&nbsp;主要过程简单归纳为：有无人机的前置摄像头接受到目标视频数据，通过网络传输给通过wifi连接的PC端，PC端进行跟踪处理，将跟踪到的目标框转化为无人机的飞行指令，传送给无人机执行。**这里跟踪返回的结果有两种，一种情况是跟踪成功，成功返回跟踪到的目标框，则无人机的下一步飞行方向有跟踪到的目标框的信息计算得到； 另一种情况是，跟踪失败，则无人机进行悬停盘桓模式，根据自身的惯性模块和光流运动估计进行自身飞行平衡。**

####**1. IBVS(Image based visual servoing)控制器**

####&nbsp;&nbsp;&nbsp;&nbsp;由OpenTLD检测到的目标框信息为左上角的横纵坐标$$x_{bb}, y_{bb}$$，框宽$$w_{bb}$$高$$h_{bb}$$, 接受到的视频图像大小为宽$$w_{im}$$高$$h_{im}$$，则转换为无人机飞行器的信息为

$$f_{\delta}=\sqrt(\frac{w_{im}\times h_{im}}{w_{bb}\times h_{bb}}) \propto x_{tm}$$

####&nbsp;&nbsp;&nbsp;&nbsp;其中$$x_{tm}$$ 表示飞行器距离前方目标的水平距离，这一距离的大小也是影响了OpenTLD的跟踪效果的(距离太远或太近都会影响跟踪目标在接受到的图像上的尺度大小)。

####&nbsp;&nbsp;&nbsp;&nbsp;飞行器的跟踪和陆地上的智能机器人跟踪比起来还有一个比较复杂的地方是，飞行器还需要定义飞行角度，仰角和俯角等。具体计算请参考文献[1](待补充部分).


####**Reference**

[1] Computer vision based general object following for GPS-denied multirotor unmanned vehicles.

[2] Vision based GPS-denied Object Tracking and Following for Unmanned Aerial Vehicles.