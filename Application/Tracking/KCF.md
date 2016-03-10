---
layout: default
---

# **KCF**

#### &nbsp;&nbsp;&nbsp;&nbsp;本篇讨论的是TIP 2015年的论文“Higt-speed tracking with kernelized correlation filters”的跟踪算法[1]。这篇是继CSK[2]的升级版，而CSK是跟踪算法里以速度占绝对优势的算法。

## **1. Correlation filter 相关滤波器**

#### &nbsp;&nbsp;&nbsp;&nbsp;相关滤波器能够用于有效定位图像中的显著特征。对于构造一个用于检测图像中的一种特殊类型目标的滤波器问题，理想的滤波器期望是在相关输出值中，目标位置产生强峰值而其他位置为0。该滤波器可以由$$n$$幅训练图像$$\{f_{1},f_{2},\cdots, f_{n}\}$$几何来构造，训练图像实例包含感兴趣目 标及背景。为了训练获得最佳的相关滤波器， 首先根据训练图像建立对应的期望输出图$$\{g_{1}, g_{2}, \cdots, g_{n}\}$$。建立的期望输出可定义为目标位置是峰值而背景位置近似为0，可以采用二维高斯函数来定义:

$$
g_{i}(x, y) = exp\{-[(x-x_{i}^{2}) + (y-y_{i})^{2}]/ \sigma^{2}\}
$$

#### 其中$$(x_{i}, y_{i})$$为训练图像的目标真实坐标，$$\sigma$$为高斯参数，用以调节输入的尖锐程度。

#### &nbsp;&nbsp;&nbsp;&nbsp;相关又分为自相关和互相关，它是用来表示数据之间想死想的一种度量。自相关是指某个特征信息与其自身的相关性。互相关是指两个信息之间的相似关系。按照共振原理,将信息看成频域中的波形则互相关是判定两个波形的频率、幅度的相似性。如果两个波形的频率相似,则波形能产生很大的共振,当两个波形的频率相同时,产生的共振结果最大,即互相关能判定特定频率的波形与未知信息波形的相关性。

#### 一维相关运算的数学表达式为：

$$
c(x) = \int g(t)f(t+x)dt
$$

#### 其中，$$f(t), g(t)$$是运算函数，$$x$$是位移变量，函数$$c(x)$$是$$f(t)和g(t)$$的相关函数，它代表着两个函数在定义域上的相似度。若转换到频域上计算，则上式可以记为：

$$
c(x) = \mathscr{F}^{-1}\{G(t)*F(t+x)\}
$$

其中， $$G(t), F(t+x)$$分别为$$g(t), f(t+x)$$的傅里叶变换， $$\mathscr{F}$$为傅里叶反变换， ‘*’为内积运算。 <font color = "red">通过上述两式的比较可以发现，在频域计算相关性的计算复杂度明显降低主要是因为操作由卷积换成了内积运算</font>。

#### 相关滤波器具有良好的退化性，时移不变性和闭解性等优点。 

- 由于相关滤波器处理测试图像的是指是计算两个模式之间的相关性，所以当某一模式发生部分退化时，相关输出并不会改变尖峰位置，只会将峰值减少。良好的退化性可以体现在若测试图像中的一些像素被遮挡或者污染时，并不会影响尖峰在相关屏幕的位置(只会将峰值降低，但峰值依然明显)。
- 时移不变性是指若为输入的测试图像在空域上做任何的平移操作，则尖峰在相关平面上的位置也会发生相应的平移。<font color = "red">当图像发生平移时，只需要通过尖峰的位置既可以判断测试图像的偏移程度。(这也是本文中预测检测位置的核心思想)</font>

## **2. KCF tracking**

#### &nbsp;&nbsp;&nbsp;&nbsp;这里主要是讨论的文献[1]的跟踪方法。

#### &nbsp;&nbsp;&nbsp;&nbsp;现有的跟踪算法主要分为两种方法：一种是generative跟踪方法，通常可以理解为模板匹配法；另一种是discriminative方法，又称为 tracking-by-detection方法，是目前跟踪方法的主流思想。训练样本的选择基本上就以目标中心为提取正样本，然后基于周围的图像提取负样本，如下图所示。这种方法是将跟踪方法视为二值分类方法，通过对目标周围区域采样正负样本，训练二分类分类，并在候选测试区域用分类器进行检测，寻找和上一帧模板最为相近的候选patch作为当前帧的跟踪结果。大部分算法都是采用非正即负的方法来标记训练样本，即正样本标签为1，负样本为0。这种标记样本的方法有一个问题就是不能很好的反应每个负样本的权重，即对离中心目标远的样本和离中心目标的近的样本同等看待。所以就有算法提出使用连续的标签进行标记样本，即根据样本中心里目标的远近分别赋值[0,1]范围的数。离目标越近，值越趋向于1，离目标越远，值越趋向于0。事实也证明这种标记样本的方法能得到更好的效果，比如Struck[3]。

<div style="text-align: center">
<img src="../Images/KCF1.jpg">
</div>


### **1. 线性回归[4]**

#### &nbsp;&nbsp;&nbsp;&nbsp;线性回归过程实际上是一个岭回归的问题，或者叫做正则化最小二乘问题，它有一个闭式的解。假设给定一些训练样本及其回归值$$\{(x_{1}, y_{1}), (x_{2}, y_{2}), \cdots,(x_{i}, y{i}), \cdots\}$$, 训练的最终目标是找到一个函数$$f(z) = w^{T}z$$使得如下残差函数最小：

$$
\underset{w}\min \underset{i}\sum(f(x_{i})-y_{i})^{2}+\lambda\|w\|^{2}
$$

其中$$\lambda$$是正则化参数，是为了防止过拟合的。上述最小化问题是可导的，因此可以求得闭型解为:

$$
w = (X^{T}X+\lambda I)^{-1}x^{T}y
$$

其中$$X^{T}$$为转置, $$I$$为单位矩阵。

#### &nbsp;&nbsp;&nbsp;&nbsp;<font color = "red">虽然可以通过上述公式求得解w，但是求逆的过程是相当费时的，而本文中则将问题转换在傅里叶域进行求解，从而避免了求逆的过程。 是这篇文章最重要的特点，也是实现跟踪速度较快的原因之一。</font>

在傅里叶域里上述$$w$$的公式可以转化为：

$$
w = (X^{H}X+\lambda I)^{-1}x^{H}y                       
$$

其中$$X^{H}$$为共轭转置，即$$X^{H} = (X^{*})^{T}, X^{*}$$为复共轭。

#### &nbsp;&nbsp;&nbsp;&nbsp;<font color = "red">显然，单单仅仅是转换到傅里叶里是还是需要求解较大的线性系统才能求得解，这是实时跟踪中也是不乐观的，这里需要探究傅里叶域的X的一些性质来进一步简化运算。</font>

### **2. 循环矩阵**

#### &nbsp;&nbsp;&nbsp;&nbsp;在线性代数中，循环矩阵是一种特征形式的Toeplitz矩阵，它的行向量的每个元素都是前一个行向量各元素依次右移一个未知得到的结果。循环矩阵可以用离散傅里叶变换迅速求解。

#### 形为：

<div style="text-align: center">
 <img src="../Images/KCF2.png">
 </div>

的$$n\times n$$维矩阵就是循环矩阵。

 - **特性**
 
   **1)**&nbsp;&nbsp;&nbsp;&nbsp;对于两个循环矩阵$$A$$与$$B$$来说，$$A+B$$也是循环矩阵。$$AB$$也是循环矩阵，且$$AB=BA$$.

   **2)**&nbsp;&nbsp;&nbsp;&nbsp;循环矩阵的逆矩阵也是循环矩阵

   **3)**&nbsp;&nbsp;&nbsp;&nbsp;循环矩阵的特征向量矩阵是同样维数的离散傅里叶变换矩阵，因此循环矩阵的特征值也可以很容易的通过傅里叶变化计算出来。假设上述循环矩阵$$C$$的特征值为$$[\lambda_{0}, \lambda_{1}, \cdots, \lambda_{n}]$$, 而循环矩阵的一行，即生成循环矩阵的基向量为$$c = [c_{1}, c_{2}, \cdots, c_{n}]$$, 则有特征值为相应的基向量的傅里叶变换：
      
<div style="text-align: center">
 <img src="../Images/KCF3.png">
 </div>

其中$$\mathscr{F}$$为傅里叶变换。

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**4)**&nbsp;&nbsp;&nbsp;&nbsp;循环矩阵可以被傅里叶矩阵对角化， 即
$$
C = F^{H} \Lambda F
$$

其中$$F$$为傅里叶矩阵，元素为：$$[F]_{ij} = \frac{1}{\sqrt{n}e^{j2\pi ik/n}}, 0<=i,j<n-1$$
  
 - **用循环矩阵来解线性方程**
 
  设矩阵方程

$$
Cx=b
$$

其中$$C$$是$$n$$维方形循环矩阵，这样就可以将方程表示成循环卷积

$$
c*x = b
$$

其中$$c$$是循环矩阵$$C$$的第一列，$$c,x,b$$分别向每个方向循环。用离散傅里叶变化将循环卷积转成成两个变量之间的乘积：

$$
\mathscr{F}_{n}(c*x) = \mathscr{F}_{n}(c)\mathscr{F}_{n}(x)=\mathscr{F}_{n}(b)
$$

因此有：

$$
x = \mathscr{F}^{-1}\left(\frac{\mathscr{F}_{n}(b)}{\mathscr{F}_{n}(c)}\right)
$$

用这种傅里叶变换方法求解线性方程组，比起高斯消去法的速度要快很多，尤其是当使用快速傅里叶变换的时候速度要更快。

### **3. 构建循环样本矩阵**

#### 由上述循环矩阵的性质可以看出，若是训练样本矩阵$$X$$能够为循环矩阵，则将会很大程度上减少计算量。而我们常用的采取正负样本的方法是以目标图像为base图像，以及该图像左右上下偏移得到一系列的图像块作为负样本进行训练，如下图所示：

<div style="text-align: center; height: 370px">
 <img src="../Images/KCF4.png">
 </div>

#### 而本文中可以采用利用基样本循环偏移构造训练样本，如下图所示：

<div style="text-align: center; height: 230px">
 <img src="../Images/KCF5.png">
 </div>

#### 为根据基样本向下和向上分别循环15行和30行得到的样本，这样通过循环构造方式，可以有一个基样本通过循环方式构造出所有的可能的训练样本，实现了稠密采样(dense sample)。

#### 这里我们为了简单计算，依然以一维为例标记，基样本记为$$x=[x_{1}, x_{2}, x_{3},\cdots, x_{n}]$$,则由基样本构造的所有训练样本可以记为循环矩阵

<div style="text-align: center; height: 200px">
 <img src="../Images/KCF6.png">
 </div>

则根据上面我们所讲述的循环矩阵的性质，其循环矩阵$$X$$可以通过傅里叶矩阵对角化，即：

$$
X = Fdiag(\widehat{x})F^{H}
$$

$$F,F^{H}$$分别为傅里叶矩阵和其共轭转置， $$\widehat{x}$$为基样本的傅里叶变换$$\widehat{x} = \mathscr{F}(x)$$。则有如下：

$$
\begin{array}{cc}
X^{H}X &=& F diag(\widehat{x}^{*})F^{H}Fdiag(\widehat{x}F^{H})\\
& =& Fdiag(\widehat{x}^{*})diag(\widehat{x})F^{H}\\
&=& Fdiag(\widehat{x}^{*}\odot\widehat{x})F^{H}
\end{array}
$$

其中$$\widehat{x}^{*}$$为傅里叶变化的复共轭, $$\odot$$表示按元素内积运算。

#### 将这一表达式带入到线性回归的方程中，可以求得在傅里叶域的$$\widehat{w}$$:

$$
\begin{array}{cc}
\widehat{w}& =& diag\left(\frac{\widehat{x}^{*}}{\widehat{x}^{*}\odot\widehat{x}+\lambda}\right)\widehat{y}\\
&=&\frac{\widehat{x}^{*}\odot\widehat{y}}{\widehat{x}^{*}\odot\widehat{x}+\lambda}
\end{array}
$$

#### 将这一公式与线性回归下的$$w$$求解的公式做对比，可以看到将$$w$$变换到傅里叶域之后避免求解逆矩阵之后，计算量会大大降低。

### **4. 核回归**

#### 以上介绍的都是线性回归的情况，如果能引入核函数，分类器的性能将会更好。核函数的引入是把特征空间映射到一个更高维的空间去，这里我们假设这个映射函数为$$\varphi(x)$$，则分类器的权重向量变为:

$$
w = \underset{i}\sum\alpha_{i}\varphi(x)
$$

这样我们最终要求解的参数就由$$w$$变为$$\alpha$$ ，这里$$\alpha=[\alpha_{1}, \alpha_{2}, \cdots, \alpha_{n}]$$ 。因为其实我们并不知道核函数映射的高维空间是什么，我们只是知道高维空间下的两个向量的乘积可以通过一个映射函数把其在低维空间下的乘积映射到高维空间，也就是核函数。这里设<font color = "red">不同样本之间的乘积的核函数结果</font>组成的矩阵为:

$$
K_{ij} = k(x_{i}, x_{j}) = <\varphi(x_{i}), \varphi(x_{j})>
$$

这样最终的回归函数变为，

$$
f(z) = w^{T}z = \overset{n}{\underset{i=1}{\sum}}\alpha_{i}k(z, x_{i})
$$

直接计算上述函数相对来说是很耗时的，下面还是结合循环矩阵的特性实现一种快速的核函数计算方法。

 - **快速训练**

#### &nbsp;&nbsp;&nbsp;&nbsp;基于核函数的岭回归的解为：

$$\alpha = (K+\lambda I)^{-1}y$$

其中$$K$$是核函数矩阵，由于求解逆的过程是耗时的，我们利用循环矩阵进行求解(<font color="red">这里因为训练样本X是循环矩阵，则核矩阵K也是循环矩阵，在论文中有具体证明</font>)。既然已知了$$K$$是循环矩阵，则上式的求解就可以同理转换到傅里叶域进行计算，从而避免求解逆矩阵的过程，可以得到解为：

$$
\widehat{\alpha} = \frac{\widehat{y}}{\widehat{k}^{xx} + \lambda}
$$

其中， $$k^{xx}$$是核函数矩阵$$K$$的第一行元素组成的向量,是向量$$x$$和其自身的和相关(kernel correlation)。
<!--
$$
K = \left[\begin{array}{cccc}
k(x_{1}, x_{1})& & k(x_{1}, x_{2})& & \cdots & &k(x_{1}, x_{n})\\
k(x_{2}, x_{1})& & k(x_{2}, x_{2})& & \cdots & &k(x_{2}, x_{n})\\
\vdots          & &\vdots           & &\ddots  & &\vdots\\
k(x_{n}, x_{1}) & &k(x_{n}, x_{2})  & &\cdots  & & k(x_{n}, x_{n})\end{array}\right]
$$
-->


 - **快速检测**

#### &nbsp;&nbsp;&nbsp;&nbsp;通过训练样本基样本$$x$$和测试样本基样本$$x$$构造其核相关：$$k^{xz}$$,并构造循环核函数矩阵：

$$K^{z} = C(k^{xz})$$

其中， $$k^{xz}$$是这个循环矩阵的第一行组成的向量。这样就可以同时计算基于测试样本$$z$$的循环偏移构成的所有测试样本的响应，即，

$$f(z) = (K^{z})^{T}\alpha$$

这里的$$f(z)$$是一个向量，由测试基样本$$z$$不同循环骗一下的响应值组成。根据循环矩阵变换到傅里叶域为：

$$\widehat{f}(z) = (\widehat{k}^{xz})\odot\widehat{\alpha}$$

这里$$k^{xz}$$是当前测试帧的基样本(以上一帧的目标框的中心为中心采样)， 和训练基样本(上一帧的精确检测的目标框的中心为中心采样)的核相关在傅里叶域的值，$$\widehat{\alpha}$$为训练过程中(上一帧)计算得到的值。$$\odot$$ 表示按元素内积运算。
<font color="red">最后可以根据</font>$$f(z)$$<font color = "red">中的最大的响应值对应的位置即当前测试帧中目标偏移的位置.</font>

- **核函数**
 
####&nbsp;&nbsp;&nbsp;&nbsp;这里用到的核函数可以是高斯核函数也可以是多项式核函数[1]。

- **具体实现过程**

&nbsp;&nbsp;&nbsp;&nbsp;1. 计算期望输出值$$y$$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;以中心点坐标为0，目标框转换成中心点为原点的坐标系，假设目标框大小为10x10，则x的坐标为[-5,-4,-3,-2,-1,0,1,2,3,4],y的坐标也为上述。假设$$Y$$为高斯函数定义的期望输出图，应该是目标位置是峰值而其余背景为0.

$$Y = exp[-\frac{1}{2\sigma^{2}}(x^{2}+y^{2})]$$

将Y的值转换到傅里叶域

$$\widehat{Y} = \mathcal{F}(Y)$$

&nbsp;&nbsp;&nbsp;&nbsp;2. 从第一帧中训练模型

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1. 以目标框扩大1.5倍的图像块作为输入进行特征提取$$x$$(可以是HOG特征，原始灰度图像特征，也可以是CNN提取的特征，在后面的博文中会详细讲述).

将提取的特征转换到傅里叶域：

$$\widehat{x} = \mathcal{F}(x)$$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2. 计算$$\widehat{x},\widehat{x}$$本身之间的核相关性，高斯核函数的和相关性为(假设下述公式中$$x=\widehat{x}$$为了方便书写)：

$$\widehat{K}^{xx} = \mathcal{F}(exp(-\frac{1}{\sigma^{2}}(\|x\|^{2} + \|x\|^{2}-2\mathcal{F}^{-1}(\widehat{x^{*}}\odot\widehat{x^{*}}))))$$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3. 计算参数$$\alpha$$

$$\widehat{\alpha} = \frac{\widehat{Y}}{\widehat{K}^{xx}+\lambda}$$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4. 更新模型

$$\text{model_}\widehat{\alpha} = \widehat{\alpha}$$

$$\text{model_}\widehat{x} = \widehat{x}$$

&nbsp;&nbsp;&nbsp;&nbsp;3. 从第二帧开始检测

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1. 以上一帧检测到的中心点为中心点，以目标框扩大1.5倍在当前帧上提取图像库，提取特征为

$$\widehat{x}_{i} = \mathcal{F}(x_{i})$$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2. 计算$$\widehat{x}_{i}$$和$$\text{model_}x$$的核相关性$$\widehat{k}^{\text{model_}xx_{i}}$$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3. 计算相关性检测当前帧的位置：

$$\widehat{f(x)}_{i} = (\widehat{k}_{xx_{i}})\odot(\widehat{\alpha})$$

$$\text{response} = \mathcal{F}^{-1}(\widehat{k}^{xx_{i}}\odot\text{model_}\widehat{\alpha})$$

取response中最大位置的坐标为当前帧的偏移量。

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4. 以预测到的新的目标框在当前帧重新计算$$\widehat{\alpha}， \widehat{x}$$,重复第一帧训练的第1, 2，3步骤。

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;5. 重新更新模型：

$$\text{model_}\widehat{\alpha} = (1-\mu)\text{model_}\widehat{\alpha} + \mu\widehat{\alpha}$$

$$\text{model_}\widehat{x} = (1-\mu)\text{model_}\widehat{x} + \mu\widehat{x}$$

#### **Reference**

[1] Joao F. Henriques, **Higt-speed tracking with kernelized correlation filters**, TPAMI 2015.

[2] Joao F. Henriques, **Exploiting the circulant structure of tracking-by-detection with kernels**, 	ECCV 2012.

[3] S. Hare, **Struck: Structured output tracking with kernels**, ICCV 2011.

[4] <http://www.skyoung.org/kcf-tracking-method/>
