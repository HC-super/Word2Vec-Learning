# Word2Vec-Learning

For learning NLP！

### 部分代码的神经网络结构

[CNN](https://github.com/HC-super/Word2Vec-Learning/blob/main/imdb_lstm.py)

![imdb_cnn](https://github.com/HC-super/Word2Vec-Learning/blob/main/imdb_cnn.png)

[LSTM](https://github.com/HC-super/Word2Vec-Learning/blob/main/imdb_lstm.py)

![imdb_lstm](https://github.com/HC-super/Word2Vec-Learning/blob/main/imdb_lstm.png)

[CNN_LSTM](https://github.com/HC-super/Word2Vec-Learning/blob/main/imdb_cnn_lstm.py)

![imdb_cnn_lstm](https://github.com/HC-super/Word2Vec-Learning/blob/main/imdb_cnn_lstm.png)

[stacked_LSTM（堆叠循环神经网络)](https://github.com/HC-super/Word2Vec-Learning/blob/main/imdb_stacked_lstm.py)

![imdb_stacked_lstm](https://github.com/HC-super/Word2Vec-Learning/blob/main/imdb_stacked_lstm.png)

[bidirectional_lstm（双向循环神经网络）](https://github.com/HC-super/Word2Vec-Learning/blob/main/imdb_bidirectional_lstm.py)

![imdb_bidirectional_lstm](https://github.com/HC-super/Word2Vec-Learning/blob/main/imdb_bidirectional_lstm.png)
## 其他模型

[胶囊网络capsule](https://github.com/HC-super/Word2Vec-Learning/blob/main/imdb_capsulenet.py)

[自注意力模型](https://github.com/HC-super/Word2Vec-Learning/blob/main/imdb_attention_lstm.py)

# 词向量

最简单的方法可以采用one-hot encoder用向量来表示取值和分类，若有4个值会取

$[1,0,0,0]$	$[0,1,0,0] $	$[0,0,1,0]$	$[0,0,0,1]$	四个单位向量

```python
from sklearn.preprocessing import  OneHotEncoder

enc = OneHotEncoder()
enc.fit([[0, 0, 3],
         [1, 1, 0],
         [0, 2, 1],
         [1, 0, 2]])

ans = enc.transform([[0, 1, 3]]).toarray() 
# 如果不加 toarray() 的话，输出的是稀疏的存储格式，即索引加值的形式，也可以通过参数指定 sparse = False 来达到同样的效果
print(ans) 
# 输出 [[ 1.  0.  0.  1.  0.  0.  0.  0.  1.]]
```

如上，第1列有两个取值0和1，则用二维列向量表示为$[1,0]^T,[0,1]^T$
第二列有3个取值用，每个取值便用三维单位向量来表示。以此类推，几个取值就用几维向量。

若训练样本中有丢失分类特征时，必须显式地设置n_value来防止编码出错。

跟一般地，我们会使用`fit_transform`直接用要转换的one-hot格式的矩阵来直接得到输出编码。不会用到训练样本。

但是one-hot有几个缺点，首先它没有考虑词与词之间相似度的问题，有多少不同的词就会有多少维向量。这样词与词之间的相似度无法计算，其次，如果词太多会造成向量的维度过高。

word2vec是谷歌开发的用于试图理解词之间情感和意义的工具，本质上是一种降维操作。工作方法类似于深层方法（Deep Approaches）如递归神经网络和深层神经网络，但它的计算效率更高。它可以利用**向量**来表示词与词之间的关系和相似度（余弦相似度），越有区分度的词越远离空间分为窗口词和中心词（前后文预测中间词skip-gram，中间词预测前后文CBOW）以计算词向量。

skip-gram（跳字模型）的网络结构如下：

![image-20210531114959080](https://tva1.sinaimg.cn/large/008i3skNgy1gr1us5kkmcj30aw0cimz0.jpg)

（skip-gram可以理解为通过一个词来预测上下文）每个词被分为两个d维向量用来计算条件概率。假设这个词的索引为i，则当它为中心词时的的向量表示为$v_i\in\mathbb{R}^d$时，背景词向量表示为$\boldsymbol{u}_i\in\mathbb{R}^d$。设中心词$w_c$在词典中索引为$c$，背景词$w_o$在词典中索引为$o$，给定中心词生成背景词的条件概率可以通过**对向量内积做softmax运算**而得到：

$$P(w_o \mid w_c) = \frac{\text{exp}(\boldsymbol{u}_o^\top \boldsymbol{v}_c)}{ \sum_{i \in \mathcal{V}} \text{exp}(\boldsymbol{u}_i^\top \boldsymbol{v}_c)},$$

其中词典索引集$\mathcal{V} = \{0, 1, \ldots, |\mathcal{V}|-1\}$。假设给定一个长度为$T$的文本序列，设时间步$t$的词为$w^{(t)}$。假设给定中心词的情况下背景词的生成相互独立，当背景窗口大小为$m$时，跳字模型的**似然函数**即**给定任一中心词生成所有背景词的概率**。

CBOW是用上下文来预测这个词。网络结构如下

![image-20210531120551378](https://tva1.sinaimg.cn/large/008i3skNgy1gr1usiv1gdj30a00cidh8.jpg)

> 统计学较深度学习更快

倒排索引：通过对文章进行索引以便尽快找到相关文档，CBOW类似于完形填空。通过上下文来预测中心词。而我们通常会用skip-gram，因为其效果更好。

词向量的几种典型应用：

1. 将词向量以某种特定方法结合起来就可以对某句话加以理解
2. 可以在向量空间中寻找同义词
3. 词语的距离换算———如：king - man = queen

但无法解决一词多义问题。



## tf-idf

 它使用词语的**重要程度**与**独特性**来代表每篇文章，然后通过**对比搜索词与代表的相似性**，给你提供最相似的文章列表

TF：词频（文章为中心的局部信息）

某一个词出现频率高预示着本词在文档中更加重要。

IDF: 逆文本频率指数（全局词信息）

光用TF没有办法将一些没有代表性的常用词如：“你、我、他”等区分开来，IDF的任务是将在其他所有文档中出现的词的IDF降低。

TF-IDF实际上是一个数学矩阵。

IDF的选取很重要。

tfdif忽略了词与词的顺序

## 句向量

将离散的词向量加工成句向量。encoding将复杂信息压缩为精简信息。循环神经网络RNN可以通过同层神经元的串联来按顺序理解句子。

sequence to sequence模型和CNN模型

## 深度前馈网络

![image-20210531120957135](https://tva1.sinaimg.cn/large/008i3skNgy1gr1usmpxbhj30my09e0up.jpg)

激活函数：

![image-20210531152642083](https://tva1.sinaimg.cn/large/008i3skNgy1gr1uspzt1gj30gs09odhy.jpg)

![image-20210531152715170](https://tva1.sinaimg.cn/large/008i3skNgy1gr1usu76dcj30ku0agdjc.jpg)

ReLu函数在x<0时会抹去信息。

sigmoid函数用于解决二分类问题，但会因饱和现象导致梯度消失

![image-20210531152845306](https://tva1.sinaimg.cn/large/008i3skNgy1gr1usx22xpj30m20b2n1c.jpg)

![image-20210531153123758](https://tva1.sinaimg.cn/large/008i3skNgy1gr1ut0ahz6j30ly0a2adr.jpg)

![image-20210531153206195](https://tva1.sinaimg.cn/large/008i3skNgy1gr1ut5hho7j30ki0a8aet.jpg)

![image-20210531153421481](https://tva1.sinaimg.cn/large/008i3skNgy1gr1ut8qz0mj30uq0f4dnx.jpg)

![image-20210531153510319](https://tva1.sinaimg.cn/large/008i3skNgy1gr1utdz9lvj30tw0j4wx2.jpg)

![image-20210531153518592](https://tva1.sinaimg.cn/large/008i3skNgy1gr1uth4tb6j30m40aqn1w.jpg)

![image-20210531153525359](https://tva1.sinaimg.cn/large/008i3skNgy1gr1utkd8ipj30gg08k0vw.jpg)

较为常用，且斜率可选定（超参数）

MLP（Multi-layer Perceptron）Dense net 密集连接 

神经网络：多个线性决策边界的叠加➡️非线性决策边界

![image-20210531153749910](https://tva1.sinaimg.cn/large/008i3skNgy1gr1utn8tbzj30ca0cigss.jpg)

链式求导和反向传播

使损失函数最小化

![image-20210531154103252](https://tva1.sinaimg.cn/large/008i3skNgy1gr1utq3gspj30fc07ijto.jpg)

由于神经网络层数过多导致导数解析式难以写出，利用数值方法

![image-20210531154256899](https://tva1.sinaimg.cn/large/008i3skNgy1gr1utsdmj6j30i208ognh.jpg)

![image-20210531154305338](https://tva1.sinaimg.cn/large/008i3skNgy1gr1utwh5oqj30m20c0wib.jpg)

mul：如2x关于x求导为2，关于2求导为x

max：将梯度传给较大路

tensorflow（张量流动）	pytorch（计算图中可求梯度的Numpy）

CS231N CV

CS 229 ML

CS224 NLP

经典论文：ImageNet Classification with Deep Convolutional Neural Networks

介绍了不使用ReLu和使用ReLu的区别

![image-20210531154827929](https://tva1.sinaimg.cn/large/008i3skNgy1gr1uty3rtuj30q00a0wg7.jpg)



# CNN

称为卷积神经网络

卷积：对图片一块像素的信息进行处理

利用卷积核对图像进行特征提取



输出就叫做feature map，有多少个卷积核就有多少个feature map，卷积核具体的值不需要进行设置，系统自己会设置

池化（pooling）：下采样。平均池化和最大值池化。池化的作用：1. 减少参数量，尽量保持原数据的原始特征	2. 防止过拟合 	3. 可以为卷积神经网络带来**平移不变性** （会丢失距离信息）

Embedding：

embedding 在深度学习中经常和manifold（流形）搭配使用

流行假设：**Manifold Hypothesis**（流形假设）。流形假设是指“自然的原始数据是低维的流形嵌入于(embedded in)原始数据所在的高维空间”。深度学习的任务就是把高维原始数据（图像，句子）映射到低维流形，使得高维的原始数据被映射到低维流形之后变得可分，而这个映射就叫嵌入（Embedding）。比如Word Embedding，就是把**单词组成的句子映射到一个表征向量。**

![image-20210522103538598](https://tva1.sinaimg.cn/large/008i3skNgy1gr1uu15ircj31km0heh08.jpg)

![image-20210522103633516](https://tva1.sinaimg.cn/large/008i3skNgy1gr1uu4423sj31xy0lw18t.jpg)

![image-20210522103725729](https://tva1.sinaimg.cn/large/008i3skNgy1gr1uu6h0hkj31sg0rqwqj.jpg)

利用卷积核对底层特征进行抽取，符合卷积核的特征会被提取到feature map中

二分类问题（sigmoid），多分类问题（softmax）

激活函数——ReLU 函数：![img](https://bkimg.cdn.bcebos.com/formula/ae9d12662d9e1073200f081659ff7ea3.svg)线性整流作为神经元的激活函数，定义了该神经元在线性变换![img](https://bkimg.cdn.bcebos.com/formula/fb2b2510cb8c97bb4b8ee347317804a4.svg)之后的非线性输出结果。换言之，对于进入神经元的来自上一层神经网络的输入向量![img](https://bkimg.cdn.bcebos.com/formula/40482bf9a174030a55e50aa416fb29af.svg)，使用线性整流激活函数的神经元会输出![img](https://bkimg.cdn.bcebos.com/formula/24175eeaf4905a7acc3025fa7f3f660f.svg)至下一层神经元或作为整个神经网络的输出（取决现神经元在网络结构中所处位置）。


七层卷积神经网络（用于手写数字识别）

充分利用CNN的局部感受野，权值共享（每次卷积用同一个卷积核）、下采样的特点保证平移、缩放、变形的不变性

![image-20210522120644403](https://tva1.sinaimg.cn/large/008i3skNgy1gr1uub7uhgj31s20fq7u5.jpg)

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gr1uuca7sej31hw0u0hdu.jpg" alt="image-20210522121756245" style="zoom:200%;" />

每个神经元仅与输入神经元的一块区域连接

上采样：转置卷积，将输入的信息扩大。
![image-20210522174612691](https://tva1.sinaimg.cn/large/008i3skNgy1gr1uues2z4j31cg0syaoi.jpg)

![image-20210523120856168](https://tva1.sinaimg.cn/large/008i3skNgy1gr1uuvaz3sj31jk0ss4mn.jpg) 

LSTM（长短时循环）和GRU（门控单元）

解决序列问题

1对1，1对多，多对一，多对多。

![image-20210531161621366](https://tva1.sinaimg.cn/large/008i3skNgy1gr1uuz09xcj31650lytmp.jpg)

![image-20210531161630574](https://tva1.sinaimg.cn/large/008i3skNgy1gr1uv1tpsoj30ow0csadi.jpg)

![image-20210531161640914](https://tva1.sinaimg.cn/large/008i3skNgy1gr1uv45zhdj30s20e243q.jpg)

![image-20210531161647770](https://tva1.sinaimg.cn/large/008i3skNgy1gr1uv6h7hdj30qm0dk7ah.jpg)

令神经网络拥有了序列维度的记忆

![image-20210531161721423](https://tva1.sinaimg.cn/large/008i3skNgy1gr1uvb1ol0j30nc0bogqb.jpg)

![image-20210531161949583](https://tva1.sinaimg.cn/large/008i3skNgy1gr1uvfgvukj31cb0rs17r.jpg)

![image-20210531162001760](https://tva1.sinaimg.cn/large/008i3skNgy1gr1uvk3m2dj30ow0cmafa.jpg)

![image-20210531162008030](https://tva1.sinaimg.cn/large/008i3skNgy1gr1ure54o0j30my0d0wld.jpg)

![image-20210531162023288](https://tva1.sinaimg.cn/large/008i3skNgy1gr1urayopgj31a40o44cl.jpg)

![image-20210531162741314](https://tva1.sinaimg.cn/large/008i3skNgy1gr1ur89hbnj31ks0qih2a.jpg)

![image-20210531162802964](https://tva1.sinaimg.cn/large/008i3skNgy1gr1ur3h67qj318m0mqqds.jpg)

![image-20210531162847600](https://tva1.sinaimg.cn/large/008i3skNgy1gr1ur06rncj30k609gmyk.jpg)

![image-20210531163004553](https://tva1.sinaimg.cn/large/008i3skNgy1gr1uqxt9dyj30vw0lpdm4.jpg)

![image-20210531163020407](https://tva1.sinaimg.cn/large/008i3skNgy1gr1uqulfu2j30jm0a6abw.jpg)

![image-20210531163051669](https://tva1.sinaimg.cn/large/008i3skNgy1gr1uqqwet9j30rg0fcdiq.jpg)

![image-20210531163119575](https://tva1.sinaimg.cn/large/008i3skNgy1gr1uqo502qj30lk0b8q65.jpg)

![image-20210531163336944](https://tva1.sinaimg.cn/large/008i3skNgy1gr1uqkso81j30lo09qdi7.jpg)

![image-20210531163451039](https://tva1.sinaimg.cn/large/008i3skNgy1gr1uvwrf0fj30l40asn9d.jpg)

![image-20210531163501798](https://tva1.sinaimg.cn/large/008i3skNgy1gr1upwv4skj30f408amyw.jpg)

![image-20210531163514259](https://tva1.sinaimg.cn/large/008i3skNgy1gr1uprtuy2j30zc0ii0xf.jpg)







## keras

优化器（optimizer）

常用：

SGD：随机梯度下降

momentum：动量优化，避免取得局部最优解

AdaGrad：抑制陡峭方向上的梯度震荡

激活函数：sigmoid➡️改进 hardsigmoid

tanh 会有梯度消失的问题

不同问题用不同的激活函数，一般常用的有：二分类sigmoid，多分类softmax，CV用relu，RNN和GAN用tanh

回调函数来监控模型异常，用来查看模型内在状态和统计。

CALL BACK

常用数据集

特征提取：不要全连接层，仅保留卷积层

微调（fine-tuning）将原模型的全连接层换为自己的全连接层

当要改动模型内部的卷积层时，优化器最好选用SGD并设置一个较低的学习率。

keras模型对比中：

TOP1:模型预测的第一个结果正确

TOP5:模型预测的前5个结果包含正确结果

Application模块很重要

Word-net：知识图谱

layer.tranable = False 将基模型的所有层都冻住，不许它们训练

Inception:同一个输入进行分别4种处理后汇总输出给下一层

正则化：防止过拟合如

1. 数据增强：Data Augmentation（如对训练图像进行翻转和裁切）

2. 早停
3. Dropout随机掐死部分神经元

kernel-regularizer 惩罚权重（常用）

对w的平方和进行惩罚为L2正则项

$|w|$为L1正则化

bias_regularize 偏置项惩罚 是模型输入到输出尽可能一致

activity_regularize 激活值惩罚输出尽可能小

模型可视化：sci-kit learn的grid_search API来寻找最优超参数

结构化机器学习

keras.utils（utility常用的轮子）

one-hot独热向量编码，利用向量来表示不同的类

## 自注意力模型

Self-Attention

on the relationship between SelfAttention and CNN<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gr1upisvs0j30py0c83zm.jpg" alt="image-20210531182927927" style="zoom:33%;" />

Self-Attention VS. RNN

RNN中的vector无法同时产生需传递

Self-Attention中的vector可同时产生

## capsulenet



paper:	Transformers are RNNS ：Fast Autoregressive Transformers with Linear Attention

​				Long Range Arena: A benchmark for Efficient Transformers

Capsule: output a vector

Neuron:output a value

Geoffrey Hinton深度学习创始人之一

他的论文[CapsuleNet](https://arxiv.org/abs/1710.09829)

![image-20210531194908351](https://tva1.sinaimg.cn/large/008i3skNgy1gr1uoz24wxj31ic0fc0v9.jpg)

![image-20210531200840704](https://tva1.sinaimg.cn/large/008i3skNgy1gr1v9as623j30uy0fo0ua.jpg)

![image-20210531201459652](https://tva1.sinaimg.cn/large/008i3skNgy1gr1vfvum35j31lu0lc7nx.jpg)

![image-20210531202119963](https://tva1.sinaimg.cn/large/008i3skNgy1gr1vmhrcs2j31d10u01ky.jpg)

过程类似于RNN

capsule can also be convolutional

- simply replace filter with capsule

output layer and loss

![image-20210531202509621](https://tva1.sinaimg.cn/large/008i3skNgy1gr1vqggiinj30du07wmz2.jpg)

![image-20210531202528562](https://tva1.sinaimg.cn/large/008i3skNgy1gr1vqtt1tgj30dw0aiqb5.jpg)

![image-20210531202552677](https://tva1.sinaimg.cn/large/008i3skNgy1gr1vr7bh6ej30dw0aan4v.jpg)

使得capsule net是Equivariance的 

![image-20210531202856210](https://tva1.sinaimg.cn/large/008i3skNgy1gr1vudtjfdj30cs09g45u.jpg)

![image-20210531202842704](https://tva1.sinaimg.cn/large/008i3skNgy1gr1vu56iozj30dw0akgts.jpg)
