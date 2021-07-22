



# Word2Vec-Learning

For learning NLP！

## 代码索引

[数据预处理](https://github.com/HC-super/Word2Vec-Learning/blob/main/imdb_preprocess.py)

[ALBERT](https://github.com/HC-super/Word2Vec-Learning/blob/main/Bert%20Family/Albert.py)

[BERT](https://github.com/HC-super/Word2Vec-Learning/blob/main/Bert%20Family/Bert.py)

[DistilBert](https://github.com/HC-super/Word2Vec-Learning/blob/main/Bert%20Family/DistilBert.py)

[RoBERTa](https://github.com/HC-super/Word2Vec-Learning/blob/main/Bert%20Family/RoBERTa.py)

[CNN](https://github.com/HC-super/Word2Vec-Learning/blob/main/imdb_lstm.py)

[LSTM](https://github.com/HC-super/Word2Vec-Learning/blob/main/imdb_lstm.py)

[CNN_LSTM](https://github.com/HC-super/Word2Vec-Learning/blob/main/imdb_cnn_lstm.py)

[stacked_LSTM（堆叠循环神经网络)](https://github.com/HC-super/Word2Vec-Learning/blob/main/imdb_stacked_lstm.py)

[bidirectional_lstm（双向循环神经网络）](https://github.com/HC-super/Word2Vec-Learning/blob/main/imdb_bidirectional_lstm.py)

[胶囊网络capsule](https://github.com/HC-super/Word2Vec-Learning/blob/main/imdb_capsulenet.py)

[自注意力模型](https://github.com/HC-super/Word2Vec-Learning/blob/main/imdb_attention_lstm.py)

[Tokenization](https://github.com/HC-super/Word2Vec-Learning/blob/main/Tokenization.py)

[TF-IDF](https://github.com/HC-super/Word2Vec-Learning/blob/main/TfidfTransformer.py)

[平均向量](https://github.com/HC-super/Word2Vec-Learning/blob/main/Vector%20Averaging.py)

[K-means](https://github.com/HC-super/Word2Vec-Learning/blob/main/K-means.py)

[TOC]

# NLP learning

## 词向量

## Word2vec



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

word2vec是谷歌开发的用于试图理解词之间情感和意义的工具，也是一种相较于one-hot的降维操作。工作方法类似于深层方法（Deep Approaches）如递归神经网络和深层神经网络，但它的计算效率更高。它可以利用**向量**来表示词与词之间的关系和相似度（余弦相似度），越有区分度的词越远离空间分为窗口词和中心词（前后文预测中间词skip-gram，中间词预测前后文CBOW）以计算词向量。

skip-gram（跳字模型）的网络结构如下：

![image-20210531114959080](https://tva1.sinaimg.cn/large/008i3skNgy1gr1us5kkmcj30aw0cimz0.jpg)

（skip-gram可以理解为通过一个词来预测上下文）每个词被分为两个d维向量用来计算条件概率。假设这个词的索引为i，则当它为中心词时的的向量表示为$v_i\in\mathbb{R}^d$时，背景词向量表示为$\boldsymbol{u}_i\in\mathbb{R}^d$。设中心词$w_c$在词典中索引为$c$，背景词$w_o$在词典中索引为$o$，给定中心词生成背景词的条件概率可以通过**对向量内积做softmax运算**而得到：

$$P(w_o \mid w_c) = \frac{\text{exp}(\boldsymbol{u}_o^\top \boldsymbol{v}_c)}{ \sum_{i \in \mathcal{V}} \text{exp}(\boldsymbol{u}_i^\top \boldsymbol{v}_c)},$$

其中词典索引集$\mathcal{V} = \{0, 1, \ldots, |\mathcal{V}|-1\}$。假设给定一个长度为$T$的文本序列，设时间步$t$的词为$w^{(t)}$。假设给定中心词的情况下背景词的生成相互独立，当背景窗口大小为$m$时，跳字模型的**似然函数**即**给定任一中心词生成所有背景词的概率**。

CBOW是用上下文来预测这个词。网络结构如下

![image-20210531120551378](https://tva1.sinaimg.cn/large/008i3skNgy1gr1usiv1gdj30a00cidh8.jpg)

> 统计学较深度学习更快

> 倒排索引：通过对文章进行索引以便尽快找到相关文档，CBOW类似于完形填空。通过上下文来预测中心词。而我们通常会用skip-gram，因为其效果更好。

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

MLP（Multi-layer Perceptron）Densenet 密集连接 

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

##  CNN

称为卷积神经网络

卷积：对图片一块像素的信息进行处理

利用卷积核对图像进行特征提取



输出就叫做feature map，有多少个卷积核就有多少个feature map，卷积核具体的值不需要进行设置，系统自己会设置

池化（pooling）：下采样。平均池化和最大值池化。池化的作用：1. 减少参数量，尽量保持原数据的原始特征	2. 防止过拟合 	3. 可以为卷积神经网络带来**平移不变性** （会丢失距离信息）

Embedding：

embedding 在深度学习中经常和manifold（流形）搭配使用

流行假设：**Manifold Hypothesis**（流形假设）。流形假设是指“自然的原始数据是低维的流形嵌入于(embedded in)原始数据所在的高维空间”。深度学习的任务就是把高维原始数据（图像，句子）映射到低维流形，使得高维的原始数据被映射到低维流形之后变得可分，而这个映射就叫嵌入（Embedding）。比如Word Embedding，就是把**单词组成的句子映射到一个表征向量。**

![image-20210522103538598](https://tva1.sinaimg.cn/large/008i3skNgy1gr1uu15ircj31km0heh08.jpg)

![image-20210522103633516](https://tva1.sinaimg.cn/large/008i3skNgy1gr1vwc7n88j31xy0lwgor.jpg)

![image-20210522103725729](https://tva1.sinaimg.cn/large/008i3skNgy1gr1uu6h0hkj31sg0rqwqj.jpg)

利用卷积核对底层特征进行抽取，符合卷积核的特征会被提取到feature map中

二分类问题（sigmoid），多分类问题（softmax）

激活函数——ReLU 函数：![img](https://bkimg.cdn.bcebos.com/formula/ae9d12662d9e1073200f081659ff7ea3.svg)线性整流作为神经元的激活函数，定义了该神经元在线性变换![img](https://bkimg.cdn.bcebos.com/formula/fb2b2510cb8c97bb4b8ee347317804a4.svg)之后的非线性输出结果。换言之，对于进入神经元的来自上一层神经网络的输入向量![img](https://bkimg.cdn.bcebos.com/formula/40482bf9a174030a55e50aa416fb29af.svg)，使用线性整流激活函数的神经元会输出![img](https://bkimg.cdn.bcebos.com/formula/24175eeaf4905a7acc3025fa7f3f660f.svg)至下一层神经元或作为整个神经网络的输出（取决现神经元在网络结构中所处位置）。

七层卷积神经网络（用于手写数字识别)

充分利用CNN的局部感受野，权值共享（每次卷积用同一个卷积核）、下采样的特点保证平移、缩放、变形的不变性

![image-20210522120644403](https://tva1.sinaimg.cn/large/008i3skNgy1gr1uub7uhgj31s20fq7u5.jpg)

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gr1uuca7sej31hw0u0hdu.jpg" alt="image-20210522121756245" style="zoom:200%;" />

每个神经元仅与输入神经元的一块区域连接

上采样：转置卷积，将输入的信息扩大。
![image-20210522174612691](https://tva1.sinaimg.cn/large/008i3skNgy1gr1uues2z4j31cg0syaoi.jpg)

![image-20210523120856168](https://tva1.sinaimg.cn/large/008i3skNgy1gr1uuvaz3sj31jk0ss4mn.jpg) 

## LSTM（长短时循环）和GRU（门控单元）

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



## capsule-net



paper:	Transformers are RNNS ：Fast Autoregressive Transformers with Linear Attention

​				Long Range Arena: A benchmark for Efficent Transformers

Capsule: output a vector

Neuron:output a value

Geoffrey Hinton深度学习创始人之一

它的论文[CapsuleNet](https://arxiv.org/abs/1710.09829)

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



## 自注意力模型

Self-Attention

on the rellationship between SelfAttention and CNN<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gr1upisvs0j30py0c83zm.jpg" alt="image-20210531182927927" style="zoom:33%;" />

Self-Attention VS. RNN

RNN中的vector无法同时产生需传递

Self-Attention中的vector可同时产生

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gr1w78qfnlj30vo0nm49q.jpg" alt="image-20210531204118549" style="zoom:50%;" />

文章：Attention is all you need



<img src="https://tva1.sinaimg.cn/large/008i3skNly1gr1w8optimj30v40n6tcg.jpg" alt="image-20210531204241513" style="zoom:50%;" />

$\alpha$为两向量之间的关联性，称为attention score

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gr1wa46zsqj60v40nkwl302.jpg" alt="image-20210531204404096" style="zoom:50%;" />

这里不一定是softmax，其他的激活函数都可以

![image-20210531204646128](https://tva1.sinaimg.cn/large/008i3skNly1gr1wcxbc8tj30vm0ncjz6.jpg)

哪一个向量的v大，最后的b就会dominate那个向量

![](https://i.loli.net/2021/07/22/V5XBf4TpGuoeWs3.png)



$b_i$是同时产生的

![image-20210531204956261](https://tva1.sinaimg.cn/large/008i3skNly1gr1wg83rqnj30v60ni79m.jpg)

$ I $​​为self-attention的input：$[a^1,a^2,a^3,a^4]$​​​

$O$​为output

self-attention唯一需要学习的是$W^q,W^k,W^v$

**multi-head self-attention 是说有几个不同的$q,k,v$​​​​** 

![image-20210722145459259](https://i.loli.net/2021/07/22/4mJowFzgtaC9qVQ.png)



$b^i$可以同时算出

self-attention的变形——multi-head Self-attention（2 heads as example）

![image-20210604163139306](https://tva1.sinaimg.cn/large/008i3skNgy1gr6bgq92dkj311q0t44cv.jpg)

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gr6bhgxqj2j30f4078myg.jpg" alt="image-20210604163223410" style="zoom:50%;" />

head的数目为超参数

对于self-attention来说input的顺序不重要

![image-20210604163504330](https://tva1.sinaimg.cn/large/008i3skNgy1gr6bk9v18pj30be0eygo5.jpg)

原论文中$e^i$​不是训练出来的，他的目的是标识input的顺序

No position information in self-attention 

换句话说可以使每一个$x^i$添加一个one-hot向量$p^i$​

​	**Positional Encoding** 

each position has a unique positional vector $e^i$​​​ 



<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gr6bp97dg5j311a0dkgqs.jpg" alt="image-20210604163950916" style="zoom:50%;" />





<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gr6br0zs2mj311e0r2drb.jpg" alt="image-20210604164133386" style="zoom: 50%;" />

几种不同的position编码形式

![image-20210722151303430](https://i.loli.net/2021/07/22/1Rv2zpHMTj9lWGA.png)





![image-20210604164220216](https://tva1.sinaimg.cn/large/008i3skNgy1gr6brtpnqgj31180r6apo.jpg)

![img](https://tva1.sinaimg.cn/large/008i3skNly1gr78hviyfog30hs0fqhco.gif)

bidirectional RNN可以用self-attention取代掉

decoder的selfattention也可以用selfattention替换掉



![image-20210604170230662](https://tva1.sinaimg.cn/large/008i3skNly1gr6cctl8bmj313o0ru1f5.jpg)

layer norm通常用于rnn

![image-20210604170404805](https://tva1.sinaimg.cn/large/008i3skNly1gr6ceg1ac3j31320nedx1.jpg)

![image-20210604170614899](https://tva1.sinaimg.cn/large/008i3skNly1gr6cgpcxjuj313e0rsqgq.jpg)
![image-20210604174713757](https://tva1.sinaimg.cn/large/008i3skNly1gr6dnd2v5tj312s0syaxu.jpg)

![image-20210604174958212](https://tva1.sinaimg.cn/large/008i3skNly1gr6dq7erdyj60zq0og7f502.jpg)

![image-20210604175027268](https://tva1.sinaimg.cn/large/008i3skNly1gr6dqp93y4j611u0pytma02.jpg)



## seq2seq



![image-20210603110211882](https://tva1.sinaimg.cn/large/008i3skNly1gr4wblxvwkj310o0omb0a.jpg)



![image-20210602183225023](https://tva1.sinaimg.cn/large/008i3skNly1gr43pr6hc2j31630u0ka0.jpg)

1. one-hot
2. 词汇归类（word class）

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gr43rvare5j30pc0l0n1w.jpg" alt="image-20210602183427475" style="zoom:50%;" />



3. word Embedding 每个词用向量表示，词汇意思相近的向量距离较近

但是一个词汇会有不同的意思。

![image-20210602183652822](https://tva1.sinaimg.cn/large/008i3skNly1gr43ue9qrrj31240lugxs.jpg)

bank的不同的token但是是同一个Embedding（不同的词义却用一个向量来表示了）

前两个bank是银行的意思，后两个是岸的意思

如 The hospitial has its own blood **bank**. 这里的bank指的是血库。*是否该归类于第三个意思？*



所以我们希望每一个的token有不同的Embedding .根据token的上下文来判断

![image-20210602184434052](https://tva1.sinaimg.cn/large/008i3skNly1gr442e8r4xj31260ny4fg.jpg)

contextualized word embedding

rnn会考虑到前文所说的，cnn可以用多个卷积层来扩大感受野

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gr605yuilrj30z40rwh3k.jpg" alt="image-20210604100042497" style="zoom:50%;" />

![image-20210604101434138](https://tva1.sinaimg.cn/large/008i3skNgy1gr60kd74e9j311y0rs7k6.jpg)

image caption gereration

将图像用cnn生成一个vector之后喂到rnn中

![image-20210604101902140](https://tva1.sinaimg.cn/large/008i3skNgy1gr60p0lyc9j31080qck9m.jpg)

encoder的rnn和decoder的rnn可以相同也可以不同

seq2seq的learning

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gr60qve47ij30jc0l2jwb.jpg" alt="image-20210604102049068" style="zoom:50%;" />

由于模型中没有考虑到之前是否说过Hi，要将之前的回答考虑进来我们可以这样设计模型：

![image-20210604104129451](https://tva1.sinaimg.cn/large/008i3skNly1gr61cdhmp9j30xc0nu7kf.jpg)

machine translation

attention-based model

$h_i$为RNN hidden layer 的output

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gr61kt91hdj311s0qo7js.jpg" alt="image-20210604104936033" style="zoom: 33%;" />

![image-20210604105141596](https://tva1.sinaimg.cn/large/008i3skNly1gr61mze9bij30yw0m4k1i.jpg)



$c^0$是Decoder input，这里的softmax不一定要加，不加有可能更好。

 ![image-20210604105514106](https://tva1.sinaimg.cn/large/008i3skNly1gr61qnxm9yj310q0r6nb1.jpg)



![image-20210604112822723](https://tva1.sinaimg.cn/large/008i3skNgy1gr62p5tuvxj30k00mkn5u.jpg)

![image-20210604113644461](https://tva1.sinaimg.cn/large/008i3skNgy1gr62xvj1tdj310u0pwna6.jpg)

document产生句向量后，对每一个句子通过mach和q进行运算后产生$\alpha$  score，之后通过weight sum得到extracted information输入到DNN中得到answer。

memory network还有更复杂的版本算match的部分和抽information的部分不见得是一样的。

![image-20210604130100767](https://tva1.sinaimg.cn/large/008i3skNgy1gr65dl6zdoj31280qsduc.jpg)

![image-20210604130325307](https://tva1.sinaimg.cn/large/008i3skNgy1gr65g1yyvsj313e0to16g.jpg)



相同颜色可以看成一样也可以看成不一样

这里讲的memory machine是在memory的基础上做attention，然后从memory里面将information extract出来⬆️



下面要讲的neural Turing machine不只是可以从memory里面得到信息，并且其还能将信息写到memory里面然后在之后的time step中得出来

![image-20210604130921198](https://tva1.sinaimg.cn/large/008i3skNgy1gr65m8iq1aj310i0mo16i.jpg)

![image-20210604151256714](https://tva1.sinaimg.cn/large/008i3skNgy1gr696u0gqqj30zk0r27ha.jpg)

f为任意的如RNN或者lstm等网络

![image-20210604152759499](https://tva1.sinaimg.cn/large/008i3skNgy1gr69mhjqp4j31bq0tik6e.jpg)

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gr69nuarb7j31200rotnj.jpg" alt="image-20210604152918088" style="zoom:33%;" />



transformer非常知名的应用seq2seq model

Bert 是非监督学习的transformer

RNN不容易被平行化，所谓平行化就是每一个rnn的unit，其输出都是有先后顺序的，不是同时产生的，而cnn可以做到类似并行处理的效果

![image-20210604161323166](https://tva1.sinaimg.cn/large/008i3skNgy1gr6axpow9pj31120tah4d.jpg)

cnn比较容易平行化

self-attention来取代CNN，输入和输出和rnn相同

self-attention和CNN的关系

![image-20210722152951421](https://i.loli.net/2021/07/22/ROeYdTP2lHxSkjv.png)

![image-20210722153008783](https://i.loli.net/2021/07/22/QXp53Bzsdg1ahYC.png)

![image-20210722155031200](https://i.loli.net/2021/07/22/8LJuFakdWPAhKmf.png)

CNN good for less data

self-attention good for more data

[上面的⬆️文献](https://arxiv.org/pdf/2010.11929.pdf)

![image-20210722165524280](https://i.loli.net/2021/07/22/FRZolyux68CMmIj.png)

self-attention的运算效率由于parallel使得其更有效率

[Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236)



<img src="https://i.loli.net/2021/07/22/y3RhP1Gc79mFMlx.png" alt="image-20210722170401488" style="zoom:50%;" />



self-attention最早用在transformer中，可以理解为广义的transformer是一种self-attention

传统的transformer模型庞大，训练的parameter多，speed变快往往需要压缩模型，但会带来performance变低







![](https://tva1.sinaimg.cn/large/008i3skNgy1gr6b1id8waj313k0q24dz.jpg)

multi-class classification 是说从众多class里面选一个

multi-label classification 是说同一个object属于不同的labels

![image-20210722182619594](https://i.loli.net/2021/07/22/KT46ud2XkCDoxbg.png)

### Encoder

Encoder要做的事情,就是**给一排向量，输出另外一排向量**

![image-20210429205911444](https://i.loli.net/2021/07/22/vdS5O6j7wohpiHW.png)

![image-20210429210126607](https://gitee.com/unclestrong/deep-learning21_note/raw/master/imgbed/image-20210429210126607.png)







## ELMO

Embedding from Language Model（ELMO）//  RNN based language mode

通过许多句子来训练

https://arxiv.org/abs/1802.05365

![image-20210602191510519](https://tva1.sinaimg.cn/large/008i3skNly1gr44y8duyhj31160gkamd.jpg)

通过学习预测下一个token

将RNN的hidden layer拿出来就是该词通过上下文输出的embedding

考虑前文：前向RNN

考虑下文：反向RNN

![image-20210602191632248](https://tva1.sinaimg.cn/large/008i3skNly1gr44zni108j31200r6h71.jpg)



 ![image-20210602193406287](https://tva1.sinaimg.cn/large/008i3skNly1gr45hxsmc0j31140soarj.jpg)

通过RNN学习得到两个embedding后，根据要做的任务来确定出$\alpha_1$和$\alpha2$这两个参数得到蓝色的向量。

## BERT （Bidirectional Encoder Representations from Transformers）

BERT = Encoder of Transformer

Learned from  a large amount of text without annotation

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gr4606kbbgj30l00e6whi.jpg" alt="image-20210602195138523" style="zoom:25%;" />

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gr460u59zjj60jc0re45w02.jpg" alt="image-20210602195216120" style="zoom:50%;" />

在中文处理中BERT通常用字，由于常用字是的数量较词来说少很多。

 BERT的训练

方法：

1. Masked LM

遮盖一个句子里面15%的词，之后进行预测，对于给预测的词embedding

如果两个词添在同一个地方没有违和感呢那么它们就有类似的embedding

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gr4770kvj5j60f20f878b02.jpg" alt="image-20210602203248829" style="zoom: 50%;" />

2. 下一个句子预测

[SEP]：the boundary of two sentences

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gr47m7z1bqj31200kyjzz.jpg" alt="image-20210602204725544" style="zoom:50%;" />

[CLS]：the position that outputs classification results

BERT 内部使用attention实现的

方法1和方法2同时使用。

如何应用bert：

![image-20210602205816524](https://tva1.sinaimg.cn/large/008i3skNly1gr47xi8qngj30n40ga0yd.jpg)

linear classifier从头学，bert微调

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gr47zxxfvpj30oa0gm7ag.jpg" alt="image-20210602210037045" style="zoom:67%;" />

对于每一个词作分类⬆️

![image-20210602210230897](https://tva1.sinaimg.cn/large/008i3skNly1gr481x4pnnj310s0pwgyj.jpg)

给一个前提和假设推出是否正确或者未知⬆️

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gr48a37713j311w0qie0s.jpg" alt="image-20210602211021541" style="zoom:67%;" />

基于提取的回答系统。

给一个文档和问题的embedding表示，给出的答案在A中。

红蓝vector的维度和bert输出的黄色维度相同。

黄色vector和红色vector作dot product.

![image-20210602211639417](https://tva1.sinaimg.cn/large/008i3skNly1gr48goysf7j31220putb3.jpg)

![image-20210602211713456](https://tva1.sinaimg.cn/large/008i3skNly1gr48h88ps8j312a0pw15a.jpg)

红色算出s蓝色算出e=3，最后的答案就是

如果e落在s前面则此题无解。

Enhanced Representation through knowledge integration（ERNIE）

designed for Chinese

字为单位，一次盖一个词汇

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gr4vj3mrk1j30og0f4aib.jpg" alt="image-20210603103448495" style="zoom:50%;" />

https://arxiv.org/abs/1905.05950
https://openreview.net/pdf?id=SJzSgnRcKX



![image-20210603105646199](https://tva1.sinaimg.cn/large/008i3skNly1gr4w5yttztj31160q6x53.jpg)

multilingual BERT

![image-20210603125313010](https://tva1.sinaimg.cn/large/008i3skNly1gr4zj5vkzfj30ty0k8dqt.jpg) 

给定英文文章的分类可以输出中文文章的分类

Generative Pre-Training (GPT)

![image-20210603125607600](https://tva1.sinaimg.cn/large/008i3skNly1gr4zm542qhj311o0pgk8e.jpg)

是transformer的decoder

![image-20210603125940920](https://tva1.sinaimg.cn/large/008i3skNly1gr4zpuh2krj31300qck3v.jpg)

Zero-shot Learning?

![image-20210603130402755](https://tva1.sinaimg.cn/large/008i3skNly1gr4zueer9aj314f0u0nnt.jpg)

![image-20210603130605979](https://tva1.sinaimg.cn/large/008i3skNly1gr4zwj38u7j30jq03itap.jpg)

![image-20210603130614315](https://tva1.sinaimg.cn/large/008i3skNly1gr4zwoayfrj316x0u04qp.jpg)

attention总会聚焦到第一个词

![image-20210603160323326](https://tva1.sinaimg.cn/large/008i3skNgy1gr551092bnj313w0u0k8j.jpg)



同样的token就是同样的Embedding——word2vec，glove

pre-train model

将每个token用embedding向量表示。

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gr56v36oncj30f40dejsr.jpg" alt="image-20210603170654336" style="zoom:33%;" />

也可以通过输入字母来输出这个单词所对应的向量预测这个单词——FastText

中文：image喂到CNN

ELMo，BERT等为contextualized Embedding（吃一整个句子再给每一个token embedding）

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gr5717rks1j312u0igadq.jpg" alt="image-20210603171247998" style="zoom:50%;" />

通常这样的model有很多layers

Tree-based model （考虑了文法的因素）

<img src="../../../Library/Application Support/typora-user-images/image-20210603174643738.png" alt="image-20210603174643738" style="zoom:50%;" />

![image-20210603181254893](https://tva1.sinaimg.cn/large/008i3skNgy1gr58rt3rj9j310y0t07w2.jpg)



albert 12层和24层都是一样的参数

![image-20210603181815949](https://tva1.sinaimg.cn/large/008i3skNgy1gr58xcic7jj31400qywnr.jpg)

![image-20210603182107580](https://tva1.sinaimg.cn/large/008i3skNgy1gr590bres5j313y0q8asc.jpg)













## Encoder-Decoder

Encoder-Decoder 通常称作 编码器-解码器，是深度学习 中常⻅的模型框架。

Encoder-Decoder 并不是 一个具体的模型，而是一个通用的框架。

Decoder部分可以是任意文字，语音，图像以及视频数据，模型也可以结合CNN，RNN，LSTM





nlp

1. Translation

2. summarization

3. Chat-bot
4. Question Answering

![image-20210602165919140](https://tva1.sinaimg.cn/large/008i3skNly1gr410vtgqbj30tc0kwte6.jpg)

文法剖析树

![image-20210602170148297](https://tva1.sinaimg.cn/large/008i3skNly1gr413gp6flj30pk0jowhi.jpg)

隐状态可以理解为隐状态的活性值$h_t$，部分文献称为状态（state）或者（Hidden State）



