layer normalization做的事情,比batch normalization更简单一点

![image-20210724200802572](https://i.loli.net/2021/07/24/6mjcLwylf3W5axX.png)

**the model is too simple** :

find a needle in a haystack but there is no needle

![image-20210724194701527](https://i.loli.net/2021/07/24/gaziRIOZ8eGdFHf.png)

solution: redesign your model to make it more flexible.

![image-20210724194839463](https://i.loli.net/2021/07/24/6Wnjmox5rPI4LRh.png)

optimization Issue:

A needle is in a haystack ...... just cannot find it.

**究竟是模型偏差还是优化问题？**

![image-20210724195823496](https://i.loli.net/2021/07/24/cNHAUMSV1kF7hpD.png)

模型flexible更强，但效果不好：可能是optimization问题

 如果是optimization issue 

- gaining the insights from comparison
- Start from shallower networks(or other models), which are easier to optimize.

或者使用SVM等统计学习方法

- If deeper networks do not obtain smaller loss on training data, the there is optimization issue.

![image-20210724200603511](https://i.loli.net/2021/07/24/MfluqsEANahO7eT.png)

- Solution: more powerful optimization technology (next lecture) 

  **Overfitting:**

![image-20210724201434022](https://i.loli.net/2021/07/24/1wmtOI6FXhC8dSf.png)

![image-20210724201638726](https://i.loli.net/2021/07/24/ZQlw8A4iEOJjqR6.png)

Data augmentation (you can do that in HWs)

![image-20210724201920099](https://i.loli.net/2021/07/24/nVt9gZBGMdKsYqj.png)

augmentation 要有道理，否则机器会学到奇怪的东西

![image-20210724202334942](https://i.loli.net/2021/07/24/mBdtLwFb5ZUu14J.png)

对于函数不能太约束，否则就会出现model bias

![image-20210724202904942](https://i.loli.net/2021/07/24/sPty7VwLkK6Bh1f.png)

![image-20210724203010503](https://i.loli.net/2021/07/24/sYQJbaZjuhfCkSx.png)

随着模型越来越负责，training loss会变小但是Testing loss会增大。

![image-20210724203440214](https://i.loli.net/2021/07/24/kYIEx3XrtlvRGMs.png)

只根据public data中的testing set选择mse最小的模型，可能会在private data的testing set中得到很差的结果。

[What will happen？](http://www.chioka.in/how- to-select-your-final-models- in-a-kaggle-competitio/) this explains why machine usually beats human on benchmark corpora. 

![image-20210724204521614](https://i.loli.net/2021/07/24/I6BkD9jscCYmfvp.png)

交叉验证集很重要，只关注testing set来挑模型又会使模型陷入对testing set的过拟合中

N-fold Cross Validation

![image-20210724204921361](https://i.loli.net/2021/07/24/maWdY6QUsyTw4CG.png)

![image-20210724205128010](https://i.loli.net/2021/07/24/LaRpkwJ7InHPOCu.png)

Mismatch

训练集和测试集的分布不同。

Most HWs do not have this problem, except HW11.

​              

