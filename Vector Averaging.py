# %%
import re
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

train = pd.read_csv("labeledTrainData.tsv", header=0,  # 读入标记训练集
                    delimiter="\t", quoting=3)

test = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3)  # 读入测试集

unlabeled_train = pd.read_csv("unlabeledTrainData.tsv", header=0,  # 读入未标记训练集
                              delimiter="\t", quoting=3)

# 确认被读取评论的数量
print("Read %d labeled train reviews, %d labeled test reviews, "
      "and %d unlabeled reviews\n" % (train["review"].size,
                                      test["review"].size, unlabeled_train["review"].size))


# %%
# 导入各种模块进行字符串清理

def review_to_wordlist(review, remove_stopwords=False):
    # 是否移除停止词由remove_stopwords决定，
    # 本函数主要考虑移除HTML标识和非字母元素

    # 函数将评论转换为单词序列，可选择删除停止词。返回单词列表。

    # 1. Remove HTML
    review_text = BeautifulSoup(review, features="html.parser").get_text()
    #
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
        words = set(words)
    #
    # 5. Return a list of words
    return words


# Download the punkt tokenizer for sentence splitting
import nltk.data

# Load the punkt tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


# Define a function to split a review into parsed sentences
# 定义一个函数，将评论拆分为已解析的句子

def review_to_sentences(review, tokenizer, remove_stopwords=False):
    # 函数将评论拆分为已解析的句子。返回句子列表，其中每个句子都是单词列表
    #
    # 1. 使用NLTK标记器将段落拆分为句子
    raw_sentences = tokenizer.tokenize(review.strip())
    #
    # 2. 在每个句子上循环
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append(review_to_wordlist(raw_sentence,
                                                remove_stopwords))

    # Return the list of sentences (each sentence is a list of words,so this returns a list of lists
    return sentences


# %%
sentences = []  # Initialize an empty list of sentences
# 将未标记和标记的训练集都加入了训练
# 下面两个for循环将评论分成句子
print("Parsing sentences from training set")
for review in train["review"]:
    sentences += review_to_sentences(review, tokenizer)

print("Parsing sentences from unlabeled set")
for review in unlabeled_train["review"]:
    sentences += review_to_sentences(review, tokenizer)
    # 需要一些时间
# %%
# Import the built-in logging module and configure it so that Word2Vec creates nice output messages
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

# Set values for various parameters
num_features = 300  # Word vector dimensionality    词向量维数
min_word_count = 40  # Minimum word count   最小字数
num_workers = 16  # Number of threads to run in parallel 并行运行的线程数
context = 10  # Context window size 上下文窗口大小
downsampling = 1e-3  # Downsample setting for frequent words    频繁词的下采样设置

# Initialize and train the model (this will take some time)
from gensim.models import word2vec

print("Training model...")
model = word2vec.Word2Vec(sentences, workers=num_workers,
                          size=num_features, min_count=min_word_count,
                          window=context, sample=downsampling)
# %%
# 如果不打算进一步训练模型，那么调用init_sims将使模型的内存效率大大提高。
model.init_sims(replace=True)

# It can be helpful to create a meaningful model name and
# 创建一个有意义的模型名和
# save the model for later use. You can load it later using Word2Vec.load()
# 保存模型供以后使用。您可以稍后使用Word2Vec.load（）加载它
model_name = "300features_40minwords_10context"
model.save(model_name)

# %%
# Load the model that we created in Part 2
from gensim.models import Word2Vec

model = Word2Vec.load("300features_40minwords_10context")
# type(model.syn0)
# model.syn0.shape
print(type(model.wv.vectors))
print(model.wv.vectors.shape)

# %%
import numpy as np  # Make sure that numpy is imported


def makeFeatureVec(words, model, num_features):
    # 对单个的评论进行平均向量化
    # 所谓单个评论的平均向量化是指在word2vec中生成的对应每个单词的向量后，在传入的单个评论中将所有单词的向量相加再平均
    featureVec = np.zeros((num_features,), dtype="float32")
    #
    nwords = 0.
    #
    # Index2word is a list that contains the names of the words in the model's vocabulary. Convert it to a set, for speed
    # Index2word是一个列表，其中包含模型词汇表中单词的名称。为了速度，把它转换成一组
    index2word_set = set(model.index2word)
    #
    # Loop over each word in the review and, if it is in the model's vocaublary, add its feature vector to the total
    # 在评论中循环每个单词，如果它在模型的词汇表中，则将其特征向量添加到总数中
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1. # 如果单词在index2word的集合中，那么将nwords+1
            featureVec = np.add(featureVec, model[word])

    #
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec, nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    # 对评论集中的所有评论进行平均向量化
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")
    for review in reviews:

        if counter % 1000 == 0:
            print("Review %d of %d" % (counter, len(reviews)))
        reviewFeatureVecs[counter] = makeFeatureVec(review, model,
                                                    num_features)
        counter = counter + 1
    return reviewFeatureVecs


import pandas as pd

# Read data from files


# Verify the number of reviews that were read (100,000 in total)
print("Read %d labeled train reviews, %d labeled test reviews, " \
      "and %d unlabeled reviews\n" % (train["review"].size,
                                      test["review"].size, unlabeled_train["review"].size))

num_features = 300  #300个特征

clean_train_reviews = []
for review in train["review"]:
    clean_train_reviews.append(review_to_wordlist(review, remove_stopwords=True))

trainDataVecs = getAvgFeatureVecs(clean_train_reviews, model.wv, num_features)

print("Creating average feature vecs for test reviews")

clean_test_reviews = []
for review in test["review"]:
    clean_test_reviews.append(review_to_wordlist(review, remove_stopwords=True))

testDataVecs = getAvgFeatureVecs(clean_test_reviews, model.wv, num_features)

# Fit a random forest to the training data, using 100 trees
from sklearn.ensemble import RandomForestClassifier

# %%
forest = RandomForestClassifier(n_estimators=100)

print("Fitting a random forest to labeled training data...")
forest = forest.fit(trainDataVecs, train["sentiment"]) # 监督式学习

# Test & extract results
result = forest.predict(testDataVecs)

# Write the test results
output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
output.to_csv("Word2Vec_AverageVectors.csv", index=False, quoting=3)
