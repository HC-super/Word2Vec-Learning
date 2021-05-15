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

print("Read %d labeled train reviews, %d labeled test reviews, " \
      "and %d unlabeled reviews\n" % (train["review"].size,
                                      test["review"].size, unlabeled_train["review"].size))

num_features = 300  # 300个特征

clean_train_reviews = []
for review in train["review"]:
    clean_train_reviews.append(review_to_wordlist(review, remove_stopwords=True))
print("Creating average feature vecs for test reviews")

clean_test_reviews = []
for review in test["review"]:
    clean_test_reviews.append(review_to_wordlist(review, remove_stopwords=True))

# %%
from sklearn.cluster import KMeans
import time

start = time.time()  # Start time

# Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an
# average of 5 words per cluster
word_vectors = model.wv.vectors
num_clusters = int((word_vectors.shape[0] / 5))
# Initalize a k-means object and use it to extract centroids
kmeans_clustering = KMeans(n_clusters=num_clusters)
idx = kmeans_clustering.fit_predict(word_vectors)
# 将word2vec的词向量聚类 sizeof(word_vectors) =(16490,300)
# 16490个单词每个单词对应300个特征
# 输出的是一个单词对应的聚类号（一维数组）

# Get the end time and print how long the process took
end = time.time()
elapsed = end - start
print("Time taken for K Means clustering: ", elapsed, "seconds.")

word_centroid_map = dict(zip(model.wv.index2word, idx))
# 将得到的聚类结合单词得到每个单词的聚类zip之后成为元组再转化为dict
# %%
# For the first 10 clusters
for cluster in range(0, 10):
    #
    # Print the cluster number
    print("\nCluster %d" % cluster)
    #
    # Find all of the words for that cluster number, and print them out
    words = []
    mapvalue = list(word_centroid_map.values())
    for i in range(0, len(mapvalue)):
        if mapvalue[i] == cluster:
            words.append(list(word_centroid_map.keys())[i])
    print(words)


# %%
def create_bag_of_centroids(wordlist, word_centroid_map):
    #
    # 聚类中心数
    num_centroids = max(word_centroid_map.values()) + 1
    #
    # Pre-allocate the bag of centroids vector (for speed)
    bag_of_centroids = np.zeros(num_centroids, dtype="float32")
    # 创建一个聚类数大小的0数组，统计
    #
    # 如果单词在reviews中将单词的索引所对应的 bag_of_centroids 数组对应数组加一
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
    #
    # Return the "bag of centroids"
    return bag_of_centroids


# %%

# Pre-allocate an array for the training set bags of centroids (for speed)
train_centroids = np.zeros((train["review"].size, num_clusters), dtype="float32")

# Transform the training set reviews into bags of centroids
counter = 0
for review in clean_train_reviews:  # clean train reviews是一个评论组成的列表，对每个评论的单词其聚类进行统计
    train_centroids[counter] = create_bag_of_centroids(review, word_centroid_map)
    # review作为words传入create_bag_of_centroids
    counter += 1

# Repeat for test reviews
test_centroids = np.zeros((test["review"].size, num_clusters),
                          dtype="float32")

counter = 0
for review in clean_test_reviews:
    test_centroids[counter] = create_bag_of_centroids(review, word_centroid_map)
    counter += 1
# Fit a random forest and extract predictions
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=100)

# Fitting the forest may take a few minutes
print("Fitting a random forest to labeled training data...")
forest = forest.fit(train_centroids, train["sentiment"])
result = forest.predict(test_centroids)

# Write the test results
output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
output.to_csv("BagOfCentroids.csv", index=False, quoting=3)
