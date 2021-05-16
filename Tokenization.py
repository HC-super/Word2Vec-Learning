# %%

import pandas as pd

from bs4 import BeautifulSoup

import re

from nltk.corpus import stopwords

stops = set(stopwords.words("english"))
train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3)


def review_to_words(raw_review):
    review_text = BeautifulSoup(raw_review, features="html.parser").get_text()

    letters_only = re.sub("[^a-zA-Z]", " ", review_text)

    words = letters_only.lower().split()

    meaningful_words = [w for w in words if not w in stops]

    return " ".join(meaningful_words)


num_reviews = train["review"].size  # 评论数

clean_train_reviews = []
for i in range(0, num_reviews):  # 由于处理较慢，分阶阶段打印处理进度

    if (i + 1) % 1000 == 0:
        print("Review %d of %d\n" % (i + 1, num_reviews))
    clean_train_reviews.append(review_to_words(train["review"][i]))

print("Creating the bag of words...\n")

from sklearn.feature_extraction.text import CountVectorizer

# 文字袋工具

vectorizer = CountVectorizer(analyzer="word",
                             tokenizer=None,
                             preprocessor=None,
                             stop_words=None,
                             max_features=5000)

train_data_features = vectorizer.fit_transform(clean_train_reviews)

train_data_features = train_data_features.toarray()

# %% 随机森林算法
print("Training the random forest...")
from sklearn.ensemble import RandomForestClassifier

# 随机森林中100个决策树
forest = RandomForestClassifier(n_estimators=100)  # 拟合train数据集

forest = forest.fit(train_data_features, train["sentiment"])
# %%清洗test数据集
test = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3)

print(test.shape)

num_reviews = len(test["review"])

clean_test_reviews = []

print("Cleaning and parsing the test set movie reviews...\n")

for i in range(0, num_reviews):
    if ((i + 1) % 1000 == 0):
        print("Review %d of %d\n" % (i + 1, num_reviews))
    clean_review = review_to_words(test["review"][i])
    clean_test_reviews.append(clean_review)

# %%将测试集向量化
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

# %% 利用随机森林来预测并输出文件
result = forest.predict(test_data_features)

output = pd.DataFrame(data={"id": test["id"], "sentiment": result})

output.to_csv("Bag_of_Words_model.csv", index=False, quoting=3)
