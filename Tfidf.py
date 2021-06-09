# %%

import pandas as pd

from bs4 import BeautifulSoup

import re

from nltk.corpus import stopwords

stops = set(stopwords.words("english"))
train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3)


def review_to_words(raw_review):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review, features="html.parser").get_text()
    #
    # 2. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set

    #    stops = set(stopwords.words("english"))
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    #
    # 6. Join the words back into one string separated by space,
    # and return the result.
    return " ".join(meaningful_words)


num_testreviews = train["review"].size  # 评论数

clean_train_reviews = []
for i in range(0, num_testreviews):  # 由于处理较慢，分阶阶段打印处理进度
    # If the index is evenly divisible by 1000, print a message
    if (i + 1) % 1000 == 0:
        print("Review %d of %d\n" % (i + 1, num_testreviews))
    clean_train_reviews.append(review_to_words(train["review"][i]))

print("Creating the bag of words...\n")
# %%

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(analyzer='word', max_features=5000)

train_data_features = vectorizer.fit_transform(clean_train_reviews)
train_data_features = train_data_features.toarray()
# %%
num_testreviews = len(test["review"])

clean_test_reviews = []

print("Cleaning and parsing the test set movie reviews...\n")

for i in range(0, num_testreviews):
    if ((i + 1) % 1000 == 0):
        print("Review %d of %d\n" % (i + 1, num_testreviews))
    clean_review = review_to_words(test["review"][i])
    clean_test_reviews.append(clean_review)

test_data_features = vectorizer.fit_transform(clean_test_reviews)
test_data_features = test_data_features.toarray()
# %%
print("Fitting a random forest to labeled training data...")
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=300, n_jobs=-1)
forest = forest.fit(train_data_features, train["sentiment"])

# Test & extract results
result = forest.predict(test_data_features)

# Write the test results
output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
output.to_csv("tfidf.csv", index=False, quoting=3)
