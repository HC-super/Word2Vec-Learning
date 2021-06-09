#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


import pandas as pd

train = pd.read_csv("/kaggle/input/others/labeledTrainData.tsv", header=0,
                    delimiter="\t", quoting=3)
test = pd.read_csv("/kaggle/input/testset/testData.tsv", header=0,
                   delimiter="\t", quoting=3)

train_texts = train["review"].tolist()
train_labels = train["sentiment"].tolist()
test_texts = test["review"].tolist()

from sklearn.model_selection import train_test_split

train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)

from transformers import RobertaTokenizerFast

tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

import tensorflow as tf

train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    train_labels
))
val_dataset = tf.data.Dataset.from_tensor_slices((
    dict(val_encodings),
    val_labels
))


test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings)
))

# %%
from transformers import TFRobertaForSequenceClassification

model = TFRobertaForSequenceClassification.from_pretrained('roberta-base')
# %%
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)

model.compile(optimizer=optimizer, loss=model.compute_loss,
              metrics=tf.metrics.SparseCategoricalAccuracy())  # can also use any keras loss fn
# %%

history = model.fit(train_dataset.batch(10), epochs=1)

# In[12]:


evalu = model.evaluate(val_dataset.batch(5))

# In[13]:


model.save("TFRobertaForSequenceClassification")

# In[14]:


labels_pred = model.predict(test_dataset.batch(5))

# In[15]:


print(labels_pred)

# In[16]:


y = labels_pred.logits

# In[17]:


y_pred = np.argmax(y, axis=1)
print(y_pred)

# In[18]:


result_output = pd.DataFrame(data={"id": test["id"], "sentiment": y_pred})

result_output.to_csv("TFroberta.csv", index=False, quoting=3)
