#!/usr/bin/env python
# coding: utf-8

# In[2]:


from pathlib import Path

import numpy as np
import pandas as pd

train = pd.read_csv("corpus/imdb/labeledTrainData.tsv", header=0,
                    delimiter="\t", quoting=3)
test = pd.read_csv("corpus/imdb/testData.tsv", header=0,
                   delimiter="\t", quoting=3)
train_texts = train["review"].tolist()
train_labels = train["sentiment"].tolist()
test_texts = test["review"].tolist()


from sklearn.model_selection import train_test_split

train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)

from transformers import DistilBertTokenizerFast

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

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

# test_labels = [1]*len(test1)

test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings)
))

from transformers import TFDistilBertForSequenceClassification

model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')


# In[3]:


optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)

model.compile(optimizer=optimizer, loss=model.compute_loss, metrics=['acc'])  # can also use any keras loss fn


# In[4]:


history = model.fit(train_dataset.batch(5), epochs=5, batch_size=5)


# In[5]:


model.evaluate(val_dataset.batch(5), batch_size=5)


# In[6]:


labels_pred = model.predict(test_dataset.batch(5))


# In[9]:


from matplotlib import pyplot as plt

plt.plot(history.history['acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
# 绘制训练 & 验证的损失值
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# In[10]:


y = labels_pred.logits

y_pred = np.argmax(y,axis = 1)


# In[15]:


y


# In[12]:


y_pred


# In[13]:


result_output = pd.DataFrame(data={"id": test["id"], "sentiment": y_pred})

result_output.to_csv("bert.csv", index=False, quoting=3)


# In[14]:


model.save("TFDistilBertForSequenceClassification")


# In[16]:


labels_pred_train = model.predict(train_dataset.batch(5))


# In[24]:


y_train = labels_pred_train.logits
y_pred_train = np.argmax(y,axis = 1)
y_pred_train[4]


# In[18]:


train_labels


# In[ ]:




