from pathlib import Path

import numpy as np
import pandas as pd

train = pd.read_csv("corpus/imdb/labeledTrainData.tsv", header=0,
                    delimiter="\t", quoting=3)
test = pd.read_csv("corpus/imdb/testData.tsv", header=0,
                   delimiter="\t", quoting=3)
train_texts = train["review"].tolist()
train_labels = train["sentiment"].tolist()
test = train["review"].tolist()

# %%
from sklearn.model_selection import train_test_split

train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)
# %%
from transformers import DistilBertTokenizerFast

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
# %%
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test, truncation=True, padding=True)
# %%
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


from transformers import TFDistilBertForSequenceClassification

model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)

model.compile(optimizer=optimizer, loss=model.compute_loss)  # can also use any keras loss fn
model.fit(train_dataset.shuffle(1000).batch(16), epochs=3, batch_size=16)
