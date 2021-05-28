# max len = 56
from __future__ import print_function
from __future__ import absolute_import
import os
import sys
import logging

from keras.layers import Dropout, Embedding, LSTM, Dense
import matplotlib.pyplot as plt
from keras.utils import plot_model
import pickle
import numpy as np
import pandas as pd

from tensorflow import keras
from keras.models import Sequential

# maxlen = 56
batch_size = 32
nb_epoch = 10
hidden_dim = 128

test = pd.read_csv("corpus/imdb/testData.tsv", header=0,
                   delimiter="\t", quoting=3)


def get_idx_from_sent(sent, word_idx_map):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
        else:
            x.append(1)

    return x


def make_idx_data(revs, word_idx_map, maxlen=60):
    """
    Transforms sentences into a 2-d matrix.
    """
    X_train, X_test, X_dev, y_train, y_dev = [], [], [], [], []
    for rev in revs:
        sent = get_idx_from_sent(rev['text'], word_idx_map)
        y = rev['y']

        if rev['split'] == 1:
            X_train.append(sent)
            y_train.append(y)
        elif rev['split'] == 0:  # 验证集
            X_dev.append(sent)
            y_dev.append(y)
        elif rev['split'] == -1:
            X_test.append(sent)

    X_train = keras.preprocessing.sequence.pad_sequences(np.array(X_train), maxlen=maxlen)
    X_dev = keras.preprocessing.sequence.pad_sequences(np.array(X_dev), maxlen=maxlen)
    X_test = keras.preprocessing.sequence.pad_sequences(np.array(X_test), maxlen=maxlen)

    y_train = keras.utils.to_categorical(np.array(y_train))
    y_dev = keras.utils.to_categorical(np.array(y_dev))

    return [X_train, X_test, X_dev, y_train, y_dev]


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    logging.info('loading data...')
    pickle_file = os.path.join('pickle', 'imdb_train_val_test.pickle3')
    revs, W, word_idx_map, vocab, maxlen = pickle.load(open(pickle_file, 'rb'))
    logging.info('data loaded!')

    X_train, X_test, X_dev, y_train, y_dev = make_idx_data(revs, word_idx_map, maxlen=maxlen)

    n_train_sample = X_train.shape[0]
    logging.info("n_train_sample [n_train_sample]: %d" % n_train_sample)

    n_test_sample = X_test.shape[0]
    logging.info("n_test_sample [n_train_sample]: %d" % n_test_sample)

    len_sentence = X_train.shape[1]  # 200
    logging.info("len_sentence [len_sentence]: %d" % len_sentence)

    max_features = W.shape[0]
    logging.info("num of word vector [max_features]: %d" % max_features)

    num_features = W.shape[1]  # 400
    logging.info("dimension of word vector [num_features]: %d" % num_features)

    # Keras Model
    # this is the placeholder tensor for the input sequence
    print('Build model...')
    model = Sequential()

    # model.add(keras.Input(shape=(maxlen,), dtype='int32'))

    model.add(Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen,
                        mask_zero=True, weights=[W], trainable=False))

    # embedded = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen, weights=[W],
    # trainable=False) (sequence)
    model.add(Dropout(0.25))

    # LSTM
    model.add(LSTM(hidden_dim, dropout=0.2, recurrent_dropout=0.25))

    # GRU
    # hidden = GRU(hidden_dim, recurrent_dropout=0.25) (embedded)

    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    print('plot model...')

    plot_model(model, to_file='imdb_lstm.png', show_shapes=True, show_layer_names=True)  # 网络可视化
    print('Train...')

    history = model.fit(X_train, y_train, validation_data=[X_dev, y_dev], batch_size=batch_size, epochs=nb_epoch)

    y_pred = model.predict(X_test, batch_size=batch_size)
    y_pred = np.argmax(y_pred, axis=1)

    result_output = pd.DataFrame(data={"id": test["id"], "sentiment": y_pred})

    # Use pandas to write the comma-separated output file
    # result_output.to_csv("./result/bi-lstm.csv", index=False, quoting=3)

    result_output.to_csv("./result/lstm.csv", index=False, quoting=3)
    score, acc = model.evaluate(X_dev, y_dev, batch_size=batch_size)

    print('Test score:', score)
    print('Test accuracy:', acc)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # 绘制训练 & 验证的损失值
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
