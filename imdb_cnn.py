# max len = 56
from __future__ import absolute_import
from __future__ import print_function

import logging
import os
import sys

import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Embedding, Dropout, Convolution1D, MaxPooling1D, Flatten, Dense, Activation
from matplotlib import pyplot as plt
from tensorflow import keras
from keras.utils import plot_model
import pickle

batch_size = 32  # 每次梯度更新的样本数。
nb_epoch = 10  # 迭代次数
hidden_dim = 128  # 隐层单元个数

kernel_size = 3  # 卷积核大小
nb_filter = 60  # 卷积核个数

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
        elif rev['split'] == 0:
            X_dev.append(sent)
            y_dev.append(y)
        elif rev['split'] == -1:
            X_test.append(sent)

    X_train = keras.preprocessing.sequence.pad_sequences(np.array(X_train), maxlen=maxlen)
    X_dev = keras.preprocessing.sequence.pad_sequences(np.array(X_dev), maxlen=maxlen)
    X_test = keras.preprocessing.sequence.pad_sequences(np.array(X_test), maxlen=maxlen)
    # X_valid = sequence.pad_sequences(np.array(X_valid), maxlen=maxlen)
    y_train = keras.utils.to_categorical(np.array(y_train))
    y_dev = keras.utils.to_categorical(np.array(y_dev))
    # y_valid = np.array(y_valid)

    return [X_train, X_test, X_dev, y_train, y_dev]  # dev为验证集


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    logging.info('loading data...')
    # pickle_file = os.path.join('pickle', 'vader_movie_reviews_glove.pickle3')
    # pickle_file = sys.argv[1]
    pickle_file = os.path.join('pickle', 'imdb_train_val_test.pickle3')

    revs, W, word_idx_map, vocab, maxlen = pickle.load(open(pickle_file, 'rb'))  # 读取二进制文件
    # revs 为评论集合，W为74402*300的numpy词向量矩阵，word_idx_map为单词索引，vocab为单词出现次数，maxlen为1416
    logging.info('data loaded!')

    X_train, X_test, X_dev, y_train, y_dev = make_idx_data(revs, word_idx_map, maxlen=maxlen)

    n_train_sample = X_train.shape[0]  # 20072
    logging.info("n_train_sample [n_train_sample]: %d" % n_train_sample)

    n_test_sample = X_test.shape[0]  # 25000
    logging.info("n_test_sample [n_train_sample]: %d" % n_test_sample)

    len_sentence = X_train.shape[1]  # 1416
    logging.info("len_sentence [len_sentence]: %d" % len_sentence)

    max_features = W.shape[0]  # 77402
    logging.info("num of word vector [max_features]: %d" % max_features)

    num_features = W.shape[1]  # 300
    logging.info("dimension of word vector [num_features]: %d" % num_features)

    # Keras Model
    # this is the placeholder tensor for the input sequence

    model = Sequential()

    model.add(Embedding(input_dim=max_features, output_dim=num_features,
                        # input_dim 词典中的词语个数，output_dim词向量的维数
                        input_length=maxlen, weights=[W], trainable=False))

    model.add(Dropout(0.25))
    # Dropout 包括在训练中每次更新时，
    # 将输入单元的按比率随机设置为 0，
    # 这有助于防止过拟合。

    # convolutional layer 卷积层
    model.add(Convolution1D(filters=nb_filter,  # 卷积核数量  一维卷积
                            kernel_size=kernel_size,  # 卷积核大小
                            padding='valid',  # padding不填充
                            activation='relu',  # rulu激活函数
                            strides=1  # 步长
                            ))

    model.add(MaxPooling1D(pool_size=2))  # 使用最大池化
    model.add(Flatten())  # 展平一个张量后喂给全连接接神经网络

    # We add a vanilla hidden layer:
    model.add(Dense(120, activation='relu'))  # best: 120  #dense为全连接层神经元个数为70
    model.add(Dropout(0.25))  # best: 0.25 # 有效防止过拟合

    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    # categorical_crossentropy 输出张量与目标张量之间的分类交叉熵。
    # Adam 优化器。
    # 评价函数用于评估当前训练模型的性能。当模型编译后（compile），评价函数应该作为 metrics 的参数来输入。
    print('plot model...')

    plot_model(model, to_file='imdb_cnn.png', show_shapes=True, show_layer_names=True)  # 网络可视化

    history = model.fit(X_train, y_train, validation_data=[X_dev, y_dev], batch_size=batch_size,
                        epochs=nb_epoch)  # 开始训练

    y_pred = model.predict(X_test, batch_size=batch_size)
    y_pred = np.argmax(y_pred, axis=1)

    result_output = pd.DataFrame(data={"id": test["id"], "sentiment": y_pred})

    # Use pandas to write the comma-separated output file
    result_output.to_csv("./result/cnn.csv", index=False, quoting=3)
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
