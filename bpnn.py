from typing import List
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, Activation
from keras.optimizers import SGD
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time

learning_rate = 0.0001
epochs = 50
batch_size = 32

import matplotlib
matplotlib.use('TkAgg')

class BPNN:
    __slots__ = ["learning_rate", "epochs", "batch_size", "activation", "units", "model", "score", "speed", "fpr",
                 "trainHistory","x_train", "x_test", "y_train", "y_test"]

    def __init__(self, learning_rate: float = learning_rate, epochs: int = epochs,
                 batch_size: int = batch_size, activation=None, units=None, model=None,
                 x_train=None, x_test=None, y_train=None, y_test=None):
        '''
        :param learning_rate: 学习率
        :param epochs: 训练的轮数
        :param batch_size: 训练时每组数据的大小
        :param activation: 各隐藏层定义的激活函数，以数组的形式存储，默认为['relu', 'relu', 'relu', 'relu']
        :param units: 各隐藏层的神经元数量,以数组的形式存储，默认为[256,128,128,64]
        :param model: 定义的模型
        :param x_train: 训练数据集
        :param x_test:  测试数据集
        :param y_train: 训练数据集的标签
        :param y_test:  测试数据集的标签
        '''
        if activation is None:
            activation = ['relu', 'relu', 'relu', 'relu']
        if units is None:
            units = [256, 128, 128, 64]
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.activation = activation
        self.units = units
        self.model = model
        self.score = [0, 0]  # 模型测试的得分，第一个为loss，后面的看设置的评价指标，在这里第二个表示accuracy
        self.speed = None  # 推理速度
        self.fpr = None     # 假阳性
        self.trainHistory = None  # 模型训练的历史信息，包括每个epochs的loss和accuracy
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def readdata(self, data_path: str = './data/prepared.csv') -> None:
        column_name = ['id']
        for i in range(205):
            column_name.append(i)
        column_name.append('label')
        df = pd.read_csv(data_path, header=None, names=column_name, sep=',')
        y = np.array(df[df.columns[-1]], dtype=int)
        x = df.drop(df.columns[[0, -1]], axis=1)
        x = np.array(x)
        # 将数据划分为训练集和测试集
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.05, random_state=42)

    def init_model(self, x_train):
        '''
        :param x_train: 传入训练的数据
        :return:
        '''
        # 定义 Keras 模型
        model = Sequential()
        # 加入隐藏层
        # 第一层需要给到数据的shape input_dim=X_train.shape[1]，也可以input_shape=(X_train.shape[1],)
        model.add(Dense(self.units[0], activation=self.activation[0], input_dim=x_train.shape[1]))
        # 设置后面的隐藏层
        for i in range(1, len(self.units)):
            model.add(Dense(self.units[i], activation=self.activation[i]))
        model.add(Dropout(0.5))
        model.add(Dense(4, activation='softmax'))  # 由于需要分类4个类别，最后一层需要输出 4
        opt = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=opt, loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])  # 使用多分类，损失函数为sparse_categorical_crossentropy
        self.model = model

    def init_model_with_leakyrelu(self, x_train):
        '''
        :param x_train: 传入训练的数据
        :return:
        '''
        # 定义 Keras 模型
        model = Sequential(name="BP Neural Network")
        # 加入隐藏层
        # 第一层需要给到数据的shape input_dim=X_train.shape[1]，也可以input_shape=(X_train.shape[1],)
        model.add(Dense(self.units[0], input_dim=x_train.shape[1]))
        model.add(keras.layers.LeakyReLU(alpha=0.05))
        # 设置后面的隐藏层
        for i in range(1, len(self.units)):
            model.add(Dense(self.units[i]))
            model.add(keras.layers.LeakyReLU(alpha=0.05))
        model.add(Dropout(0.5))
        model.add(Dense(4, activation='softmax'))  # 由于需要分类4个类别，最后一层需要输出 4
        opt = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=opt, loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])  # 使用多分类，损失函数为sparse_categorical_crossentropy
        self.model = model

    def training(self):
        # print(type(x_train)) # <class 'pandas.core.frame.DataFrame'>
        # 训练 Keras 模型
        self.trainHistory = self.model.fit(self.x_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size,
                                           validation_data=(self.x_test, self.y_test))
        # 查看模型层及参数
        # print(self.model.summary())
        # 评估 Keras 模型
        # 计算推理时间
        start = time.perf_counter()
        pre = self.model.predict(self.x_test)
        end = time.perf_counter()
        self.speed = end-start

        # 计算fpr
        fp = 0
        tn = 0
        for i in range(len(pre)):
            max_col = 0
            for j in range(len(pre[i])):
                if (pre[i][j] > pre[i][max_col]):
                    max_col = j
            if (self.y_test[i] != 0 and max_col == 0):
                fp += 1
            if (self.y_test[i] != 0 and max_col == self.y_test[i]):
                tn += 1
        self.fpr = fp / (tn + fp)
        self.score = self.model.evaluate(self.x_test, self.y_test)
        print('Test loss:', self.score[0])
        print('Test accuracy:', self.score[1])
        print(f'speed:{self.speed}')
        print(f'FPR:{self.fpr}')

    def savemodel(self, file_name: str = 'BPNeuralNetwork'):
        self.model.save('./pretrained/' + file_name + '.keras')


# BP神经网络测试main函数
if __name__ == '__main__':
    bpnn = BPNN()  # 实例化对象
    bpnn.readdata()  # 读取数据
    bpnn.init_model(x_train=bpnn.x_train)
    bpnn.training()  # 训练模型
    bpnn.savemodel()  # 保存模型
    # history中有四个参数'val_loss','val_accuracy','loss','accuracy'，val是validation验证的值，不带val的才是training的值
    # print(bpn.trainHistory.history['loss'])
    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(13, 5))  # ncols，nrows用于划分子图，figsize为画布的长宽比
    axs[0].plot(bpnn.trainHistory.epoch, bpnn.trainHistory.history['loss'], label="loss")
    axs[0].set_ylabel('loss')
    axs[0].set_xlabel('epoch')
    axs[0].set_title('loss')
    axs[0].legend()

    axs[1].plot(bpnn.trainHistory.epoch, bpnn.trainHistory.history['accuracy'], label="accuracy")
    # axs[1].set_aspect(3)  # 压缩x轴与y轴的长度比，传入3表示x轴长度为y轴的3倍
    axs[1].set_ylabel('accuracy')
    axs[1].set_xlabel('epoch')
    axs[1].set_title('accuracy')
    axs[1].legend()
    fig.suptitle('BP Neural Network plots')
    plt.show()
