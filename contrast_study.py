from cnn import CNN
from bpnn import BPNN
from rnn import RNN
import matplotlib.pyplot as plt

contrast_list = []  # 存放每个模型的评价结果，用于对比
loss_convergence = []   # 存放每个模型训练的loss收敛过程
accuracy_convergence = []   # 存放每个模型训练的accuracy收敛过程
model_sequence = ["4 CovLayer CNN", "LeNet-5", "BP Neural Network", "RNN"]  # 本代码模型测试顺序
times = 1  # 模型重复实验次数

def training(model_object):
    metric = [0.0] * 4  # 4类评价指标
    training_loss = [0]*model_object.epochs
    training_accuracy = [0]*model_object.epochs
    for _ in range(times):
        model_object.training()
        # 每训练一次，记录评价指标
        metric[0] += model_object.trainHistory.history['accuracy'][-1]
        metric[1] += model_object.score[1]
        metric[2] += model_object.speed
        metric[3] += model_object.fpr
        # 每次训练的收敛过程记录
        training_loss = list(map(lambda x, y: x+y, training_loss, model_object.trainHistory.history['loss']))
        training_accuracy = list(map(lambda x, y: x + y, training_accuracy, model_object.trainHistory.history['accuracy']))
    # 重复实验times次，对参数取平均后存入数组
    contrast_list.append(list(map(lambda x: x / times, metric)))
    loss_convergence.append(list(map(lambda x: x / times, training_loss)))
    accuracy_convergence.append(list(map(lambda x: x / times, training_accuracy)))
    print(model_object.model.summary())

def cnn_training():
    cnn = CNN()  # 实例化对象
    cnn.readdata()  # 获取数据
    # ---------------------训练4covlayer----------------------
    cnn.init_4covlayer()  # 初始化模型
    training(cnn)
    cnn.savemodel("4CovLayer")  # 保存模型
    # ---------------------训练结束---------------------------
    # ---------------------训练lenet5-------------------------
    cnn.init_lenet5()  # 初始化模型
    training(cnn)
    cnn.savemodel("LeNet5")  # 保存模型
    # ---------------------训练结束---------------------------
def bpnn_training():
    bpnn = BPNN()
    bpnn.readdata()
    # ---------------------训练bp neural network----------------------
    bpnn.init_model(bpnn.x_train)  # 初始化模型
    training(bpnn)
    bpnn.savemodel("BPNeuralNetwork")  # 保存模型
    # ---------------------训练结束---------------------------

def rnn_training():
    rnn = RNN()
    rnn.readdata()
    rnn.initmodel(rnn.x_train)
    training(rnn)
    rnn.savemodel("RNN")

if __name__ == '__main__':
    cnn_training()
    bpnn_training()
    rnn_training()
    # 输出表头
    print(f"{' ':24}", end='')
    for i in ["training accuracy", "test accuracy", "inference speed", "FPR"]:
        print(f"{i:<22}", end='')
    print(' ')
    # 输出结果
    for i in range(len(contrast_list)):
        print(f"{model_sequence[i]:<22}: ", end='')
        for metric in contrast_list[i]:
            # <20表示左对齐20格宽度
            print(f"{metric:<22}", end='')
        print(' ')
    # 绘制收敛图
    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(13, 5))  # ncols，nrows用于划分子图，figsize为画布的长宽比
    for index, loss in enumerate(loss_convergence):
        axs[0].plot(range(len(loss)), loss, label=model_sequence[index])
    axs[0].set_ylabel('loss')
    axs[0].set_xlabel('epoch')
    axs[0].set_title('loss')
    axs[0].legend()

    for index, accuracy in enumerate(accuracy_convergence):
        axs[1].plot(range(len(accuracy)), accuracy, label=model_sequence[index])
    axs[1].set_ylabel('accuracy')
    axs[1].set_xlabel('epoch')
    axs[1].set_title('accuracy')
    axs[1].legend()
    fig.suptitle('Contrast Study')
    fig.show()