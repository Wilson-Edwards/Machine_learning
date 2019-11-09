import random
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

def loadDataSet():
    '''
    函数说明：加载数据集，并统一返回matrix格式数据
    :return: 训练数据集，训练标签，测试数据集，测试标签
    :param train_data: x训练数据集
    :param test_data: x验证数据集
    :param train_label: y训练数据集
    :param y_test: y验证数据集
    '''
    train_data, train_label = datasets.load_svmlight_file("a9a.txt")
    test_data, test_label = datasets.load_svmlight_file("a9at.txt", n_features = 123)
    train_data = train_data.A   #将scr.matrix转换为np.array
    test_data = test_data.A
    train_label[train_label == -1] = 0
    test_label[test_label == -1] = 0
    #-----增添一列-----------
    insert_column = np.ones(train_data.shape[0])   #增添的一列，偏重bias
    train_data = np.c_[train_data, insert_column]
    insert_another = np.ones(test_data.shape[0])
    test_data = np.c_[test_data, insert_another]
    return train_data, train_label, test_data, test_label

def sigmoid(inX):
    '''
    函数说明：sigmoid激活函数
    :param inX: 一行数据
    :return: sigmoid函数值
    '''
    return 1.0 / (1 + np.exp(-inX))

def batch_SGD(train_data, train_label, numIter = 5, minibatch_size = 2):
    '''
    函数说明：小批量梯度下降
    :param train_data: x训练数据集
    :param train_label: y训练数据集
    :param numIter: 迭代次数，默认10次，
    :param minibatch_size: 小批量梯度下降的样本数，默认为2
    :return: 权重数组
    '''
    row, col = np.shape(train_data)
    weights = np.ones(col).reshape(col, 1)   #权重，初始化为1
    weights_array = np.ones_like(weights) #权重数组，保存每次迭代的权重
    for j in range(numIter):
        for i in range(row):
            rate = 4/(1.0 + j + i)  #每次迭代后rate会减小
            randIndex = int(random.uniform(0, row - minibatch_size + 1)) #该次迭代所选取的行下标
            #计算选取minibatch_size行时的梯度
            train_data_temp = train_data[randIndex: randIndex + minibatch_size]
            train_label_temp = train_label[randIndex : randIndex + minibatch_size].reshape(minibatch_size, 1)
            y_predict = sigmoid(train_data_temp @ weights)
            error = y_predict - train_label_temp
            grad = train_data_temp.T @ error
            weights = weights - rate * grad
        weights_array = np.append(weights_array, weights, axis = 1)
    weights_array = np.delete(weights_array, 0, 1) #删除第一列无用数据
    return weights_array

def cross_entropy_loss(train_data, train_label, test_data, test_label, weights_array):
    '''
    函数说明：交叉熵损失函数
    :param train_data: 训练数据集
    :param train_label: 训练标签
    :param test_data: 测试数据集
    :param test_label: 测试标签
    :param weights_array: 权重数组
    :return: loss_train_array: train数据集的loss数组
              loss_val_array: test数据集的loss数组
    '''
    weights_row, weights_col = weights_array.shape
    loss_train_array = np.array([])
    loss_val_array = np.array([])
    for i in range(weights_col):
        weights = weights_array[:, i].reshape(weights_row)
        loss_train = -(np.log(sigmoid(train_data @ weights)).T @ train_label +
                       np.log(1 - sigmoid(train_data @ weights)).T @ (1 - train_label))
        loss_train = loss_train / train_label.shape[0]  # 除以数据的行数
        loss_val = -(np.log(sigmoid(test_data @ weights)).T @ test_label +
                     np.log(1 - sigmoid(test_data @ weights)).T @ (1 - test_label))
        loss_val = loss_val / test_label.shape[0]
        loss_train_array = np.append(loss_train_array, loss_train)
        loss_val_array = np.append(loss_val_array, loss_val)
    return loss_train_array, loss_val_array


def plot_loss_and_numIter(loss_train_array, loss_val_array):
    '''
    函数说明：绘制迭代次数和损失函数值得关系曲线
    :param loss_train_array: train数据集的loss数组
    :param loss_val_array: test数据集的loss数组
    :return: 无
    '''
    plt.plot(loss_train_array, label = "loss_train")
    plt.plot(loss_val_array, label = "loss_val")
    plt.xlabel("number of iteration")
    plt.ylabel("loss")
    plt.title("logistic_regression")
    plt.legend() #显示标签
    plt.show()

if __name__ == '__main__':
    train_data, train_label, test_data, test_label = loadDataSet()
    #---------可以通过更改numIter和minibatch_size来获得不同的曲线效果
    numIter = 50; minibatch_size = 2
    #----------------------------------------------------------------
    weights_array = batch_SGD(train_data, train_label, numIter, minibatch_size)
    loss_train_array, loss_val_array = cross_entropy_loss(train_data, train_label,
                                                          test_data, test_label, weights_array)
    print("loss_train:\n", loss_train_array)
    print("loss_val:\n", loss_val_array)
    plot_loss_and_numIter(loss_train_array, loss_val_array)