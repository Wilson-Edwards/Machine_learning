import random
from sklearn import datasets
from sklearn.model_selection import train_test_split
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
    train_row = train_data.shape[0]
    test_row = test_data.shape[0]
    train_label = train_label.reshape(train_row, 1)  #将一维的label转换成二维
    test_label = test_label.reshape(test_row, 1)
    train_label[train_label == -1] = 0
    test_label[test_label == -1] = 0
    #-----增添一列-----------
    insert_column = np.ones(train_data.shape[0])   #增添的一列，偏重bias
    train_data = np.c_[train_data, insert_column]
    insert_another = np.ones(test_data.shape[0])
    test_data = np.c_[test_data, insert_another]
    return train_data, train_label, test_data, test_label


def batch_SGD(train_data, train_label, test_data, test_label,
              max_iteration = 100,batch_size = 1500, C = 0.5):
    '''
    函数说明：批量梯度下降
    :param train_data: 训练数据集
    :param train_label: 训练标签
    :param test_data:测试数据集
    :param test_label: 测试标签
    :param max_iteration: 最大迭代次数
    :param batch_size: 每次迭代所选取的批量
    :param C: 惩罚参数
    :return: loss_train_array:  训练数据集的loss数组
              loss_val_array: 验证数据集的loss数组
    '''
    train_row, train_col = train_data.shape
    weights = np.zeros((train_col, 1))
    loss_train_array = np.array([])
    loss_val_array = np.array([])
    for iteration in range(max_iteration):
        learning_rate = 1/(1+ iteration)
        #分割数据集提取出该次迭代的批量数据X_train和y_train
        X_train, train_data_notuse, y_train, train_label_notuse =\
            train_test_split(train_data, train_label, test_size = 1 - batch_size / train_label.size)

        error = 1 - y_train * (X_train @ weights)  #是否被正确分类，error>0不正确
        y_predict = np.where(error > 0, y_train, 0) #将y_train中正确分类的置0
        weights = weights - learning_rate * (weights - C * (X_train.T @ y_predict))
        toler_train = np.maximum(0, 1 - y_train * (X_train @ weights)) #松弛变量
        loss_train = weights.T @ weights + C * np.sum(toler_train)
        loss_train_array = np.append(loss_train_array, loss_train / X_train.shape[0])

        toler_val = np.maximum(0, 1 - test_label * (test_data @ weights))
        loss_val = weights.T @ weights + C * np.sum(toler_val)
        loss_val_array = np.append(loss_val_array, loss_val / test_data.shape[0])

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
    plt.title("SVM")
    plt.legend() #显示标签
    plt.show()
    return




if __name__ == '__main__':
    train_data, train_label, test_data, test_label = loadDataSet()
    loss_train_array, loss_val_array = batch_SGD(train_data, train_label, test_data, test_label)
    print("loss_train_array:\n", loss_train_array)
    print("loss_val_array:\n", loss_val_array)
    plot_loss_and_numIter(loss_train_array, loss_val_array)



