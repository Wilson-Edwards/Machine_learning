import random
from sklearn import datasets
from sklearn import model_selection
import numpy as np
import matplotlib.pyplot as plt

def loadDataSet():
    '''
    函数说明：加载数据集，并统一返回matrix格式数据
    :return: 
    :param x_train: x训练数据集
    :param x_test: x验证数据集
    :param y_train: y训练数据集
    :param y_test: y验证数据集
    '''
    housing = datasets.load_svmlight_file("C:/users/lin78/desktop/housing_scale.txt")
    x_train, x_test, y_train, y_test = model_selection.train_test_split(housing[0], housing[1],
                                                        test_size=0.33, random_state=42)
    x_train = x_train.A   #将scipy中的matrix转换为ndarray ,注意使用type函数检验数据类型
    x_test = x_test.A
    insert_column = np.ones(x_train.shape[0])   #增添的一列，偏重bias
    x_train = np.c_[x_train, insert_column]
    insert_another = np.ones(x_test.shape[0])
    x_test = np.c_[x_test, insert_another]
    #-------将ndarray类型转换为matrix--------------
    x_train = np.mat(x_train)
    x_test = np.mat(x_test)
    y_train = np.mat(y_train).T
    y_test = np.mat(y_test).T
    #-----------------------------------------------
    return x_train, x_test, y_train, y_test

def SGD(x_train, y_train, numIter = 100):
    '''
    函数说明：随机梯度下降
    :param x_train: x训练数据集
    :param y_train: y训练数据集
    :param numIter: 迭代次数，默认2次，
    经测试numIter为2时效果比较好,不过仍然具有随机性，数值不稳定
    :return: 更新后的权重
    '''
    row, col = np.shape(x_train)
    weights = np.ones((col, 1))   #权重，初始化为0
    weights_array = np.ones_like(weights)  #权重数组，保存每次的权重
    weights = np.mat(weights)
    print("type(weights): ", type(weights))
    print("weights: ", weights.shape)
    for j in range(numIter):
        dataIndex = list(range(row))  # 每一次梯度下降所选取的行的下标
        for i in range(row):
            rate = 1/(1.0 + j + i) + 0.0001 #每次迭代后rate会减小，但永远会比0.01大、
            randIndex = int(random.uniform(0, len(dataIndex))) #该次迭代所选取的行下标
            y_predict = x_train[randIndex] * weights   #求该行样本预测y值
            y_true = y_train[randIndex]
            grad = x_train[randIndex].T * (y_predict - y_train[randIndex])   #求该行梯度
            weights = weights - rate * grad    #更新权重
            # del(dataIndex[randIndex])
        weights_array = np.append(weights_array, weights, axis = 1)
    weights_array = weights_array.A
    weights_array = np.delete(weights_array, 0, 1) #删除第一列数据
    return weights_array

def loss(x_train, x_test, y_train, y_test, weights_array):
    '''
    函数说明：计算L2范式的train集上的loss和test集上的loss
    :param x_train: x训练数据集
    :param x_test: x验证数据集
    :param y_train: y训练数据集
    :param y_test: y验证数据集
    :param weights_array: 训练得到的权重数组
    :return: 
    loss_train: 训练集的L2损失函数值
    loss_val: 验证集的L2损失函数值
    '''
    row, col = weights_array.shape
    loss_train_array = np.array([])
    loss_val_array = np.array([])
    for i in range(col):
        weights = np.mat(weights_array[:, i].reshape(row, 1))

        loss_train = y_train.T * y_train - 2 * weights.T * x_train.T * y_train\
                 + weights.T * x_train.T * x_train * weights
        loss_train = loss_train / (2 * y_train.shape[0])
        loss_val = y_test.T * y_test - 2 * weights.T * x_test.T * y_test \
               + weights.T * x_test.T * x_test * weights
        loss_val = loss_val / (2 * y_test.shape[0])
        '''
        loss_train = np.sum(np.square(y_train - x_train * weights)) / (2 * y_train.shape[0])
        loss_val = np.sum(np.square(y_test - x_test * weights)) / (2 * y_test.shape[0])
        '''
        loss_train_array = np.append(loss_train_array, loss_train[0][0])
        loss_val_array = np.append(loss_val_array, loss_val[0][0])

    return loss_train_array, loss_val_array


def plot_loss_and_numIter( loss_train_array, loss_val_array):
    '''
    :param loss_train_array: loss_train的数组
    :param loss_val_array:  loss_val的数组
    :return: 
    '''
    #-------------shape----------------------
    plt.plot(loss_train_array, label = "loss_train")
    plt.plot(loss_val_array, label = "loss_val")
    plt.xlabel("number of iteration")
    plt.ylabel("loss")
    plt.title("SGD")
    plt.legend() #显示标签
    plt.show()





if __name__ == '__main__':
    x_train, x_test, y_train, y_test = loadDataSet()
    print("y_train: ", y_train.shape[0])
    numIter = 100
    weights_array = SGD(x_train, y_train, numIter)
    loss_train_array, loss_val_array = loss(x_train, x_test, y_train, y_test, weights_array)
    print("loss_train_array:\n", loss_train_array)
    print("loss_val_array:\n", loss_val_array)
    plot_loss_and_numIter(loss_train_array, loss_val_array)





