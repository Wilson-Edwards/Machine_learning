import random
from sklearn import datasets
from sklearn import model_selection
import numpy as np

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

def SGD(x_train, y_train, numIter = 2):
    '''
    函数说明：随机梯度下降
    :param x_train: x训练数据集
    :param y_train: y训练数据集
    :param numIter: 迭代次数，默认2次，
    经测试numIter为2时效果比较好,不过仍然具有随机性，数值不稳定
    :return: 更新后的权重
    '''
    row, col = np.shape(x_train)
    maxCycles = 500  #随机梯度下降迭代500次
    weights = np.ones((col, 1))   #权重，初始化为1

    for j in range(numIter):
        dataIndex = list(range(row))   #每一次梯度下降所选取的行的下标
        for i in range(row):
            rate = 4/(1.0 + j + i) + 0.01 #每次迭代后rate会减小，但永远会比0.01大、
            randIndex = int(random.uniform(0, len(dataIndex))) #该次迭代所选取的行下标
            #计算选取一行时的梯度
            grad = x_train[randIndex].T * (y_train[randIndex] - x_train[randIndex] * weights)
            weights = weights - rate * grad
            del(dataIndex[randIndex])  #删除该次迭代所选取的行下标
    return weights

def loss(x_train, x_test, y_train, y_test, weights):
    '''
    函数说明：计算L2范式的train集上的loss和test集上的loss
    :param x_train: x训练数据集
    :param x_test: x验证数据集
    :param y_train: y训练数据集
    :param y_test: y验证数据集
    :param weights: 训练得到的权重
    :return: 
    loss_train: 训练集的L2损失函数值
    loss_val: 验证集的L2损失函数值
    '''
    loss_train = np.sum(np.square(y_train - x_train * weights)) / (2 * y_train.shape[0])
    loss_val = np.sum(np.square(y_test - x_test * weights)) / (2 * y_test.shape[0])
    return loss_train, loss_val


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = loadDataSet()
    weights = SGD(x_train, y_train)
    loss_train, loss_val = loss(x_train, x_test, y_train, y_test,weights)
    print("loss_train: ", loss_train)
    print("loss_val: ", loss_val)





