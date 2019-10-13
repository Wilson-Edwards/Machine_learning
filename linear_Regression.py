from sklearn import datasets
from sklearn import model_selection
import numpy as np

def get_weight(x_train, y_train):
    '''
    函数说明：计算权重参数
    :param x_train: x训练数据集
    :param y_train: y训练数据集
    :return: 
    '''
    xMat = np.mat(x_train)
    yMat = np.mat(y_train).T
    xTx = xMat.T * xMat  # 根据推导的公示计算回归系数
    if np.linalg.det(xTx) == 0.0:
        print("矩阵为奇异矩阵,不能求逆")
        return
    weight = xTx.I * (xMat.T * yMat)
    return weight

def loss(x_train, x_test, y_train, y_test):
    '''
    函数说明：计算L2范式的train集上的loss和test集上的loss
    :param x_train: x训练数据集
    :param x_test: x验证数据集
    :param y_train: y训练数据集
    :param y_test: y验证数据集
    :return: 
    loss_train: 训练集的L2损失函数值
    loss_val: 验证集的L2损失函数值
    '''
    weight = get_weight(x_train, y_train)
    # 将ndarray类型转换为matrix
    x_train = np.mat(x_train)
    x_test = np.mat(x_test)
    y_train = np.mat(y_train).T
    y_test = np.mat(y_test).T
    #-----------------------------
    loss_train = np.sum(np.square(y_train - x_train * weight)) / (2 * y_train.shape[0])
    loss_val = np.sum(np.square(y_test - x_test * weight)) / (2 * y_test.shape[0])
    return loss_train, loss_val


if __name__ == '__main__':
    housing = datasets.load_svmlight_file("C:/users/lin78/desktop/housing_scale.txt")
    x_train, x_test, y_train, y_test = model_selection.train_test_split(housing[0], housing[1],
                                                        test_size=0.33, random_state=42)
    x_train = x_train.A   #将scipy中的matrix转换为ndarray ,注意使用type函数检验数据类型
    x_test = x_test.A
    insert_column = np.ones(x_train.shape[0])   #增添的一列，偏重b
    x_train = np.c_[x_train, insert_column]
    insert_another = np.ones(x_test.shape[0])
    x_test = np.c_[x_test, insert_another]
    loss_train, loss_val = loss(x_train, x_test, y_train, y_test)
    print("loss_train: ", loss_train)
    print("loss_val: ", loss_val)
