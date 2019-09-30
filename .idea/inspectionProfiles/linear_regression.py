import torch
import torchsnooper   #可以使用torchsnooper进行调试
from torch.autograd import Variable
import matplotlib.pyplot as plt
from IPython import display
torch.manual_seed(8000) #设置随机数种子
def get_fake_data(batch_size = 8):
    '''
    生成数据，生成的数据相当于真实值
    :param batch_size: 返回值x和y的数组大小
    :return: 
    '''
    x = torch.rand(batch_size) * 20
    y = x * 2 + (1 + torch.randn(batch_size)) * 3
    return x, y

w = Variable(torch.rand(1), requires_grad = True)
b = Variable(torch.zeros(1), requires_grad = True)
lr = 0.001 #学习率
for i in range(8001):
    x, y = get_fake_data()
    x, y = Variable(x), Variable(y)
    y_pred = x * w + b
    loss = 0.5 * (y_pred - y) ** 2
    loss = loss.sum()
    loss.backward()
    w.data.sub_(lr * w.grad.data)
    b.data.sub_(lr * w.grad.data)
    w.grad.data.zero_()
    b.grad.data.zero_()
    if i % 1000 == 0:
        print("w: ", w)
        print("b: ", b)
        display.clear_output(wait=True)
        x = torch.arange(0, 20, dtype=torch.float32)
        y = x * w + b
        plt.plot(x.detach().numpy(), y.detach().numpy())
        x2, y2 = get_fake_data(batch_size=20)
        plt.scatter(x2, y2)
        plt.xlim(0, 20)
        plt.xlim(0, 41)
        plt.show()
        plt.pause(0.5)