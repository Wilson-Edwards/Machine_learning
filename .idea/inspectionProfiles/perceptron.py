import torch
import torchsnooper
from torch import nn
from torch.autograd import Variable

class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.w = nn.Parameter(torch.randn(in_features, out_features))
        self.b = nn.Parameter(torch.randn(out_features))
    def forward(self, x):
        x = x.mm(self.w)
        return x + self.b

class Perceptron(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        nn.Module.__init__(self)
        self.layer1 = Linear(in_features, hidden_features)
        self.layre2 = Linear(hidden_features, out_features)
    def forward(self, x):
        x = self.layer1(x)
        x = torch.sigmoid(x)
        return self.layer2(x)

if __name__ == '__main__':
    perceptron = Perceptron(3, 4, 1)
    for name, param in perceptron.named_parameters():
        print(name, param.size())

