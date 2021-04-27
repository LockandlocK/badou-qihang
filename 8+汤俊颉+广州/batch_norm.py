import torch.nn as nn
import numpy as np
import torch as t
class MyBn:
    def __init__(self, momentum, eps, num_featrues):
        """
        :param momentum:追踪样本的整体均值和方差的动量 the value used for the running_mean and running_var computation. Can be set to None for cumulative moving average (i.e. simple average). Default: 0.1
        :param eps: 防止数值计算错误  a value added to the denominator for numerical stability.
        :param num_featrues: 特征数量  CC from an expected input of size (N, C, L)(N,C,L) or LL from input of size (N, L)(N,L)
        """
        # 对每个batch的mean和var进行追踪统计
        self._running_mean = 0
        self._running_var = 1
        #
        self._momentum = momentum
        self._eps = eps
        self._beta = np.zeros(shape = (num_featrues, ))
        self._gamma = np.zeros(shape = (num_featrues, ))

    def batch_norm(self, x):
        """
        bn 向前传播
        :param x: 数据
        :return: 输出
        """
        x_mean = x.mean(axis=0)
        x_var = x.var(axis=0)
        print(x_mean,"x_mean")
        print(x_var,"x_var")

        # 对应running_mean的更新公式
        self._running_mean = (1-self._momentum)*x_mean + self._momentum*self._running_mean
        self._running_var = (1-self._momentum)*x_var + self._momentum*self._running_var
        x_hat = (x-x_mean)/np.sqrt(x_var+self._eps)
        y = self._gamma*x_hat + self._beta
        return  y



data = np.array([
    [1,2],
    [1,3],
    [1,4]]).astype(np.float32)

print(data.shape)
# create batch normal layer by nn tool
torch_bn = nn.BatchNorm1d(num_features=2)
torch_data = t.from_numpy(data)
torch_bn_output = torch_bn(torch_data)
# diy bn
my_bn = MyBn(momentum=0.01, eps=0.001,num_featrues=2)

my_bn._beta = torch_bn.bias.detach().numpy()
my_bn._gamma = torch_bn.weight.detach().numpy()
my_bn_output = my_bn.batch_norm(data, )
print(torch_bn_output)
print(my_bn_output)

