# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
os.chdir('./1-3/')
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import *

np.random.seed(1)


# %%
X, Y = load_planar_dataset()
print(f'X.shape = {X.shape}')
print(f'Y.shape = {Y.shape}')


# %%
plt.scatter(X[0], X[1], s=40, c=np.squeeze(Y), cmap=plt.cm.Spectral)
# c: 用数值表示颜色, cmap: 数值到颜色的映射关系


# %%
clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T, Y.T)
plot_decision_boundary(clf.predict, X, Y)
plt.title('Logistic Regression')
plt.show()


# %%
print('逻辑回归准确率', clf.score(X.T, Y.T))
# 对输入X正规化缩放
X = X / 5
# 数据集中Y.dtype是'uint8'类型, 使得-Y取值变成(0, 255), 必须要用astype()方法转化成'float64'
# 记住使用数据前一定先检查数据的类型!!!!!!!!!!!
Y = Y.astype(np.float64)
print(Y.shape, Y.dtype, '\n', Y)


# %%
def tanh(x):
    s = np.tanh(x)
    return s

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def ReLU(x):
    s = np.maximum(x, 0)
    return s



class Layer(object):


    def __init__(self, n_input, n_layer, activate, name=None):
        self.n = n_layer
        self.name = name
        self.activate = activate
        self.X = None
        self.W = np.random.randn(n_layer, n_input) * 1e-4
        self.b = np.zeros((n_layer, 1))
        self.Z = None
        self.A = None
        self.dA = None
        self.dZ = None
        self.dW = None
        self.db = None
        self.dX = None

    def get_W(self):
        return self.W

    def get_b(self):
        return self.b

    def get_A(self):
        return self.A

    def get_dX(self):
        return self.dX    

    def fwprop(self, X):
        self.X = X
        self.Z = np.dot(self.W, self.X) + self.b
        self.A = self.activate(self.Z)

    def bkprop(self, dA, alpha):
        self.dA = dA
        if self.activate == tanh:
            self.dZ = self.dA * (1 - self.A**2)
        elif self.activate == sigmoid:
            self.dZ = self.dA * self.A * (1 - self.A)
        elif self.activate == ReLU:
            self.dZ = self.dA * (self.A > 0)
        else:
            print('Wrong activate function!')
        self.dX = np.dot(self.W.T, self.dZ)
        self.dW = np.dot(self.dZ, self.X.T)
        self.db = np.mean(self.dZ, axis=1).reshape((self.n, 1))
        self.W -= alpha * self.dW
        self.b -= alpha * self.db



class Model(object):


    def __init__(self, X_train, Y_train, alpha=0.005, loops=2000):
        self.X = X_train
        self.Y = Y_train
        self.alpha = alpha
        self.loops = loops
        self.L = 0
        self.layers = []
        self.layers_reverse = []
        self.A = None
        self.J = None
        self.J_list = []

    def add_layer(self, n_input, n_layer, activate, name=None):
        self.L += 1
        new_layer = Layer(n_input, n_layer, activate, name=name)
        self.layers.append(new_layer)
        self.layers_reverse = self.layers.copy()
        self.layers_reverse.reverse()
        return new_layer

    def fwprop(self):
        input_ = self.X
        for layer in self.layers:
            layer.fwprop(input_)
            input_ = layer.get_A()

    def compute_J(self):
        self.A = self.layers[-1].get_A()
        # 防止出现log(0)等于nan的情况, 每次log()内加上极小正数
        self.J = (-self.Y * np.log(self.A + 1e-10) - (1 - self.Y) * np.log(1 - self.A + 1e-10)).mean()

    def bkprop(self):
        # 防止出现x / 0等于nan的情况, 每次 / 分母加上极小正数
        dA = -self.Y / (self.A +10e-10) + (1 - self.Y) / (1 - self.A +10e-10)
        for layer in self.layers_reverse:
            layer.bkprop(dA, alpha=self.alpha)
            dA = layer.get_dX()

    def train(self):
        for i in range(self.loops):
            self.fwprop()
            self.compute_J()
            self.bkprop()
            j = i + 1
            if j%50 == 0:
                self.J_list.append(self.J)
                print(f'第{j}次迭代的损失值: {self.J}')

    def predict(self, X_test):
        for layer in self.layers:
            layer.fwprop(X_test)
            X_test = layer.get_A()
        return X_test

    def score(self, X_test, Y_test):
        #仅实现二分类问题
        Y_predict = self.predict(X_test)
        result = (Y_predict > 0.5) == Y_test
        acc = result.mean()
        return acc

    def plot_loss(self):
        plt.plot(self.J_list)
        plt.xlabel('steps per 50')
        plt.ylabel('loss')
        plt.title('loss circle')
        plt.show()

    def plot_bound(self):
        plot_decision_boundary(lambda x: self.predict(x.T), self.X, self.Y)
        plt.title('Neual Network Decision Boundary')
        plt.show()


# %%
model = Model(X, Y, alpha=0.005, loops=4000)
layer1 = model.add_layer(2, 4, tanh, name='Hidden Layer')
layer2 = model.add_layer(4, 1, sigmoid, name='Output Layer')
model.train()


# %%
accuracy = model.score(X, Y)
print(accuracy)


# %%
model.plot_loss()


# %%
model.plot_bound()


# %%
print('layer1:', model.layers[0].name)
print('layer2:', model.layers[1].name)

