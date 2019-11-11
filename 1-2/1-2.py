# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
os.chdir('./1-2/')
import numpy as np
import matplotlib.pyplot as plt
import h5py
from lr_utils import load_dataset


# %%
X_train, Y_train, X_test, Y_test, classes = load_dataset()


# %%
print(f'X_train.shape = {X_train.shape}\nY_train.shape = {Y_train.shape}\nX_test.shape = {X_test.shape}\nY_test.shape = {Y_test.shape}')


# %%
X_train = X_train.reshape((X_train.shape[0], -1)).T / 255
X_test = X_test.reshape((X_test.shape[0], -1)).T / 255
print(X_train.shape, X_test.shape)


# %%
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

sigmoid(0)


# %%
class Model(object):

    def __init__(self, alpha=0.004, loops=2000, print_cost=True):
        self.alpha = alpha
        self.loops = loops
        self.print_cost = print_cost
        self.W = None
        self.b = None
        self.cost_per50 = []

    def get_W(self):
        return self.W

    def get_b(self):
        return self.b

    def get_cost_per50(self):
        return self.cost_per50

    def init_para(self, dim):
        self.W = np.random.randn(1, dim) * 10e-4
        self.b = 0
        return self.W, self.b

    def prop(self, X, Y, W, b):
        m = Y.shape[1]
        A = sigmoid(np.dot(W, X) + b)
        dW = np.dot(A-Y, X.T) / m
        db = np.sum(A-Y) / m
        self.W -= self.alpha * dW
        self.b -= self.alpha *db
        L = -Y * np.log(A) - (1 - Y) * np.log(1 - A)
        J = np.sum(L) / m
        return self.W, self.b, J

    def fit(self, X_train, Y_train):
        self.W, self.b = self.init_para(X_train.shape[0])
        # print('W[0, :5]:', self.W[0, :5])
        for i in range(self.loops):
            self.W, self.b, J = self.prop(X_train, Y_train, self.W, self.b)
            j = i + 1
            if j%50 == 0: #or j == 1:
                # print('W[0, :5]:', self.W[0, :5])
                self.cost_per50.append(J)
                if self.print_cost:
                    print(f'第{j}次迭代损失值: {J}')
        return self

    def predict(self, X):
        Y_predict = sigmoid(np.dot(self.W, X) + self.b)
        return Y_predict

    def get_acc(self, X, Y):
        Y_predict = self.predict(X) > 0.5
        acc = np.sum(Y_predict == Y) / Y.shape[1]
        return acc


# %%
model_list = []
for alpha in [0.005, 0.001, 0.0005]:
    print(f'\n------Start with alpha = {alpha}------\n')
    model = Model(alpha=alpha)
    model.fit(X_train, Y_train)
    model_list.append(model)
    print(f'alpha={alpha}, 训练集准确度：', model.get_acc(X_train, Y_train))
    print(f'alpha={alpha}, 测试集准确度：', model.get_acc(X_test, Y_test))


# %%
for model in model_list:
    cost = model.get_cost_per50()
    plt.plot(cost, label=f'alpha={model.alpha}')
plt.ylabel('cost')
plt.xlabel('steps')
plt.legend()
plt.show()

