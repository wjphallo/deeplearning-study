# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import os
print(os.getcwd())


# %%
os.chdir('./2-1/')


# %%
import gc_utils
import init_utils
import reg_utils
plt.rcParams['figure.figsize'] = (7.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
# # plt.rcParams['image.cmap'] = 'gray'

# %% [markdown]
# # initialize weights

# %%
X_train, Y_train, X_test, Y_test = init_utils.load_dataset(is_plot=True)


# %%
def init_zeros(layers_dims):
    params = {}
    L = len(layers_dims)
    for l in range(1, L):
        params[f'W{l}'] = np.zeros((layers_dims[l], layers_dims[l-1]))
        params[f'b{l}'] = np.zeros((layers_dims[l], 1))
    return params

def init_random(layers_dims):
    params = {}
    L = len(layers_dims)
    for l in range(1, L):
        params[f'W{l}'] = np.random.randn(layers_dims[l], layers_dims[l-1]) * 10
        params[f'b{l}'] = np.zeros((layers_dims[l], 1))
    return params

def init_he(layers_dims):
    params = {}
    L = len(layers_dims)
    for l in range(1, L):
        params[f'W{l}'] = np.random.randn(layers_dims[l], layers_dims[l-1]) * np.sqrt(2/layers_dims[l-1])
        params[f'b{l}'] = np.zeros((layers_dims[l], 1))
    return params


# %%
def model(X, Y, alpha=0.005, loops=5000, init_method='he'):
    grads = {}
    costs = []
    m = X.shape[1]
    layers_dims = [X.shape[0], 10, 5, 1]
    # inintial params
    if init_method == 'zeros':
        params = init_zeros(layers_dims)
    elif init_method == 'random':
        params = init_random(layers_dims)
    elif init_method == 'he':
        params = init_he(layers_dims)
    else:
        print('Error: unexcepted init_method!')
    # start train
    for i in range(loops):
        a3, cache = init_utils.forward_propagation(X, params)
        cost = init_utils.compute_loss(a3, Y)
        costs.append(cost)
        grads = init_utils.backward_propagation(X, Y, cache)
        params = init_utils.update_parameters(params, grads, alpha)
        if (i+1) % 100 == 0:
            print(f'No.{i+1} iteration\'s loss: {cost}')
    plt.plot(costs)
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.title('loss circle')
    plt.show()
    return params


# %%
# use 'zero' to initial parameters
params = model(X_train, Y_train, init_method='zeros')
print('train set')
prediction = init_utils.predict(X_train, Y_train, params)
print('test set')
prediction = init_utils.predict(X_test, Y_test, params)


# %%
# use 'random' to initial parameters
np.random.seed(1)
params = model(X_train, Y_train, alpha=0.001, loops=15000, init_method='random')
print('train set')
prediction = init_utils.predict(X_train, Y_train, params)
print('test set')
prediction = init_utils.predict(X_test, Y_test, params)


# %%
# use 'he' to initial parameters
np.random.seed(1)
params = model(X_train, Y_train, alpha=0.06, loops=15000, init_method='he')
print('train set')
prediction = init_utils.predict(X_train, Y_train, params)
print('test set')
prediction = init_utils.predict(X_test, Y_test, params)


# %%
axes = plt.gca()
axes.set_xlim([-1.5, 1.5])
axes.set_ylim([-1.5, 1.5])
init_utils.plot_decision_boundary(lambda X: init_utils.predict_dec(params, X.T), X_train, np.squeeze(Y_train))

# %% [markdown]
# # no-regularization, L2-regularization, dropout

# %%
X_re_train, Y_re_train, X_re_test, Y_re_test = reg_utils.load_2D_dataset(is_plot=True)


# %%
def forward_propagate_with_reg(X, params, keep_prob=1):
    # retrieve parameters
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]
    W3 = params["W3"]
    b3 = params["b3"]
    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    z1 = np.dot(W1, X) + b1
    D1 = np.random.rand(z1.shape[0], z1.shape[1]) < keep_prob
    # 每个样本dropout都不同, 所以不要用boardcast
    a1 = reg_utils.relu(z1) * D1 / keep_prob
    z2 = np.dot(W2, a1) + b2
    D2 = np.random.rand(z2.shape[0], z2.shape[1]) < keep_prob
    a2 = reg_utils.relu(z2) * D2 / keep_prob
    z3 = np.dot(W3, a2) + b3
    a3 = reg_utils.sigmoid(z3)
    cache = (D1, z1, a1, W1, b1, D2, z2, a2, W2, b2, z3, a3, W3, b3)
    return a3, cache

def compute_loss_with_reg(A3, Y, params, lambd=0):
    m = Y.shape[1]
    W1 = params['W1']
    W2 = params['W2']
    W3 = params['W3']
    cross_entropy_cost = reg_utils.compute_cost(A3, Y)
    L2_regularization_cost = lambd * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2)) / (2*m)
    loss = cross_entropy_cost + L2_regularization_cost
    return loss

def backward_propagate_with_reg(X, Y, cache, lambd=0, keep_prob=1):
    m = X.shape[1]
    (D1, z1, a1, W1, b1, D2, z2, a2, W2, b2, z3, a3, W3, b3) = cache
    dz3 = 1./m * (a3 - Y)
    dW3 = np.dot(dz3, a2.T) + W3 * (lambd/m)
    db3 = np.sum(dz3, axis=1, keepdims = True)
    da2 = np.dot(W3.T, dz3) * D2 / keep_prob
    dz2 = np.multiply(da2, np.int64(a2 > 0))
    dW2 = np.dot(dz2, a1.T) + W2 * (lambd/m)
    db2 = np.sum(dz2, axis=1, keepdims = True)
    da1 = np.dot(W2.T, dz2) * D1 / keep_prob
    dz1 = np.multiply(da1, np.int64(a1 > 0))
    dW1 = np.dot(dz1, X.T) + W1 * (lambd/m)
    db1 = np.sum(dz1, axis=1, keepdims = True)
    gradients = {"dz3": dz3, "dW3": dW3, "db3": db3,
                 "da2": da2, "dz2": dz2, "dW2": dW2, "db2": db2,
                 "da1": da1, "dz1": dz1, "dW1": dW1, "db1": db1}
    return gradients


# %%
def model_reg(X, Y, alpha=0.05, loops=10000, lambd=0, keep_prob=1, init_method='he'):
    grads = {}
    costs = []
    m = X.shape[1]
    layers_dims = [X.shape[0], 20, 3, 1]
    # inintial params
    if init_method == 'zeros':
        params = init_zeros(layers_dims)
    elif init_method == 'random':
        params = init_random(layers_dims)
    elif init_method == 'he':
        params = init_he(layers_dims)
    else:
        print('Error: unexcepted init_method!')
    # start train
    for i in range(loops):
        a3, cache = forward_propagate_with_reg(X, params, keep_prob=keep_prob)
        cost = compute_loss_with_reg(a3, Y, params, lambd=lambd)
        costs.append(cost)
        grads = backward_propagate_with_reg(X, Y, cache, lambd=lambd, keep_prob=keep_prob)
        params = reg_utils.update_parameters(params, grads, alpha)
        if (i+1) % 100 == 0:
            print(f'No.{i+1} iteration\'s loss: {cost}')
    plt.plot(costs)
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.title('loss circle')
    plt.show()
    return params


# %%
# no regularization
np.random.seed(1)
params = model_reg(X_re_train, Y_re_train, alpha=0.08, loops=10000, lambd=0, keep_prob=1, init_method='he')
print('train set')
prediction = reg_utils.predict(X_re_train, Y_re_train, params)
print('test set')
prediction = reg_utils.predict(X_re_test, Y_re_test, params)


# %%
plt.title('No regularization')
axes = plt.gca()
axes.set_xlim([-0.75, 0.40])
axes.set_ylim([-0.75, 0.65])
reg_utils.plot_decision_boundary(lambda X: reg_utils.predict_dec(params, X.T), X_re_train, np.squeeze(Y_re_train))


# %%
# L2 regularization
np.random.seed(1)
params = model_reg(X_re_train, Y_re_train, alpha=0.08, loops=10000, lambd=0.7, keep_prob=1, init_method='he')
print('train set')
prediction = reg_utils.predict(X_re_train, Y_re_train, params)
print('test set')
prediction = reg_utils.predict(X_re_test, Y_re_test, params)


# %%
plt.title('L2 regularization')
# 抑制过拟合 , 分界面更平滑
axes = plt.gca()
axes.set_xlim([-0.75, 0.40])
axes.set_ylim([-0.75, 0.65])
reg_utils.plot_decision_boundary(lambda X: reg_utils.predict_dec(params, X.T), X_re_train, np.squeeze(Y_re_train))


# %%
# dropout
np.random.seed(1)
params = model_reg(X_re_train, Y_re_train, alpha=0.08, loops=10000, lambd=0, keep_prob=0.9, init_method='he')
print('train set')
prediction = reg_utils.predict(X_re_train, Y_re_train, params)
print('test set')
prediction = reg_utils.predict(X_re_test, Y_re_test, params)


# %%
plt.title('dropout')
# 效果似乎不如L2正则, 平缓阶段会有震荡且很难通过alpha调整, 不过结果同样具备更好的泛化能力
# 如果每隔500或者1000取点作图, 你可能不会看到这么明显的震荡
axes = plt.gca()
axes.set_xlim([-0.75, 0.40])
axes.set_ylim([-0.75, 0.65])
reg_utils.plot_decision_boundary(lambda X: reg_utils.predict_dec(params, X.T), X_re_train, np.squeeze(Y_re_train))

# %% [markdown]
# # gradients check

# %%
def gradients_check(X, Y, lambd=0, keep_prob=1, init_method='he'):
    layers_dims = [X.shape[0], 5, 3, 1]
    # inintial params
    if init_method == 'zeros':
        params = init_zeros(layers_dims)
    elif init_method == 'random':
        params = init_random(layers_dims)
    elif init_method == 'he':
        params = init_he(layers_dims)
    else:
        print('Error: unexcepted init_method!')
    
    # compute grads
    a3, cache = forward_propagate_with_reg(X, params, keep_prob=keep_prob)
    grads = backward_propagate_with_reg(X, Y, cache, lambd=lambd, keep_prob=keep_prob)
    grads_vector = gc_utils.gradients_to_vector(grads)

    theta, keys = gc_utils.dictionary_to_vector(params)#转化成向量方便索引(n, 1)
    n = theta.shape[0]#参数个数
    grads_approx_vector = np.zeros((n, 1))
    
    # compute grads_approx
    for i in range(n):
        theta_p = np.copy(theta)
        theta_p[i, 0] += 1e-7
        params_p = gc_utils.vector_to_dictionary(theta_p)
        theta_m = np.copy(theta)
        theta_m[i, 0] -= 1e-7
        params_m = gc_utils.vector_to_dictionary(theta_m)
        a3_, cache_ = forward_propagate_with_reg(X, params_p, keep_prob=keep_prob)
        J_p = compute_loss_with_reg(a3_, Y, params_p, lambd=lambd)
        a3_, cache_ = forward_propagate_with_reg(X, params_m, keep_prob=keep_prob)
        J_m = compute_loss_with_reg(a3_, Y, params_m, lambd=lambd)
        d_approx = (J_p - J_m) / (2 * 1e-7)
        grads_approx_vector[i, 0] = d_approx

    # compute difference
    numerator = np.linalg.norm(grads_vector - grads_approx_vector)
    denominator = np.linalg.norm(grads_vector) + np.linalg.norm(grads_approx_vector)
    diff = numerator / denominator
    return diff


# %%
X_case = np.array([
    [1,3,5,7,9],
    [2,4,6,8,10],
    [3,6,9,12,15],
    [4,8,12,16,20]
])
Y_case = np.array([[1,1,0,0,1]])
diff = gradients_check(X_case, Y_case, lambd=0, keep_prob=1, init_method='he')
print(f'diff = {diff}', '梯度是否在阈值内(正常?):', diff<1e-7)


# %%


