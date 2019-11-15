# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import sklearn
import sklearn.datasets
import os
os.chdir('./2-2/')
import opt_utils

# %% [markdown]
# # Gradient Descent

# %%
def update_params_with_gd(params, grads, alpha):
    L = len(params) // 2
    for l in range(L):
        params[f'W{l+1}'] -= alpha * grads[f'dW{l+1}']
        params[f'b{l+1}'] -= alpha * grads[f'db{l+1}']
    return params

# %% [markdown]
# # Mini-batch Gradient Descent
# %% [markdown]
# ## shuffle and get mini-batches

# %%
def random_mini_batches(X, Y, mini_batch_size=64, seed=1):
    
    np.random.seed(seed)
    m = X.shape[1]
    mini_batches = []
    
    #shuffle
    permutattion = list(np.random.permutation(m))
    shuffled_X = X[:, permutattion]
    shuffled_Y = Y[:, permutattion]
    #segmentation
    n_complete_batches = m // mini_batch_size
    for n in range(n_complete_batches):
        mini_batch_X = shuffled_X[:, n*mini_batch_size : (n+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[:, n*mini_batch_size : (n+1)*mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, n_complete_batches*mini_batch_size:]
        mini_batch_Y = shuffled_Y[:, n_complete_batches*mini_batch_size:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
        
    return mini_batches

# %% [markdown]
# # Momentum
# %% [markdown]
# ## initial V

# %%
def init_momentum(params:dict):

    L = len(params) // 2
    V = {}

    for l in range(L):
        V[f'dW{l+1}'] = np.zeros_like(params[f'W{l+1}'])
        V[f'db{l+1}'] = np.zeros_like(params[f'b{l+1}'])

    return V

# %% [markdown]
# ## update parameters with momentum

# %%
def update_params_with_momentum(params, grads, V, beta, alpha):

    L = len(params) // 2
    for l in range(L):
        V[f'dW{l+1}'] = beta * V[f'dW{l+1}'] + (1-beta) * grads[f'dW{l+1}']
        V[f'db{l+1}'] = beta * V[f'db{l+1}'] + (1-beta) * grads[f'db{l+1}']
        params[f'W{l+1}'] -= alpha * V[f'dW{l+1}']
        params[f'b{l+1}'] -= alpha * V[f'db{l+1}']

    return params, V

# %% [markdown]
# # Adam
# %% [markdown]
# ## initial V, S

# %%
def init_adam(params:dict):

    L = len(params) // 2
    V = {}
    S = {}

    for l in range(L):
        V[f'dW{l+1}'] = np.zeros_like(params[f'W{l+1}'])
        V[f'db{l+1}'] = np.zeros_like(params[f'b{l+1}'])
        S[f'dW{l+1}'] = np.zeros_like(params[f'W{l+1}'])
        S[f'db{l+1}'] = np.zeros_like(params[f'b{l+1}'])

    return V, S

# %% [markdown]
# ## update parameters with adam

# %%
def update_params_with_adam(params, grads, V, S, t, alpha, beta1=0.9, beta2=0.999):

    e = 1e-8
    L = len(params) // 2
    
    for l in range(L):

        V[f'dW{l+1}'] = beta1 * V[f'dW{l+1}'] + (1-beta1) * grads[f'dW{l+1}']
        V[f'db{l+1}'] = beta1 * V[f'db{l+1}'] + (1-beta1) * grads[f'db{l+1}']

        Vw_corrected = V[f'dW{l+1}'] / (1 - beta1**t)
        Vb_corrected = V[f'db{l+1}'] / (1 - beta1**t)

        S[f'dW{l+1}'] = beta2 * S[f'dW{l+1}'] + (1-beta2) * np.square(grads[f'dW{l+1}'])
        S[f'db{l+1}'] = beta2 * S[f'db{l+1}'] + (1-beta2) * np.square(grads[f'db{l+1}'])

        Sw_corrected = S[f'dW{l+1}'] / (1 - beta2**t)
        Sb_corrected = S[f'db{l+1}'] / (1 - beta2**t)

        params[f'W{l+1}'] -= alpha * (Vw_corrected / np.sqrt(Sw_corrected + e))
        params[f'b{l+1}'] -= alpha * (Vb_corrected / np.sqrt(Sb_corrected + e))

    return params, V, S

# %% [markdown]
# # Test Model

# %%
def model_opt(X, Y, layer_dims, optimizer:str, alpha=0.08, mini_batch_size=64, beta1=0.9, beta2=0.999, epochs=10000):

    L = len(layer_dims)
    costs = []
    t = 0
    seed = 1

    # initial weights
    params = opt_utils.initialize_parameters(layer_dims)

    # initial optimizer
    if optimizer == 'gd':
        pass
    elif optimizer == 'momentum':
        V = init_momentum(params)
    elif optimizer == 'adam':
        V, S = init_adam(params)
    else:
        print('Unexcepted optimizer!')
    
    # train
    for i in range(epochs):

        ## shuffle and get new mini_batches
        seed += 1
        mini_batches = random_mini_batches(X, Y,mini_batch_size, seed)
        
        for batch in mini_batches:
            ### get X,Y in batch
            mini_X, mini_Y = batch
            ### forward propagate
            A3, cache = opt_utils.forward_propagation(mini_X, params)
            ### compute loss
            cost = opt_utils.compute_cost(A3, mini_Y)
            ### backward propagate
            grads = opt_utils.backward_propagation(mini_X, mini_Y, cache)
            ### update params
            if optimizer == 'gd':
                params = update_params_with_gd(params, grads, alpha)
            elif optimizer == 'momentum':
                params, V = update_params_with_momentum(params, grads, V, beta1, alpha)
            elif optimizer == 'adam':
                t += 1
                params, V, S = update_params_with_adam(params, grads, V, S, t, alpha, beta1, beta2)
            else:
                print('Unexcepted optimizer!')
        
        if (i+1) % 100 == 0:
            costs.append(cost)
            print(f'No.{i+1} iteration\'s loss: {cost}')

    plt.plot(costs)
    plt.xlabel('# iterations per 100')
    plt.ylabel('loss')
    plt.title(f'{optimizer} loss circle')
    plt.show()

    return params

# %% [markdown]
# # load dataset and test

# %%
train_X, train_Y = opt_utils.load_dataset(is_plot=True)


# %%
layer_dims = [train_X.shape[0], 5, 2, 1]
np.random.seed(3)
params = model_opt(train_X, train_Y, layer_dims, optimizer='gd', alpha=0.0008, mini_batch_size=64, beta1=0.9, beta2=0.999, epochs=10000)


# %%
prediction = opt_utils.predict(train_X, train_Y, params)

plt.title('Gradient Descent')
axes = plt.gca()
axes.set_xlim([-1.5, 2.5])
axes.set_ylim([-1, 1.5])
opt_utils.plot_decision_boundary(lambda x: opt_utils.predict_dec(params, x.T), train_X, train_Y)


# %%
layer_dims = [train_X.shape[0], 5, 2, 1]
np.random.seed(3)
params = model_opt(train_X, train_Y, layer_dims, optimizer='momentum', alpha=0.0008, mini_batch_size=64, beta1=0.9, beta2=0.999, epochs=10000)


# %%
prediction = opt_utils.predict(train_X, train_Y, params)

plt.title('Momentum')
axes = plt.gca()
axes.set_xlim([-1.5, 2.5])
axes.set_ylim([-1, 1.5])
opt_utils.plot_decision_boundary(lambda x: opt_utils.predict_dec(params, x.T), train_X, train_Y)


# %%
layer_dims = [train_X.shape[0], 5, 2, 1]
np.random.seed(3)
params = model_opt(train_X, train_Y, layer_dims, optimizer='adam', alpha=0.0008, mini_batch_size=64, beta1=0.9, beta2=0.999, epochs=10000)


# %%
prediction = opt_utils.predict(train_X, train_Y, params)

plt.title('Adam')
axes = plt.gca()
axes.set_xlim([-1.5, 2.5])
axes.set_ylim([-1, 1.5])
opt_utils.plot_decision_boundary(lambda x: opt_utils.predict_dec(params, x.T), train_X, train_Y)


# %%


