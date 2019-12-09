# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles



def generate_data(n_samples, set_type='Moon', p=0.8, seed=2019):
    assert(set_type in ['Moon', 'Circle', 'Planar']), 'You must choice set type one of "Moon" or "Circle" or "Planar" !'
    np.random.seed(seed)
    n_train = int(n_samples * p)
    n_test = n_samples - n_train
    if set_type == 'Moon':
        X_train, Y_train = make_moons(n_samples=n_train, noise=0.2, random_state=2019)
        X_test, Y_test = make_moons(n_samples=n_test, noise=0.3, random_state=2019)
    elif set_type == 'Circle':
        X_train, Y_train = make_circles(n_samples=n_train, noise=0.15, random_state=2019, factor=0.4)
        X_test, Y_test = make_circles(n_samples=n_test, noise=0.25, random_state=2019, factor=0.4)
    else:
        X_train, Y_train = load_planar_dataset(n_samples=n_train, noise=0.2)
        X_test, Y_test = load_planar_dataset(n_samples=n_test, noise=0.4)
    return (X_train.T, Y_train), (X_test.T, Y_test)

def plot(data, labels, title='Train data', s=35, axis=False, xlim=None, ylim=None):
    plt.scatter(data.T[labels==1][:, 0], data.T[labels==1][:, 1], color='b', edgecolor='k', label='label : 1', s=s)
    plt.scatter(data.T[labels==0][:, 0], data.T[labels==0][:, 1], color='r', edgecolor='k', label='label : 0', s=s)
    plt.grid(True)
    plt.title(title)
    plt.legend()
    if axis:
        plt.axvline(x=0, color='black', linewidth=1)
        plt.axhline(y=0, color='black', linewidth=1)
    if xlim:
        plt.xlim(*xlim)
    if ylim:
        plt.ylim(*ylim)
        
        
def decision_boundary(parameters, xlim, ylim, colormap):
    xmin, xmax = xlim
    ymin, ymax = ylim
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 30), np.linspace(ymin, ymax, 30))
    grids = np.c_[xx.ravel(), yy.ravel()]
    predict = _forward(grids.T, parameters)
    Z = predict.reshape(xx.shape)
    plt.contour(xx, yy, Z, levels=[0.5], colors='k')
    if colormap == True:
        plt.contourf(xx, yy, Z, cmap='RdBu', alpha=0.7)
        
        
def draw_boundary(parameters, data, labels, title='Train data', colormap=False, s=35, axis=False, xlim=None, ylim=None):
    # 먼저 데이터 플롯한다
    plot(data, np.squeeze(labels), title=title, s=s, axis=axis, xlim=xlim, ylim=ylim)
    axes = plt.gca() # 현재 플롯된 axes객체를 가져온다
    xlim = axes.get_xlim()
    ylim = axes.get_ylim()
    # 학습모델의 Decision boundary
    decision_boundary(parameters, xlim, ylim, colormap)

def sigmoid(z):
    '''
    Compute the sigmoid of z
    
    Arguments: A scalar or numpy array of any size
    '''
    return 1 / (1 + np.exp(-z))


def ReLU(z):
    return np.maximum(0, z)


def initialize_weights(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.

    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """
    ### manual seed
    np.random.seed(0)
    
    ### START CODE HERE ### (≈ 1 line of code)
    w = np.random.randn(dim, 1)
    b = np.random.randn(1).item()
    ### END CODE HERE ###

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    
    return w, b



def forward(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Return:
    yhat -- prediction of corresponding input
    
    Tips:
    - Write your code step by step for the propagation
    """
    
    m = X.shape[1]
    
    # FORWARD PROPAGATION
    ### START CODE HERE ### (≈ 2 lines of code)
    Yhat = sigmoid(np.dot(w.T, X) + b)
    cost = (- 1 / m) * np.sum(Y * np.log(Yhat) + (1 - Y) * (np.log(1 - Yhat)))
    ### END CODE HERE ###
    
    return Yhat, cost

# def _forward(w, b, X):

#     m = X.shape[1]
    
#     # FORWARD PROPAGATION
#     ### START CODE HERE ### (≈ 2 lines of code)
#     Yhat = sigmoid(np.dot(w.T, X) + b)
#     ### END CODE HERE ###
    
#     return Yhat

# GRADED FUNCTION: forward_propagation

def _forward(X, parameters):

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    Z1 = np.dot(W1, X) + b1
    A1 = ReLU(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    
    assert(A2.shape == (1, X.shape[1]))
    return A2

def backward(w, b, X, Y, Yhat):
    
    '''
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)
    Yhat -- predicted label which can be interpreted as probability(= confidence)

    Return:
    grads -- gradient of parameters
    '''
    
    m = X.shape[1]
    # print(w, b, X, Y, Yhat)
    
    # BACKWARD PROPAGATION
    ### START CODE HERE ### (≈ 2 lines of code)
    # print(Yhat-Y)
    dw = (1 / m) * np.dot(X, (Yhat - Y).T)
    db = (1 / m) * np.sum(Yhat - Y)
    # print(dw, db)
    ### END CODE HERE ###

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    
    grads = {"dw": dw,
             "db": db}
    
    return grads

def fit(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    """
    This function optimizes w and b by running a gradient descent algorithm
    
    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    
    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use forward(), backward().
        2) Update the parameters using gradient descent rule for w and b.
    """
    
    costs = []
    
    for i in range(num_iterations):
        
        # Cost and gradient calculation (≈ 1-4 lines of code)
        ### START CODE HERE ### 
        Yhat, cost = forward(w, b, X, Y)
        grads = backward(w, b, X, Y, Yhat)
        ### END CODE HERE ###
        
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        # print(dw, db)
        
        # update rule (≈ 2 lines of code)
        ### START CODE HERE ###
        w = w - learning_rate * dw  # need to broadcast
        b = b - learning_rate * db
        ### END CODE HERE ###
        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        
        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))
            # print(dw, db)
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs

# GRADED FUNCTION: predict

def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    
    m = X.shape[1]
    assert(w.shape[0] == X.shape[0])
    
    ### START CODE HERE ### (≈ 4 line of code)
    Yhat = sigmoid(np.dot(w.T, X) + b)
    Yhat[Yhat > 0.5] = 1
    Yhat[Yhat <= 0.5] = 0
    Y_prediction = Yhat
    ### END CODE HERE ###
    
    assert(Y_prediction.shape == (1, m))
    
    return Y_prediction

# GRADED FUNCTION: model

def Logistic(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """
    Builds the logistic regression model by calling the function you've implemented previously
    
    Returns:
    d -- dictionary containing information about the model.
    """
    
    ### START CODE HERE ###
    # initialize parameters with zeros (≈ 1 line of code)
    w, b = initialize_weights(X_train.shape[0])

    # Gradient descent (≈ 1 line of code)
    parameters, grads, costs = fit(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    
    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]
    
    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    ### END CODE HERE ###

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d


def layer_sizes_test_case():
    np.random.seed(1)
    X_assess = np.random.randn(5, 3)
    Y_assess = np.random.randn(2, 3)
    return X_assess, Y_assess

def initialize_parameters_test_case():
    n_x, n_h, n_y = 2, 4, 1
    return n_x, n_h, n_y

def forward_propagation_test_case():
    np.random.seed(1)
    X_assess = np.random.randn(2, 3)

    parameters = {'W1': np.array([[-0.00416758, -0.00056267],
        [-0.02136196,  0.01640271],
        [-0.01793436, -0.00841747],
        [ 0.00502881, -0.01245288]]),
     'W2': np.array([[-0.01057952, -0.00909008,  0.00551454,  0.02292208]]),
     'b1': np.array([[ 0.],
        [ 0.],
        [ 0.],
        [ 0.]]),
     'b2': np.array([[ 0.]])}

    return X_assess, parameters

def compute_cost_test_case():
    np.random.seed(1)
    Y_assess = np.random.randn(1, 3)
    parameters = {'W1': np.array([[-0.00416758, -0.00056267],
        [-0.02136196,  0.01640271],
        [-0.01793436, -0.00841747],
        [ 0.00502881, -0.01245288]]),
     'W2': np.array([[-0.01057952, -0.00909008,  0.00551454,  0.02292208]]),
     'b1': np.array([[ 0.],
        [ 0.],
        [ 0.],
        [ 0.]]),
     'b2': np.array([[ 0.]])}

    a2 = (np.array([[ 0.5002307 ,  0.49985831,  0.50023963]]))
    
    return a2, Y_assess, parameters

def backward_propagation_test_case():
    np.random.seed(1)
    X_assess = np.random.randn(2, 3)
    Y_assess = np.random.randn(1, 3)
    parameters = {'W1': np.array([[-0.00416758, -0.00056267],
        [-0.02136196,  0.01640271],
        [-0.01793436, -0.00841747],
        [ 0.00502881, -0.01245288]]),
     'W2': np.array([[-0.01057952, -0.00909008,  0.00551454,  0.02292208]]),
     'b1': np.array([[ 0.],
        [ 0.],
        [ 0.],
        [ 0.]]),
     'b2': np.array([[ 0.]])}

    cache = {'A1': np.array([[-0.00616578,  0.0020626 ,  0.00349619],
         [-0.05225116,  0.02725659, -0.02646251],
         [-0.02009721,  0.0036869 ,  0.02883756],
         [ 0.02152675, -0.01385234,  0.02599885]]),
  'A2': np.array([[ 0.5002307 ,  0.49985831,  0.50023963]]),
  'Z1': np.array([[-0.00616586,  0.0020626 ,  0.0034962 ],
         [-0.05229879,  0.02726335, -0.02646869],
         [-0.02009991,  0.00368692,  0.02884556],
         [ 0.02153007, -0.01385322,  0.02600471]]),
  'Z2': np.array([[ 0.00092281, -0.00056678,  0.00095853]])}
    return parameters, cache, X_assess, Y_assess

def update_parameters_test_case():
    parameters = {'W1': np.array([[-0.00615039,  0.0169021 ],
        [-0.02311792,  0.03137121],
        [-0.0169217 , -0.01752545],
        [ 0.00935436, -0.05018221]]),
 'W2': np.array([[-0.0104319 , -0.04019007,  0.01607211,  0.04440255]]),
 'b1': np.array([[ -8.97523455e-07],
        [  8.15562092e-06],
        [  6.04810633e-07],
        [ -2.54560700e-06]]),
 'b2': np.array([[  9.14954378e-05]])}

    grads = {'dW1': np.array([[ 0.00023322, -0.00205423],
        [ 0.00082222, -0.00700776],
        [-0.00031831,  0.0028636 ],
        [-0.00092857,  0.00809933]]),
 'dW2': np.array([[ -1.75740039e-05,   3.70231337e-03,  -1.25683095e-03,
          -2.55715317e-03]]),
 'db1': np.array([[  1.05570087e-07],
        [ -3.81814487e-06],
        [ -1.90155145e-07],
        [  5.46467802e-07]]),
 'db2': np.array([[ -1.08923140e-05]])}
    return parameters, grads

def nn_model_test_case():
    np.random.seed(1)
    X_assess = np.random.randn(2, 3)
    Y_assess = np.random.randn(1, 3)
    return X_assess, Y_assess

def predict_test_case():
    np.random.seed(1)
    X_assess = np.random.randn(2, 3)
    parameters = {'W1': np.array([[-0.00615039,  0.0169021 ],
        [-0.02311792,  0.03137121],
        [-0.0169217 , -0.01752545],
        [ 0.00935436, -0.05018221]]),
     'W2': np.array([[-0.0104319 , -0.04019007,  0.01607211,  0.04440255]]),
     'b1': np.array([[ -8.97523455e-07],
        [  8.15562092e-06],
        [  6.04810633e-07],
        [ -2.54560700e-06]]),
     'b2': np.array([[  9.14954378e-05]])}
    return parameters, X_assess