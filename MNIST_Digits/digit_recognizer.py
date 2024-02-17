import numpy as np
import pandas as pd

data = np.array(pd.read_csv('./digits.csv')) # csv -> pandas dataframe -> numpy array
m, n = data.shape # m = rows, n = columns
np.random.shuffle(data) # shuffle before splitting into testing and training sets

data_test = data[0:1000].T
Y_dev = data_test[0]
X_test = data_test[1:n]
X_test = X_test / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape

def init_params():
    w1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    w2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return w1, b1, w2, b2

def ReLU(z):
    return np.maximum(z, 0)

def softmax(z):
    a = np.exp(z) / sum(np.exp(z))
    return a
    
def forward_prop(w1, b1, w2, b2, X):
    z1 = w1.dot(X) + b1
    a1 = ReLU(z1)
    z2 = w2.dot(a1) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2

def ReLU_deriv(z):
    return z > 0

def one_hot(y):
    one_hot_Y = np.zeros((y.size, y.max() + 1))
    one_hot_Y[np.arange(y.size), y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(z1, a1, z2, a2, w1, w2, X, y):
    one_hot_Y = one_hot(y)
    dz2 = a2 - one_hot_Y
    dw2 = 1 / m * dz2.dot(a1.T)
    db2 = 1 / m * np.sum(dz2)
    dz1 = w2.T.dot(dz2) * ReLU_deriv(z1)
    dw1 = 1 / m * dz1.dot(X.T)
    db1 = 1 / m * np.sum(dz1)
    return dw1, db1, dw2, db2

def update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1    
    w2 = w2 - alpha * dw2  
    b2 = b2 - alpha * db2    
    return w1, b1, w2, b2

def get_predictions(a2):
    return np.argmax(a2, 0)

def get_accuracy(predictions, y):
    #print(predictions, y)
    return np.sum(predictions == y) / y.size

def gradient_descent(X, y, alpha, iterations):
    w1, b1, w2, b2 = init_params()
    for i in range(iterations):
        z1, a1, z2, a2 = forward_prop(w1, b1, w2, b2, X)
        dw1, db1, dw2, db2 = backward_prop(z1, a1, z2, a2, w1, w2, X, y)
        w1, b1, w2, b2 = update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)
        if i % 50 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(a2)
            print(get_accuracy(predictions, y))
    return w1, b1, w2, b2

w1, b1, w2, b2 = gradient_descent(X_train, Y_train, 0.10, 1000)

def make_predictions(X, w1, b1, w2, b2):
    _, _, _, a2 = forward_prop(w1, b1, w2, b2, X)
    predictions = get_predictions(a2)
    return predictions

test_predictions = make_predictions(X_test, w1, b1, w2, b2)
print(f"\nAccuracy on the test set is : {get_accuracy(test_predictions, Y_dev)}")