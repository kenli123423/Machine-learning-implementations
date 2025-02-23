import cupy as cp 

def relu(x):
    return cp.maximum(x, 0)

def relu_derivative(x):
    return cp.where(x>0, 1, 0)

def leaky_relu(alpha, x):
    return cp.where(x>0, x, alpha * x)

def leaky_relu(alpha, x):
    return cp.where(x>0, 1, alpha)

def sigmoid(x):
    return cp.reciprocal(1+cp.exp(-x))

def sigmoid_derivative(x):
    return cp.dot(sigmoid(x), 1-sigmoid(x))

def swish(x):
    return x*sigmoid(x)

def swish_derivative(x):
    return x*sigmoid_derivative(x) + sigmoid(x)

def softplus(x):
    return cp.log(1+cp.exp(x))

def softplus_derivative(x):
    return sigmoid(x)

def ELU(alpha, x):
    return cp.where(x>0, x, alpha * (cp.exp(x)-1))

def ELU_derivative(alpha, x):
    return cp.where(x>0, 1, alpha * cp.exp(x))
