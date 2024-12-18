from tkinter import N
import cupy as cp
import scipy 
from pyparsing import DebugExceptionAction
import scipy.special

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.W1 = cp.random.randn(input_size, hidden_size) * cp.sqrt(2. / input_size)
        self.b1 = cp.zeros(hidden_size)
        self.W2 = cp.random.randn(hidden_size, hidden_size) * cp.sqrt(2. / hidden_size)
        self.b2 = cp.zeros(hidden_size)
        self.W3 = cp.random.randn(hidden_size, output_size) * cp.sqrt(2. / hidden_size)
        self.b3 = cp.zeros(output_size)

    def relu(self, x):
        return cp.maximum(0, x)

    def relu_backward(self, dout, cache):
        dx = dout.copy()
        dx[cache <= 0] = 0
        return dx 
    
    def softmax(self,x):
        e = cp.exp(x-cp.max(x, axis=1, keepdims=True))
        return e/cp.sum(e, axis=1, keepdims=True)

    def cross_entropy_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        L = -cp.sum(y_true*cp.log(y_pred+1e-15), axis=1)
        loss = cp.mean(L)
        return loss 
    
    def forward(self, X):
        # Forward pass
        self.z1 = X.dot(self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = self.a1.dot(self.W2) + self.b2
        self.a2 = self.relu(self.z2)
        self.z3 = self.a2.dot(self.W3) + self.b3
        self.a3 = self.softmax(self.z3)  # Assuming no activation on the output layer (e.g., for regression)

        return self.a3

    def backward(self, X, y, learning_rate=1e-3):
        # Backward pass
        m = y.shape[0]

        # Assuming mean squared error loss
        dz3 = (self.a3-y)/m 

        dW3 = self.a2.T.dot(dz3)
        db3 = cp.sum(dz3, axis=0)

        da2 = dz3.dot(self.W3.T)
        dz2 = self.relu_backward(da2, self.z2)
        dW2 = self.a1.T.dot(dz2)
        db2 = cp.sum(dz2, axis=0)

        da1 = dz2.dot(self.W2.T)
        dz1 = self.relu_backward(da1, self.z1)
        dW1 = X.T.dot(dz1)
        db1 = cp.sum(dz1, axis=0)

        # Update weights and biases
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3

    def train(self, X, y, epochs=10, learning_rate=1e-3):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, learning_rate)
            loss = self.cross_entropy_loss(output, y)
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss}')

# Example usage:
input_size = 784  # Example input size (e.g., 28x28 images)
hidden_size = 4096
output_size = 10  # Example output size (e.g., 10 classes for classification)

mlp = MLP(input_size, hidden_size, output_size)

# Example data
X_train = cp.random.randn(100, input_size)  # 100 samples
y_train = cp.random.randint(0, output_size, 100)
y_train_one_hot = cp.eye(output_size)[y_train]
mlp.train(X_train, y_train_one_hot, epochs=100, learning_rate=0.01)