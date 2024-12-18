import cupy as cp
from matplotlib.font_manager import X11FontDirectories
import numpy as np
import gc
 
class AdamOptimizer:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.params = params 
        self.lr = lr 
        self.beta1 = beta1 
        self.beta2 = beta2 
        self.eps = epsilon 
        self.m = [cp.zeros_like(param) for param in params]
        self.v = [cp.zeros_like(param) for param in params]
        #Dictionary comprehension 
        self.t = 0 
    
    def update(self, grads):
        self.t+=1
        lr_t = self.lr*((cp.sqrt(1-self.beta2**self.t))/(1-self.beta**self.t))
        for param, grad in zip(self.params, grads):
            self.m[param]=self.beta1*self.m[param]+(1-self.beta1)*(grad**2)
            self.v[param]=self.beta2*self.m[param]+(1-self.beta2)*(grad)
            m_hat = self.m[param]/(1-self.beta1**self.t)
            v_hat = self.v[param]/(1-self.beta2**self.t)
            param -= lr_t*m_hat/(cp.sqrt(v_hat)+self.eps)

class CNN:
    def __init__(self, kernel, stride=1, pad=0):
        self.kernel = kernel
        self.stride = stride
        self.pad = pad
        self.k_flatten = cp.reshape(kernel, newshape=(kernel.shape[0], -1))

    def im2col_cupy(self, X_train):
        N, C, H, W = X_train.shape
        OC, IC, HH, WW = self.kernel.shape

        out_h, out_w = (H-HH+2*self.pad)//self.stride + 1, (W-WW+2*self.pad)//self.stride + 1
        
        if self.pad > 0:
            X_padded = cp.pad(X_train, pad_width=[(0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)], mode='constant')
        else:
            X_padded = X_train

        columns = cp.array([
            X_padded[:, :, i*self.stride:i*self.stride+HH, j*self.stride:j*self.stride+WW]
            for i in range(0, out_h)
            for j in range(0, out_w)
        ])

        try:
            with cp.cuda.Device(0):
                a = cp.matmul(self.k_flatten, columns.reshape((C*HH*WW, N*out_h*out_w))).reshape((N, OC, out_h, out_w))
        except cp.cuda.memory.OutOfMemoryError:
            r = cp.array([])
            col = columns.reshape((C*HH*WW, N*out_h*out_w))
            chunks = 5
            arrays = cp.array([col[:, i*(col.shape[1])//chunks:(i+1)*(col.shape[1]//chunks)] for i in range(chunks)])
            for i in range(chunks):
                m = cp.matmul(self.k_flatten, arrays[i, :, :])
                r = cp.append(r, m)
                del m
                gc.collect()
            a = cp.concatenate(r, axis=1).reshape((N, OC, out_h, out_w))
        
        return columns.reshape((N, C, HH, WW, out_h, out_w)), a

    def col2im(self, cols, input):
        N, C, H, W = input.shape
        _, _, HH, WW = self.kernel.shape
        out_h = (H + 2*self.pad - HH) // self.stride + 1
        out_w = (W + 2*self.pad - WW) // self.stride + 1
        cols = cols.reshape(N*C, HH*WW, out_h, out_w)
        X = cp.zeros((N, C, H + 2*self.pad, W + 2*self.pad))

        for i in range(HH):
            for j in range(WW):
                X[:, :, i:i+self.stride*out_h:self.stride, j:j+self.stride*out_w:self.stride] += cols[:, i*WW+j, :, :].reshape(N, C, out_h, out_w)

        if self.pad > 0:
            X = X[:, :, self.pad:-self.pad, self.pad:-self.pad]
        return X

    def conv2d_backward(self, dout, X):
        N, C, H, W = X.shape
        F, _, HH, WW = self.kernel.shape

        out_h, out_w = (H-HH+2*self.pad)//self.stride + 1, (W-WW+2*self.pad)//self.stride + 1
        cols, _ = self.im2col_cupy(X)

        db = cp.sum(dout, axis=(0, 2, 3))
        dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(F, -1)
        cols_reshaped = cols.transpose(1, 2, 3, 0, 4, 5).reshape(-1, N*out_h*out_w)

        dW = cp.matmul(dout_reshaped, cols_reshaped.T).reshape(self.kernel.shape)
        dcols = cp.matmul(self.k_flatten.T, dout_reshaped).reshape(N, C, HH, WW, out_h, out_w)

        dX = self.col2im(dcols, X)
        assert dX.shape == X.shape
        assert dW.shape == self.kernel.shape

        return dX, dW, db

class MaxPooling:
    def __init__(self, size, stride):
        self.size = size
        self.stride = stride

    def forward(self, input):
        batch_size, in_channel, in_height, _ = input.shape[:]
        output_size = (in_height - self.size) // self.stride

        output = cp.array([
            input[:, :, i * self.stride:i * self.stride + self.size, j * self.stride:j * self.stride + self.size]
            for i in range(output_size) for j in range(output_size)
        ]).reshape((batch_size, in_channel, output_size, output_size, self.size, self.size)).max(axis=(4, 5))

        assert output.shape == (batch_size, in_channel, output_size, output_size)

        return output

    def backward(self, input, output, output_error_grad):
        batch_size, in_channel, in_height, in_width = input.shape[:]
        output_height, output_width = output.shape[2], output.shape[3]
        input_error_grad = cp.zeros_like(input)

        for i in range(output_height):
            for j in range(output_width):
                region = input[:, :, i * self.stride:(i * self.stride) + self.size, j * self.stride:(j * self.stride) + self.size]
                max_region = cp.max(region, axis=(2, 3), keepdims=True)
                mask = (region == max_region)
                input_error_grad[:, :, i * self.stride:(i * self.stride) + self.size,
                                 j * self.stride:(j * self.stride) + self.size] += mask * output_error_grad[:, :, i, j][:, :, None, None]

        return input_error_grad

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
    
    def softmax(self, x):
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
        de_flattened = cp.matmul(dW1, dz1.T)

        assert dW1.shape==self.W1.shape
        assert dW2.shape==self.W2.shape
        assert dW3.shape==self.W3.shape

        # Update weights and biases
        return dW1, dW2, dW3, de_flattened

# Define the CNN architecture
class SimpleCNN:
    def __init__(self):
        # Define layers
        self.conv1 = CNN(kernel=cp.random.randn(16, 3, 3, 3), stride=1, pad=1)
        self.conv2 = CNN(kernel=cp.random.randn(32, 16, 3, 3), stride=1, pad=1)
        self.pool = MaxPooling(size=2, stride=2)
        self.mlp = MLP(input_size=5408, hidden_size=128, output_size=10)
        self.parameters = (self.conv1.kernel, self.conv2.kernel, self.mlp.W1, self.mlp.b1, self.mlp.W2, self.mlp.b2, self.mlp.W3, self.mlp.b3)
        # Initialize Adam optimizer
        self.optimizer = AdamOptimizer(
            params=self.parameters
        )

    def forward(self, X):
        # Forward pass through the network
        _, out_1 = self.conv1.im2col_cupy(X)
        _, out_2 = self.conv2.im2col_cupy(out_1)
        out_pool_1 = self.pool.forward(out_2)
        global out_flattened
        out_flattened = out.reshape(out_pool_1.shape[0], -1)  # Flatten
        #Shape after flatten = (10,32,13,13)
        out = self.mlp.forward(out_flattened)
        return out

    def backward(self, X, y):
        # Backward pass through the network
        # Compute gradients
        dW1, dW2, dW3, de = self.mlp.backward(out_flattened, y)
        # Update parameters using Adam optimizer
        dX_conv2, dW_conv2, db_conv2 = self.conv2.conv2d_backward(de, X)
        dX_pool_1 = self.pool.backward(input=self)


# Example usage
cnn = SimpleCNN()
X = cp.random.randn(10, 3, 28, 28)  # Example input
y = cp.eye(10)[np.random.choice(10, 10)]  # Example one-hot encoded labels

# Forward pass
output = cnn.forward(X)

# Backward pass
cnn.backward(X, y)