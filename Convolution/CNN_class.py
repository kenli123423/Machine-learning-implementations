'''
Note this is an example usage of CNN, you can change the data/optimization methods/architecture on your own to fit your needs.
Copyright (C) <2024> <Pony>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

from multiprocessing import pool
import cupy as cp 
import time 
import numpy as np 
import sys 
import gc
gc.enable()

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

    def backward(self, X, y, shape, learning_rate=1e-3):
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

        self.deflatten = cp.matmul(da1, self.W1.T).reshape(shape)

        # Update weights and biases
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3

def im2col_cupy(X_train:cp.array, kernel:cp.array, stride:int, pad:int) -> tuple:
    #Declare data type 

    # Ensure the kernel and input have matching in_channels
    # Flatten the kernel
    k_flatten = cp.reshape(kernel, newshape=(kernel.shape[0], -1))

    # Calculate output shape with given padding and strides
    N, C, H, W = X_train.shape
    OC, IC, HH, WW = kernel.shape
    
    #Input channel of kernel and channel of feature map input channel must be the same 

    out_h, out_w = (H-HH+2*pad)//stride + 1, (W-WW+2*pad)//stride + 1

    # Pad the input if necessary
    if pad > 0:
        X_padded = cp.pad(X_train, pad_width=[(0, 0), (0, 0), (pad, pad), (pad, pad)], mode='constant')
    else:
        X_padded = X_train

    # Generate column matrix using efficient CuPy operations
    columns = cp.array([
        X_padded[:, :, i*stride:i*stride+HH,j*stride:j*stride+WW]
        for i in range(0, out_h)
        for j in range(0, out_w)
    ])
 
    # Manually delete the variables to save memory 

    #Divide matrix into chunks if out_of_memory error occured
    try:
        with cp.cuda.Device(0):
            a = cp.matmul(k_flatten, columns.reshape((C*HH*WW, N*out_h*out_w))).reshape((N,OC,out_h,out_w))

    #Divide the matrix into chunks and do cp.matmul on each chunk and k_flatten, then use cp.concantenate to combine them together

    except cp.cuda.memory.OutOfMemoryError:
        r = cp.array([])
        col = columns.reshape((C*HH*WW, N*out_h*out_w))
        chunks = 5
        arrays = cp.array([col[:, i*(col.shape[1])//chunks:(i+1)*(col.shape[1]//chunks)] for i in range(chunks)])
        #Transform into array with shape of (chunks, ....)
        for i in range(chunks):
            m = cp.matmul(k_flatten, arrays[i, :, :])
            r = cp.append(r, m) 
            del m 
            gc.collect()

            #Concatenate the small chunks together to make the original matrix  
        a = cp.concatenate(r, axis=1).reshape((N, OC, out_h, out_w))
        
    return columns.reshape((N, C, HH, WW, out_h, out_w)) , k_flatten, a

def col2im(cols:cp.array, input:cp.array, kernel:cp.array, stride:int, pad:int) -> cp.array:

    N, C, H, W = input.shape
    _, _, HH, WW = kernel.shape 
    out_h = (H+2*pad-HH)//stride + 1
    out_w = (W+2*pad-WW)//stride + 1
    cols = cols.reshape(N*C, HH*WW, out_h, out_w)
    X = cp.zeros((N,C,H+2*pad,W+2*pad))
    cols : cp.arary 
    X : cp.array 

    #Place an assertion to detect incompitable shapes early 

    for i in range(HH): 
        for j in range(WW): 
            X[:, :, i:i+stride*out_h:stride, j:j+stride*out_w:stride] += cols[:, i*WW+j, :, :].reshape(N, C, out_h, out_w)

    if pad>0:
        X = X[:, :, pad:-pad, pad:-pad]
    return X 

def conv2d_backward(dout, X, kernel, stride, pad):

    # Dout : Upstream derivative, (N, F, out_h, out_w)
    # X : Input feature map shape, (N, C, H, W)
    # W : Kernel, (OC, IC, HH, WW)
    # stride : Number of pixels between adjacent receptive fields 
    # Pad : The number of pixels that will be used for zero padding

    N, C, H, W = X.shape 
    F, _, HH, WW = kernel.shape

    #Compute the output shape 
    out_h, out_w = (H-HH+2*pad)//stride +1, (W-WW+2*pad)//stride + 1 
    cols, k_flatten, _ = im2col_cupy(X, kernel, stride, pad)

    db = cp.sum(dout, axis=(0,2,3))

    dout_reshaped = dout.transpose(1,2,3,0).reshape(F, -1)
    cols_reshaped = cols.transpose(1,2,3,0,4,5).reshape(-1, N*out_h*out_w)

    dW = cp.matmul(dout_reshaped, cols_reshaped.T).reshape(kernel.shape)
    #Compute the gradient wrt to the input layer
    dcols = cp.matmul(k_flatten.T, dout_reshaped).reshape(N, C, HH, WW, out_h, out_w)

    dX = col2im(dcols, X, kernel, stride, pad)
    assert dX.shape == X.shape
    assert dW.shape == kernel.shape 

    return dX, dW, db

class CNN:
    def __init__(self, input_shape, num_classes):
        # Initialize layers
        self.conv_kernel = cp.random.randn(8, input_shape[1], 3, 3) * cp.sqrt(2. / (3 * 3 * input_shape[1]))
        self.conv_stride = 1
        self.conv_pad = 1
        
        self.pool_size = 2
        self.pool_stride = 2
        
        # Calculate the size of the flattened layer after pooling
        conv_output_height = (input_shape[2] - 3 + 2 * self.conv_pad) // self.conv_stride + 1
        conv_output_width = (input_shape[3] - 3 + 2 * self.conv_pad) // self.conv_stride + 1
        pool_output_height = (conv_output_height - self.pool_size) // self.pool_stride + 1
        pool_output_width = (conv_output_width - self.pool_size) // self.pool_stride + 1
        
        flattened_size = 8 * pool_output_height * pool_output_width
        
        # Initialize MLP
        self.mlp = MLP(flattened_size, 100, num_classes)
    
    def forward(self, X):
        # Convolutional layer
        self.conv_out, self.conv_cols, self.a = im2col_cupy(X, self.conv_kernel, self.conv_stride, self.conv_pad)
        
        # Max pooling layer
        self.pool_out = self.maxpool_forward(self.a)
        self.flat_shape = self.pool_out.shape 
        # Flatten
        self.flattened = self.pool_out.reshape(X.shape[0], -1)
        
        # Fully connected layer
        output = self.mlp.forward(self.flattened)
        
        return output
    
    def backward(self, X, y, learning_rate=1e-3):
        # Backward pass through MLP
        self.mlp.backward(self.flattened, y, shape=self.flat_shape, learning_rate=learning_rate)
        
        # Backward pass through max pooling
        pool_error_grad = self.mlp.deflatten
        conv_error_grad = self.maxpool_backward(self.a, self.pool_out, pool_error_grad)
        
        # Backward pass through convolutional layer
        dX, dW, db = conv2d_backward(conv_error_grad, X, self.conv_kernel, self.conv_stride, self.conv_pad)
        
        # Update convolutional kernel
        self.conv_kernel -= learning_rate * dW
    
    def maxpool_forward(self, input):
        batch_size, in_channel, in_height, in_width = input.shape
        output_height = (in_height - self.pool_size) // self.pool_stride + 1
        output_width = (in_width - self.pool_size) // self.pool_stride + 1
        
        output = cp.array([
            input[:, :, i * self.pool_stride:i * self.pool_stride + self.pool_size, j * self.pool_stride:j * self.pool_stride + self.pool_size]
            for i in range(output_height) for j in range(output_width)
        ]).reshape((batch_size, in_channel, output_height, output_width, self.pool_size, self.pool_size)).max(axis=(4, 5))
        
        return output
    
    def maxpool_backward(self, input, output, output_error_grad):
        batch_size, in_channel, in_height, in_width = input.shape
        output_height, output_width = output.shape[2], output.shape[3]
        input_error_grad = cp.zeros_like(input)
        
        for i in range(output_height):
            for j in range(output_width):
                region = input[:, :, i * self.pool_stride:(i * self.pool_stride) + self.pool_size, j * self.pool_stride:(j * self.pool_stride) + self.pool_size]
                max_region = cp.max(region, axis=(2, 3), keepdims=True)
                mask = (region == max_region)
                input_error_grad[:, :, i * self.pool_stride:(i * self.pool_stride) + self.pool_size,
                                 j * self.pool_stride:(j * self.pool_stride) + self.pool_size] += mask * output_error_grad[:, :, i, j][:, :, None, None]
        
        return input_error_grad
    
    def train(self, X, y, epochs=10, learning_rate=1e-3):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, learning_rate)
            loss = self.mlp.cross_entropy_loss(output, y)
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss}')

# Example usage
input_shape = (5000, 1, 28, 28)  # Batch size, channels, height, width
num_classes = 10
cnn = CNN(input_shape, num_classes)

# Dummy data
X_train = cp.random.randn(*input_shape)
y_train = cp.eye(num_classes)[cp.random.choice(num_classes, input_shape[0])]

cnn.train(X_train, y_train, epochs=10, learning_rate=1e-3)
