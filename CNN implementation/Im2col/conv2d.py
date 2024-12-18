import cupy as cp 
import time 
import numpy as np 
import sys 
import gc
gc.enable()
'''
Note : This is an implementation to the im2col and col2im technique of Convolutional Neural Network, they are much faster than the for-loop approaches, free to use them for your own project
This code uses cupy and GPU, please ensure you have installed cupy, cuda and other necessary toolkit to run this code
Reference : Neural network and deep learning (2023,C.C.Aggarwal)
'''

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
        #Return the columns, convolved feature map and flattened kernel for backpropagation 
    
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

# Testing the function with arbitrary X_train and kernels 
X_train = cp.random.randn(1000, 3, 30, 30)
kernel = cp.random.randn(100, 3, 3, 3)
start = time.time()
columns, k_flatten , a = im2col_cupy(X_train, kernel, stride=1, pad=1)
dout = cp.random.randn(*a.shape)
dX, dW, db = conv2d_backward(dout, X_train, kernel, stride=1, pad=1)
end = time.time()
print(f"Time elapsed : {end-start}")
