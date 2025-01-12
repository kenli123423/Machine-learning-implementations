import cupy as cp
import cupyx
import cupyx.distributed
import cupyx.distributed.array as cda 
import cupyx.profiler
import numpy as np
import scipy 
from scipy.linalg import blas
import time
import gc
gc.enable()
cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)

def cupy_matmul(a:cp.array,b:cp.array):
    _, ca = a.shape
    rb, _ = b.shape
    assert ca == rb, f"{ca!=rb}"
    return cp.matmul(a,b,dtype=cp.float16, casting='same_kind', order='C')


def im2col_cupy_optimized(X_train: cp.array, kernel: cp.array, stride: int, pad: int) -> tuple:
    # Flatten the kernel
    k_flatten = cp.reshape(kernel, newshape=(kernel.shape[0], -1))

    # Calculate output shape
    N, C, H, W = X_train.shape
    OC, IC, HH, WW = kernel.shape
    out_h, out_w = (H - HH + 2 * pad) // stride + 1, (W - WW + 2 * pad) // stride + 1

    # Pad the input
    if pad > 0:
        X_padded = cp.pad(X_train, pad_width=[(0, 0), (0, 0), (pad, pad), (pad, pad)], mode='constant')
    else:
        X_padded = X_train

    # Generate column matrix using as_strided
    start_1 = time.time()
    columns = cp.lib.stride_tricks.as_strided(
        X_padded,
        shape=(N, C, out_h, out_w, HH, WW),
        strides=(
            X_padded.strides[0],
            X_padded.strides[1],
            X_padded.strides[2] * stride,
            X_padded.strides[3] * stride,
            X_padded.strides[2],
            X_padded.strides[3]
        )
    ).reshape(C*HH*WW, N*out_h*out_w)

    # Perform matrix multiplication
    try:
        with cp.cuda.Device(0):
            with cp.cuda.Stream() as stream:
                a = cupy_matmul(k_flatten, columns)
            stream.synchronize()

    except cp.cuda.memory.OutOfMemoryError:
        chunks = 5
        r = []
        for i in range(chunks):
            chunk = columns[i * (columns.shape[0] // chunks):(i + 1) * (columns.shape[0] // chunks), :]
            m = cp.matmul(k_flatten, chunk.T)
            r.append(m)
            del m
            gc.collect()
        a = cp.concatenate(r, axis=1).reshape(N, OC, out_h, out_w)

    return columns.reshape(N, C, HH, WW, out_h, out_w), k_flatten, a

def col2im_optimized(cols: cp.array, input: cp.array, kernel: cp.array, stride: int, pad: int) -> cp.array:
    N, C, H, W = input.shape
    _, _, HH, WW = kernel.shape
    out_h = (H + 2 * pad - HH) // stride + 1
    out_w = (W + 2 * pad - WW) // stride + 1
    cols = cols.reshape(N * C, HH * WW, out_h, out_w)
    X = cp.zeros((N, C, H + 2 * pad, W + 2 * pad))

    for i in range(HH):
        for j in range(WW):
            X[:, :, i:i + stride * out_h:stride, j:j + stride * out_w:stride] += cols[:, i * WW + j, :, :].reshape(N, C, out_h, out_w)

    if pad > 0:
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
    cols, k_flatten, _ = im2col_cupy_optimized(X, kernel, stride, pad)

    db = cp.sum(dout, axis=(0,2,3))

    dout_reshaped = dout.transpose(1,2,3,0).reshape(F, -1)
    cols_reshaped = cols.transpose(1,2,3,0,4,5).reshape(-1, N*out_h*out_w)

    dW = cp.matmul(dout_reshaped, cols_reshaped.T).reshape(kernel.shape)
    #Compute the gradient wrt to the input layer
    dcols = cp.matmul(k_flatten.T, dout_reshaped).reshape(N, C, HH, WW, out_h, out_w)

    dX = col2im_optimized(dcols, X, kernel, stride, pad)
    assert dX.shape == X.shape
    assert dW.shape == kernel.shape 

    return dX, dW, db


# Testing the function
X_train = cp.random.randn(5000, 3, 30, 30).astype(cp.float16)
kernel = cp.random.randn(100, 3, 3, 3).astype(cp.float16)
start = time.time()
_, _, a = im2col_cupy_optimized(X_train, kernel, stride=1, pad=1)
end = time.time()
cp.cuda.Stream.null.synchronize()
dout = cp.random.randn(*a.shape)
dX, dW, db = conv2d_backward(dout, X_train, kernel, stride=1, pad=1)
