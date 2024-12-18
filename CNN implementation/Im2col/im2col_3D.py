import cupy as cp
import time 
# Implementation of 3D im2col

'''
X_train = (N, C, H, W), kernel = (OC, IC, HH, WW)
'''

stride = 1
pad = 1
#Specify the stride and padding

X_train = cp.random.randn(10, 3, 30, 30)
kernel = cp.random.randn(8, 3, 3, 3)
dout = cp.random.randn(10, 8, 30, 30)
N, C, H, W = X_train.shape 
OC, IC, HH, WW = kernel.shape 

assert kernel.shape[1] == X_train.shape[1]

#Their in_channel must be the same otherwise convolution cannot be done 

#Flatten the kernel 
k_flatten = cp.reshape(kernel, newshape=(OC, -1))

#Calculate output shape with given padding and strides 
out_h, out_w = (H-HH+2*pad)//stride + 1, (W-WW+2*pad)//stride + 1

if pad>0: 
    X_padded = cp.pad(X_train, pad_width=[(0, 0), (0, 0), (pad, pad), (pad, pad)], mode='constant')
else :
    X_padded = X_train 
#Pad the input if necessary

columns = cp.array([X_padded[:, :, i*stride:i*stride+HH, j*stride:j*stride+WW]
                    for i in range(0, out_h)
                    for j in range(0, out_w)])
col = columns.reshape((C*HH*WW, N*out_h*out_w))

#Columns shape (C*HH*WW, N*out_h*out_w)
a = cp.matmul(k_flatten, col).reshape((N, OC, out_h, out_w))

#Testing backward 
d = dout.transpose(0, 2, 3, 1).reshape(OC, -1)
dw = cp.matmul(d, col.T).reshape(OC, IC, HH, WW)
print(dw.shape)