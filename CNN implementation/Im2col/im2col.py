import cupy as cp 
import numpy as np 
import time 
#Approach 1 : Im2col 2D version, C=1

pad = 1
stride = 1
X_train = cp.random.randn(1, 1, 1000, 1000)
OC, IC, H, W = X_train.shape
X_padded = cp.pad(X_train, pad_width=[(0, 0), (0, 0), (pad, pad), (pad, pad)])
kernel = cp.random.randn(1, 1, 3, 3)
OCK, ICK, KH, KW = kernel.shape 
kernel_flatten = cp.ravel(kernel, order='F').reshape(9, 1)
#Calculate the out_H and out_W

X_padded = cp.array(X_padded.get())
kernel = cp.array(kernel.get())

out_H, out_W = (H-KH+2*pad)//stride + 1, (W-KW+2*pad)//stride + 1

#Calculate shapes after convolution

s = time.time()

columns = cp.array([X_padded[:, :, i:i+KH, j:j+KW]
                    for i in range(0, out_H, stride)
                    for j in range(0, out_W, stride)])
print(columns.shape)
breakpoint()
columns = cp.array(columns)

a = cp.matmul(columns.T, kernel_flatten).reshape(out_H, out_W)
end = time.time()
#Print the convolved feature map 

#Approach 2 : For loop approach
start_1 = time.time() 
col_matrix = [] 
for i in range(0, out_H, stride): 
    for j in range(0, out_W, stride): 
        patch = X_padded[:, i:i+KH, j:j+KW].flatten() 
        col_matrix.append(patch)
col_matrix = np.array(col_matrix.get())

end_1 = time.time()

print(end-s)
print(end_1 - start_1)