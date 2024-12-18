from logging import WARNING
import cupy as cp 
import time 
start = time.time()
def im2col_cupy(X_train, kernel, stride, pad):
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

    except cp.cuda.memory.OutOfMemoryError:
        r = []
        col = columns.reshape((C*HH*WW, N*out_h*out_w))
        chunks = 2
        arrays = cp.array([col[:, i*(col.shape[1])//chunks:(i+1)*(col.shape[1]//chunks)] for i in range(chunks)])
        #Transform into array with shape of (chunks, ....)
        for i in range(chunks):
            with cp.cuda.Device(0):
                m = cp.matmul(k_flatten, arrays[i, :, :])
            r.append(m)
            #Concatenate the small chunks together to make the original matrix 
        del arrays 
        a = cp.concatenate(a, axis=1).reshape((N, OC, out_h, out_w))
        
    return columns.reshape((N, C, HH, WW, out_h, out_w)) , k_flatten, a

def col2im(cols:cp.array, input, kernel, stride, pad):
    N, C, H, W = input.shape
    _, _, HH, WW = kernel.shape 
    out_h = (H+2*pad-HH)//stride + 1
    out_w = (W+2*pad-WW)//stride + 1
    cols = cols.reshape((N, C, HH, WW, out_h, out_w))
    #Place an assertion to detect incompitable shapes early 

    assert cols.shape == (N, C, HH, WW, out_h, out_w)

    X = cp.zeros((N, C, H+2*pad ,W+2*pad)) 
    #Create img from cp.zeros
    #Transform the columns back to image form, create indicies 
    for i in range(HH):
        for j in range(WW):
            X[:, :, i:i+stride*out_h:stride, j:j+stride*out_w:stride] += cols[:, :, i, j, :, :]

    if pad>0:
        X = X[:, :, pad:-pad, pad:-pad]

    return X

# Testing the function
X_train = cp.random.randn(10000, 3, 30, 30)
kernel = cp.random.randn(500, 3, 3, 3)
columns, k_flatten , a = im2col_cupy(X_train, kernel, stride=1, pad=1)
print(columns.shape, k_flatten.shape, a.shape)

b = col2im(cols=columns, input=X_train, kernel=kernel, stride=1, pad=1)
print(b.shape)
end = time.time()
print(end - start)

    