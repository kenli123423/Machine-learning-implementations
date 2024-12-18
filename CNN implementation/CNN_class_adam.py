import gc
import cupy as cp 

class CNN:
    def __init__(self, kernel, stride=1, pad=0):
        self.kernel = kernel
        self.stride = stride
        self.pad = pad
        self.A = cp.zeros_like(kernel)
        self.F = cp.zeros_like(kernel)
        self.rho = 0.9 
        self.rho_1 = 0.95
        self.lr = 0.05

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
    
    def params_udpate(self, kernel, dW, db):
        self.A = self.rho*self.A + (1-self.rho)*cp.square(dW)
        self.F = self.rho_1*self.F + (1-self.rho_1)*dW 
        kernel -= self.lr*self.F/cp.sqrt(self.A)

# Example usage
kernel = cp.random.randn(10, 3, 3, 3) 
cnn = CNN(kernel=kernel, stride=1, pad=1) 
X = cp.random.randn(100, 3, 32, 32) 
dout = cp.random.randn(100, 10, 32, 32) 