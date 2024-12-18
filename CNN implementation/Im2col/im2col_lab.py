import cupy as cp 
import time
X = cp.random.randn(100, 50, 10)
Y = cp.random.randn(50, 100, 20)
start = time.time()
a = cp.tensordot(X, Y, axes=([1,0],[0,1]))
end=time.time()
print(end-start)
del X, Y