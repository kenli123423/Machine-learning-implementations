import cupy as cp 
def maxpooling(input, size, stride):
        batch_size, in_channel, in_height, _ = input.shape[:]
        output_size = (in_height - size)//stride

        o = cp.array([
                input[:, :, i*stride:i*stride+size, j*stride:j*stride+size]
                      for i in range(output_size) for j in range(output_size)]
        ).reshape((batch_size, in_channel, output_size, output_size, size, size)).max(axis=(4,5))

        assert o.shape == (batch_size, in_channel, output_size, output_size)

        return o
X = cp.random.randn(1000, 3, 30, 30)
output = maxpooling(X, 3, 1)
