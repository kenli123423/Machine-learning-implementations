import cupy as cp 

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
#For testing usage 
X = cp.random.randn(100 ,3, 30, 30)
Y = cp.random.randn(100 ,3, 27 ,27)
o = cp.random.randn(100 ,3, 27, 27)
maxpool = MaxPooling(size=2, stride=2)
output = maxpool.forward(X)
input_error_grd = maxpool.backward(X, output, o)
