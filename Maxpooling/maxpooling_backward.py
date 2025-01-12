import cupy as cp 

def maxpooling_backward(input, output, output_error_grad, size, stride):
        batch_size, in_channel, in_height, in_width = input.shape[:]
        output_height, output_width = output.shape[2], output.shape[3]
        input_error_grad = cp.zeros_like(input)
        for i in range(output_height):
            for j in range(output_width):
                region = input[:, :, i * stride:(i * stride) + size, j * stride:(j * stride) + size]
                max_region = cp.max(region, axis=(2, 3), keepdims=True)
                # Calculating max-pooling is easy as we demonstrated, so we simply add one line here to calculate maximum rather than calling function again.
                # We need the feature map before and after max pooling for comparison of position of maximum values
                # There is no gradient wrt to non-maximum values, the neuron who achieved maximum in max-pooling stage will gain gradient, all other neurons will get a value of zero.
                mask = (region == max_region)
                # Declare mask as a generator
                input_error_grad[:, :, i * stride:(i * stride) + size,
                j * stride:(j * stride) + size] += mask * output_error_grad[:, :, i, j][:, :, None, None]
        return input_error_grad

X = cp.random.randn(100 ,3, 30, 30)
Y = cp.random.randn(100 ,3, 27 ,27)
o = cp.random.randn(100 ,3, 27, 27)
maxpooling_backward(X, Y, o, size=3, stride=1)
