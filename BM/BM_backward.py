import cupy as cp

class BatchNormalization:
    def __init__(self, num_features, epsilon=1e-5, momentum=0.9):
        self.epsilon = epsilon
        self.momentum = momentum
        self.gamma = cp.ones(num_features)
        self.beta = cp.zeros(num_features)
        self.running_mean = cp.zeros(num_features)
        self.running_var = cp.ones(num_features)

    def forward(self, x, training=True):
        if training:
            batch_mean = cp.mean(x, axis=0)
            batch_var = cp.var(x, axis=0)
            self.x_centered = x - batch_mean
            #We will always use the (v-mean) in backward phase, so define it first
            self.stddev_inv = 1.0 / cp.sqrt(batch_var + self.epsilon)
            #Add epsilon to prevent division by 0 
            #stddev_inv = inverse of standard deviation 
            x_norm = self.x_centered * self.stddev_inv
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
        else:
            x_norm = (x - self.running_mean) / cp.sqrt(self.running_var + self.epsilon)

        self.out = self.gamma * x_norm + self.beta
        return self.out

    def backward(self, dout):
        N, D = dout.shape

        x_norm = self.x_centered * self.stddev_inv
        dbeta = cp.sum(dout, axis=0)
        dgamma = cp.sum(dout * x_norm, axis=0)

        dx_norm = dout * self.gamma
        dvar = cp.sum(dx_norm * self.x_centered * -0.5 * self.stddev_inv**3, axis=0)
        dmean = cp.sum(dx_norm * -self.stddev_inv, axis=0) + dvar * cp.mean(-2.0 * self.x_centered, axis=0)

        dx = (dx_norm * self.stddev_inv) + (dvar * 2.0 * self.x_centered / N) + (dmean / N)

        return dx, dgamma, dbeta

# Example usage
num_features = 10
batch_size = 5

# Create a BatchNormalization layer
bn_layer = BatchNormalization(num_features)

# Simulate a batch of data
x = cp.random.randn(batch_size, num_features)

# Forward pass
out = bn_layer.forward(x, training=True)

# Simulate gradient from next layer
dout = cp.random.randn(batch_size, num_features)

# Backward pass
dx, dgamma, dbeta = bn_layer.backward(dout)
print(dx.shape)