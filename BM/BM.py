import cupy as cp

def BM(X: cp.array) -> cp.array:
    gamma = 0.9
    beta = 0.99
    epsilon = 1e-9

    # Compute mean and variance along the batch axis (axis=0)
    m = cp.mean(X, axis=0)
    var = cp.var(X, axis=0)

    # Normalize
    batch = (X - m) / cp.sqrt(var + epsilon)
    
    # Scale and shift
    o = gamma * batch + beta
    
    return o

# Example usage
X = cp.random.randn(100, 4200)  # Example input with batch size 10 and 5 features
output = BM(X)



