import numpy as np
"""
Implementation of Mixture of Gaussian network with relu, softplus and softmax activation.

"""
K = 8
input_size = 11
batch_size = 300 
mode = "Normal"
#mode = Normal/Xavier/He
hidden_size = 100
number_hidden = 5
output_size = 3*K
hidden = [(hidden_size, hidden_size) for _ in range(number_hidden-1)]
size = [(input_size, hidden_size)] + hidden + [(hidden_size, output_size)]

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def sample(mean, var, mix):
    """Sample from mixture model for each data point in batch.
    
    Args:
        mean (np.array): Shape (batch_size, K)
        var (np.array): Shape (batch_size, K)
        mix (np.array): Shape (batch_size, K), mixing coefficients per data point
    
    Returns:
        np.array: Sampled values, shape (batch_size, 1)
    """
    assert mean.shape == (batch_size, K), f"Mean shape {mean.shape} != {(batch_size, K)}"
    assert var.shape == (batch_size, K), f"Var shape {var.shape} != {(batch_size, K)}"
    assert mix.shape == (batch_size, K), f"Mix shape {mix.shape} != {(batch_size, K)}"
    assert np.allclose(np.sum(mix, axis=1), 1.0), "Mixing coefficients must sum to 1 per data point"

    samples = np.zeros((batch_size, 1))
    for i in range(batch_size):
        component = np.random.choice(K, p=mix[i])
        samples[i] = np.random.normal(mean[i, component], np.sqrt(var[i, component]))
    return samples

def pdf(y: np.array, mean, var) -> np.array:
    """Compute Gaussian PDF for each element in batch."""
    return (1.0 / np.sqrt(2 * np.pi * var)) * np.exp(-0.5 * (y - mean)**2 / var)

def forward_pass(X, W, b):
    """Forward pass with activation cache storage."""
    cache = X
    layer_cache = [X]
    pre_activations = []  # Store pre-activation values for softplus derivative
    
    # Hidden layers with ReLU
    for i in range(len(W)-1):
        z = np.dot(cache, W[i]) + b[i]
        cache = relu(z)
        layer_cache.append(cache)
        pre_activations.append(z)  # For ReLU derivative
    
    # Output layer (no activation here)
    z = np.dot(cache, W[-1]) + b[-1]
    layer_cache.append(z)
    cache = z
    
    # Split outputs
    means = z[:, :K]
    var_raw = z[:, K:2*K]  # Pre-softplus values
    logits = z[:, 2*K:]
    
    # Apply activations
    var = np.log(1 + np.exp(var_raw))  # Softplus
    mix_coeffs = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # Numerical stability
    mix_coeffs /= np.sum(mix_coeffs, axis=1, keepdims=True)
    
    assert means.shape == (X.shape[0], K)
    assert var.shape == (X.shape[0], K)
    assert mix_coeffs.shape == (X.shape[0], K)
    
    return means, var, mix_coeffs, layer_cache, var_raw, logits

def loss(y_true, mean, var, mix):
    """Compute negative log likelihood loss."""
    y_true = y_true.reshape(-1, 1)  # Ensure (batch_size, 1)
    likelihood = np.sum(mix * pdf(y_true, mean, var), axis=1)
    return -np.mean(np.log(likelihood + 1e-8)), likelihood

# Initialize parameters
def initialize_parameters(mode):
    if mode == "Normal":
        W = [np.random.randn(*s)*0.01 for s in size]
        b = [np.random.randn(s[1])*0.01 for s in size]
    elif mode == "Xavier":
        W = [np.random.normal(loc=0, scale=np.sqrt(2/(input_size+output_size)), size=s) for s in size]
        b = [np.random.normal(loc=0, scale=np.sqrt(2/(input_size+output_size)), size=s[1]) for s in size]
    elif mode == "He":
        W = [np.random.normal(loc=0, scale=2/input_size, size=s) for s in size]
        b = [np.random.normal(loc=0, scale=2/input_size, size=s[1]) for s in size]
    return W,b
W,b = initialize_parameters(mode)

# Generate test data
np.random.seed(42)
X_test = np.random.randn(batch_size, input_size)
y_test = np.random.randn(batch_size, 1)

# Test forward pass
means, variances, mix_coeffs, layer_cache, var_raw, logits = forward_pass(X_test, W, b)

# Test loss calculation
current_loss, likelihood = loss(y_test, means, variances, mix_coeffs)
print(f"Initial loss: {current_loss:.4f}")

# Calculate gradients ----------------------------------------------------------
# Gradient calculations with activation derivatives
batch_size = X_test.shape[0]

# 1. Gradient for means (no activation derivative needed)
dL_dmu = np.zeros_like(means)
for k in range(K):
    dL_dmu[:, k] = (mix_coeffs[:, k] * (means[:, k] - y_test.flatten()) / variances[:, k]) / likelihood


# 2. Gradient for variances (post-softplus)
dL_dvar = np.zeros_like(variances)
for k in range(K):
    dL_dvar[:, k] = (mix_coeffs[:, k] * (
        (y_test.flatten() - means[:, k])**2 / (2 * variances[:, k]**2)
        - 1/(2 * variances[:, k])
    ) / likelihood)


# Multiply by softplus derivative (1 - exp(-var))
softplus_derivative = 1 - np.exp(-variances)
dL_dvar_raw = dL_dvar * softplus_derivative

# 3. Gradient for logits (pre-softmax)
dL_dpi = (mix_coeffs * (pdf(y_test, means, variances) / (likelihood.reshape(-1, 1) + 1e-8)))
dL_dlogits = dL_dpi - mix_coeffs * np.sum(dL_dpi, axis=1, keepdims=True)

# Combine gradients
d_o = np.hstack([dL_dmu, dL_dvar_raw, dL_dlogits])

# Backprop through layers -----------------------------------------------------
dW = []
db = []
d_prev = d_o

# Reverse through layers
for i in reversed(range(len(W))):
    # Current layer gradients
    dL_dW = np.dot(layer_cache[i].T, d_prev)
    dL_db = np.sum(d_prev, axis=0)
    
    dW.append(dL_dW)
    db.append(dL_db)
    
    # Propagate gradient to previous layer
    if i > 0:
        d_prev = np.dot(d_prev, W[i].T)
        d_prev *= relu_derivative(layer_cache[i])

# Reverse gradients to match original order
dW = dW[::-1]
db = db[::-1]

# Parameter update checks
assert len(dW) == len(W), "Gradient/Weight length mismatch"
assert len(db) == len(b), "Gradient/Bias length mismatch"
for i, (dw, w) in enumerate(zip(dW, W)):
    assert dw.shape == w.shape, f"Gradient dW[{i}] shape mismatch {dw.shape} vs {w.shape}"
for i, (db_, b_) in enumerate(zip(db, b)):
    assert db_.shape == b_.shape, f"Gradient db[{i}] shape mismatch {db_.shape} vs {b_.shape}"
