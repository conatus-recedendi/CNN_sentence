"""
Utility functions for CNN sentence classification with pure NumPy implementation
"""

import numpy as np


def softmax(x):
    """
    Compute softmax values for x.

    Args:
        x: numpy array

    Returns:
        softmax probabilities
    """
    # Subtract max for numerical stability
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def relu(x):
    """
    ReLU activation function.

    Args:
        x: numpy array

    Returns:
        ReLU output
    """
    return np.maximum(0, x)


def sigmoid(x):
    """
    Sigmoid activation function.

    Args:
        x: numpy array

    Returns:
        Sigmoid output
    """
    # Clip x for numerical stability
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def tanh(x):
    """
    Tanh activation function.

    Args:
        x: numpy array

    Returns:
        Tanh output
    """
    return np.tanh(x)


def identity(x):
    """
    Identity activation function.

    Args:
        x: numpy array

    Returns:
        Input unchanged
    """
    return x


def conv2d(input_data, filters, bias=None, stride=1, padding=0):
    """
    Optimized 2D convolution operation for sentence classification.
    
    Args:
        input_data: (batch_size, channels, height, width)
        filters: (num_filters, channels, filter_height, filter_width)
        bias: (num_filters,) or None
        stride: int
        padding: int
        
    Returns:
        Convolution output
    """
    batch_size, in_channels, in_height, in_width = input_data.shape
    num_filters, _, filter_height, filter_width = filters.shape
    
    # Calculate output dimensions
    out_height = (in_height + 2 * padding - filter_height) // stride + 1
    out_width = (in_width + 2 * padding - filter_width) // stride + 1
    
    # For sentence classification, we often have 1D convolution over sentences
    # This is optimized for that case
    if filter_width == in_width and padding == 0:
        # This is essentially 1D convolution over sentence length
        output = np.zeros((batch_size, num_filters, out_height, 1))
        
        for b in range(batch_size):
            for f in range(num_filters):
                for h in range(out_height):
                    h_start = h * stride
                    h_end = h_start + filter_height
                    
                    # Extract region and compute dot product
                    region = input_data[b, :, h_start:h_end, :]  # (channels, filter_height, width)
                    filter_f = filters[f]  # (channels, filter_height, filter_width)
                    
                    # Compute convolution as element-wise multiplication and sum
                    conv_val = np.sum(region * filter_f)
                    
                    if bias is not None:
                        conv_val += bias[f]
                    
                    output[b, f, h, 0] = conv_val
    else:
        # General case - use original implementation
        # Pad input if necessary
        if padding > 0:
            padded_input = np.pad(
                input_data, 
                ((0, 0), (0, 0), (padding, padding), (padding, padding)), 
                mode='constant'
            )
        else:
            padded_input = input_data
        
        # Initialize output
        output = np.zeros((batch_size, num_filters, out_height, out_width))
        
        # Perform convolution
        for b in range(batch_size):
            for f in range(num_filters):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * stride
                        h_end = h_start + filter_height
                        w_start = w * stride
                        w_end = w_start + filter_width
                        
                        # Extract region
                        region = padded_input[b, :, h_start:h_end, w_start:w_end]
                        
                        # Compute convolution
                        output[b, f, h, w] = np.sum(region * filters[f])
                        
                        # Add bias if provided
                        if bias is not None:
                            output[b, f, h, w] += bias[f]
    
    return output


def max_pool2d(input_data, pool_size, stride=None):
    """
    2D max pooling operation.

    Args:
        input_data: (batch_size, channels, height, width)
        pool_size: (pool_height, pool_width) or int
        stride: int or None (defaults to pool_size)

    Returns:
        Max pooling output
    """
    if isinstance(pool_size, int):
        pool_height = pool_width = pool_size
    else:
        pool_height, pool_width = pool_size

    if stride is None:
        stride = pool_height

    batch_size, channels, in_height, in_width = input_data.shape

    # Calculate output dimensions
    out_height = (in_height - pool_height) // stride + 1
    out_width = (in_width - pool_width) // stride + 1

    # Initialize output
    output = np.zeros((batch_size, channels, out_height, out_width))

    # Perform max pooling
    for b in range(batch_size):
        for c in range(channels):
            for h in range(out_height):
                for w in range(out_width):
                    h_start = h * stride
                    h_end = h_start + pool_height
                    w_start = w * stride
                    w_end = w_start + pool_width

                    # Extract region and take max
                    region = input_data[b, c, h_start:h_end, w_start:w_end]
                    output[b, c, h, w] = np.max(region)

    return output


def dropout_mask(shape, dropout_rate, rng=None):
    """
    Generate dropout mask.

    Args:
        shape: tuple of dimensions
        dropout_rate: float between 0 and 1
        rng: numpy random state

    Returns:
        Binary mask for dropout
    """
    if rng is None:
        rng = np.random

    # Generate binary mask (1 = keep, 0 = drop)
    mask = (rng.random(shape) > dropout_rate).astype(np.float32)
    return mask


def apply_dropout(x, dropout_rate, training=True, rng=None):
    """
    Apply dropout to input.

    Args:
        x: input array
        dropout_rate: float between 0 and 1
        training: bool, whether in training mode
        rng: numpy random state

    Returns:
        Dropout applied output
    """
    if not training or dropout_rate == 0:
        return x

    mask = dropout_mask(x.shape, dropout_rate, rng)
    # Scale by 1/(1-p) to maintain expected value
    return x * mask / (1 - dropout_rate)


def cross_entropy_loss(predictions, targets):
    """
    Compute cross entropy loss.

    Args:
        predictions: (batch_size, num_classes) probability distribution
        targets: (batch_size,) class indices

    Returns:
        Mean cross entropy loss
    """
    batch_size = predictions.shape[0]

    # Add small epsilon for numerical stability
    epsilon = 1e-15
    predictions = np.clip(predictions, epsilon, 1 - epsilon)

    # Compute cross entropy
    log_probs = np.log(predictions[np.arange(batch_size), targets])
    return -np.mean(log_probs)


def accuracy(predictions, targets):
    """
    Compute classification accuracy.

    Args:
        predictions: (batch_size, num_classes) probability distribution
        targets: (batch_size,) class indices

    Returns:
        Accuracy as float
    """
    predicted_classes = np.argmax(predictions, axis=1)
    return np.mean(predicted_classes == targets)


def clip_gradients(gradients, max_norm):
    """
    Clip gradients by norm.

    Args:
        gradients: list of gradient arrays
        max_norm: maximum allowed norm

    Returns:
        Clipped gradients
    """
    # Compute total norm
    total_norm = 0
    for grad in gradients:
        if grad is not None:
            total_norm += np.sum(grad**2)
    total_norm = np.sqrt(total_norm)

    # Clip if necessary
    if total_norm > max_norm:
        scale = max_norm / total_norm
        gradients = [grad * scale if grad is not None else None for grad in gradients]

    return gradients


def im2col(input_data, filter_height, filter_width, stride=1, padding=0):
    """
    Convert input data to column format for efficient convolution.

    Args:
        input_data: (batch_size, channels, height, width)
        filter_height: int
        filter_width: int
        stride: int
        padding: int

    Returns:
        Column matrix for convolution
    """
    batch_size, channels, height, width = input_data.shape

    # Pad input
    if padding > 0:
        input_data = np.pad(
            input_data,
            ((0, 0), (0, 0), (padding, padding), (padding, padding)),
            mode="constant",
        )
        height += 2 * padding
        width += 2 * padding

    # Calculate output dimensions
    out_height = (height - filter_height) // stride + 1
    out_width = (width - filter_width) // stride + 1

    # Create column matrix
    col = np.zeros(
        (batch_size * out_height * out_width, channels * filter_height * filter_width)
    )

    for y in range(filter_height):
        y_max = y + stride * out_height
        for x in range(filter_width):
            x_max = x + stride * out_width
            col[:, y * filter_width + x :: channels * filter_height * filter_width] = (
                input_data[:, :, y:y_max:stride, x:x_max:stride].reshape(
                    batch_size * out_height * out_width, -1
                )
            )

    return col


def col2im(col, input_shape, filter_height, filter_width, stride=1, padding=0):
    """
    Convert column format back to input format.

    Args:
        col: column matrix
        input_shape: original input shape (batch_size, channels, height, width)
        filter_height: int
        filter_width: int
        stride: int
        padding: int

    Returns:
        Reconstructed input data
    """
    batch_size, channels, height, width = input_shape

    if padding > 0:
        padded_height = height + 2 * padding
        padded_width = width + 2 * padding
    else:
        padded_height = height
        padded_width = width

    out_height = (padded_height - filter_height) // stride + 1
    out_width = (padded_width - filter_width) // stride + 1

    img = np.zeros((batch_size, channels

    for y in range(filter_height):
        y_max = y + stride * out_height
        for x in range(filter_width):
            x_max = x + stride * out_width
            img[:, :, y:y_max:stride, x:x_max:stride] += col[
                :, y * filter_width + x :: channels * filter_height * filter_width
            ].reshape(batch_size, channels, out_height, out_width)

    if padding > 0:
        return img[:, :, padding:-padding, padding:-padding]
    else:
        return img


class AdadeltaOptimizer:
    """
    Adadelta optimizer implementation
    """
    def __init__(self, rho=0.95, epsilon=1e-6):
        self.rho = rho
        self.epsilon = epsilon
        self.exp_sqr_grads = {}
        self.exp_sqr_updates = {}
    
    def update(self, param, grad, param_name):
        """
        Update parameter using Adadelta
        """
        if param_name not in self.exp_sqr_grads:
            self.exp_sqr_grads[param_name] = np.zeros_like(param)
            self.exp_sqr_updates[param_name] = np.zeros_like(param)
        
        exp_sqr_grad = self.exp_sqr_grads[param_name]
        exp_sqr_update = self.exp_sqr_updates[param_name]
        
        # Accumulate gradient
        exp_sqr_grad = self.rho * exp_sqr_grad + (1 - self.rho) * grad**2
        
        # Compute update
        update = -np.sqrt(exp_sqr_update + self.epsilon) / np.sqrt(exp_sqr_grad + self.epsilon) * grad
        
        # Accumulate updates
        exp_sqr_update = self.rho * exp_sqr_update + (1 - self.rho) * update**2
        
        # Update stored values
        self.exp_sqr_grads[param_name] = exp_sqr_grad
        self.exp_sqr_updates[param_name] = exp_sqr_update
        
        return param + update


class SGDOptimizer:
    """
    Simple SGD optimizer
    """
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    
    def update(self, param, grad, param_name=None):
        """
        Update parameter using SGD
        """
        return param - self.learning_rate * grad
