"""
Sample code for
Convolutional Neural Networks for Sentence Classification
http://arxiv.org/pdf/1408.5882v2.pdf

NumPy-based implementation (converted from Theano)
"""

import numpy as np
from utils import (
    relu,
    sigmoid,
    tanh,
    identity,
    softmax,
    conv2d,
    max_pool2d,
    apply_dropout,
    cross_entropy_loss,
    accuracy,
)


def ReLU(x):
    return relu(x)


def Sigmoid(x):
    return sigmoid(x)


def Tanh(x):
    return tanh(x)


def Iden(x):
    return identity(x)


class HiddenLayer(object):
    """
    Class for HiddenLayer
    """

    def __init__(self, rng, n_in, n_out, activation, W=None, b=None, use_bias=True):
        self.activation = activation
        self.use_bias = use_bias

        if W is None:
            if hasattr(activation, '__name__') and activation.__name__ == "ReLU":
                W_values = np.asarray(
                    0.01 * rng.standard_normal(size=(n_in, n_out)),
                    dtype=np.float32,
                )
            else:
                W_values = np.asarray(
                    rng.uniform(
                        low=-np.sqrt(6.0 / (n_in + n_out)),
                        high=np.sqrt(6.0 / (n_in + n_out)),
                        size=(n_in, n_out),
                    ),
                    dtype=np.float32,
                )
            self.W = W_values
        else:
            self.W = W

        if b is None and use_bias:
            b_values = np.zeros((n_out,), dtype=np.float32)
            self.b = b_values
        else:
            self.b = b

    def forward(self, input_data):
        """Forward pass through the hidden layer"""
        if self.use_bias:
            lin_output = np.dot(input_data, self.W) + self.b
        else:
            lin_output = np.dot(input_data, self.W)

        if self.activation is None:
            return lin_output
        else:
            return self.activation(lin_output)


class DropoutHiddenLayer(HiddenLayer):
    def __init__(
        self,
        rng,
        n_in,
        n_out,
        activation,
        dropout_rate,
        use_bias=True,
        W=None,
        b=None,
    ):
        super(DropoutHiddenLayer, self).__init__(
            rng=rng,
            n_in=n_in,
            n_out=n_out,
            W=W,
            b=b,
            activation=activation,
            use_bias=use_bias,
        )
        self.dropout_rate = dropout_rate
        self.rng = rng

    def forward(self, input_data, training=True):
        """Forward pass with dropout"""
        output = super().forward(input_data)
        if training:
            output = apply_dropout(output, self.dropout_rate, training, self.rng)
        return output


class MLPDropout(object):
    """A multilayer perceptron with dropout"""

    def __init__(self, rng, layer_sizes, dropout_rates, activations, use_bias=True):
        self.rng = rng
        self.layer_sizes = layer_sizes
        self.dropout_rates = dropout_rates
        self.activations = activations
        self.use_bias = use_bias

        # Set up all the hidden layers
        self.weight_matrix_sizes = list(zip(layer_sizes, layer_sizes[1:]))
        self.dropout_layers = []
        self.layers = []

        # Build dropout layers
        layer_counter = 0
        for n_in, n_out in self.weight_matrix_sizes[:-1]:
            dropout_layer = DropoutHiddenLayer(
                rng=rng,
                n_in=n_in,
                n_out=n_out,
                activation=activations[layer_counter],
                dropout_rate=dropout_rates[layer_counter],
                use_bias=use_bias,
            )
            self.dropout_layers.append(dropout_layer)

            # Regular layer (for inference) with scaled weights
            regular_layer = HiddenLayer(
                rng=rng,
                n_in=n_in,
                n_out=n_out,
                activation=activations[layer_counter],
                W=dropout_layer.W * (1 - dropout_rates[layer_counter]),
                b=dropout_layer.b,
                use_bias=use_bias,
            )
            self.layers.append(regular_layer)
            layer_counter += 1

        # Set up the output layer (LogisticRegression)
        n_in, n_out = self.weight_matrix_sizes[-1]
        dropout_output_layer = LogisticRegression(n_in=n_in, n_out=n_out)
        self.dropout_layers.append(dropout_output_layer)

        # Regular output layer with scaled weights
        output_layer = LogisticRegression(
            n_in=n_in,
            n_out=n_out,
            W=dropout_output_layer.W * (1 - dropout_rates[-1]),
            b=dropout_output_layer.b,
        )
        self.layers.append(output_layer)

    def forward(self, input_data, training=True):
        """Forward pass through the network"""
        if training:
            layers = self.dropout_layers
        else:
            layers = self.layers

        next_layer_input = input_data

        # Apply input dropout
        if training:
            next_layer_input = apply_dropout(
                next_layer_input, self.dropout_rates[0], training, self.rng
            )

        # Forward through hidden layers
        for i, layer in enumerate(layers[:-1]):
            if training and hasattr(layer, "forward"):
                if hasattr(layer, 'dropout_rate'):
                    next_layer_input = layer.forward(next_layer_input, training=True)
                else:
                    next_layer_input = layer.forward(next_layer_input)
            else:
                next_layer_input = layer.forward(next_layer_input)

        # Output layer
        output = layers[-1].forward(next_layer_input)
        return output

    def negative_log_likelihood(self, input_data, y):
        """Compute negative log likelihood for training"""
        output = self.forward(input_data, training=True)
        return cross_entropy_loss(output, y)

    def dropout_negative_log_likelihood(self, input_data, y):
        """Alias for negative_log_likelihood for compatibility"""
        return self.negative_log_likelihood(input_data, y)

    def errors(self, input_data, y):
        """Compute classification errors for validation/test"""
        output = self.forward(input_data, training=False)
        return 1.0 - accuracy(output, y)

    def predict(self, input_data):
        """Make predictions"""
        output = self.forward(input_data, training=False)
        return np.argmax(output, axis=1)

    def predict_proba(self, input_data):
        """Predict class probabilities"""
        return self.forward(input_data, training=False)


class LogisticRegression(object):
    """Multi-class Logistic Regression Class"""

    def __init__(self, n_in, n_out, W=None, b=None):
        """Initialize the parameters of the logistic regression"""

        # Initialize weights
        if W is None:
            self.W = np.zeros((n_in, n_out), dtype=np.float32)
        else:
            self.W = W

        # Initialize biases
        if b is None:
            self.b = np.zeros((n_out,), dtype=np.float32)
        else:
            self.b = b

    def forward(self, input_data):
        """Forward pass through logistic regression"""
        linear_output = np.dot(input_data, self.W) + self.b
        return softmax(linear_output)

    def negative_log_likelihood(self, input_data, y):
        """Compute negative log likelihood"""
        p_y_given_x = self.forward(input_data)
        return cross_entropy_loss(p_y_given_x, y)

    def errors(self, input_data, y):
        """Compute classification errors"""
        p_y_given_x = self.forward(input_data)
        return 1.0 - accuracy(p_y_given_x, y)

    def predict(self, input_data):
        """Make predictions"""
        p_y_given_x = self.forward(input_data)
        return np.argmax(p_y_given_x, axis=1)


class LeNetConvPoolLayer(object):
    """Convolutional and pooling layer"""

    def __init__(
        self, rng, filter_shape, image_shape, poolsize=(2, 2), non_linear="tanh"
    ):
        """
        Initialize a convolutional layer.

        Args:
            rng: random number generator
            filter_shape: (number of filters, num input feature maps, filter height, filter width)
            image_shape: (batch size, num input feature maps, image height, image width)
            poolsize: (pool_height, pool_width)
            non_linear: activation function name
        """
        assert image_shape[1] == filter_shape[1]

        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.non_linear = non_linear

        # Calculate fan in and fan out
        fan_in = np.prod(filter_shape[1:])
        fan_out = filter_shape[0] * np.prod(filter_shape[2:]) / np.prod(poolsize)

        # Initialize weights
        if self.non_linear == "none" or self.non_linear == "relu":
            self.W = np.asarray(
                rng.uniform(low=-0.01, high=0.01, size=filter_shape),
                dtype=np.float32,
            )
        else:
            W_bound = np.sqrt(6.0 / (fan_in + fan_out))
            self.W = np.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=np.float32,
            )

        # Initialize biases
        self.b = np.zeros((filter_shape[0],), dtype=np.float32)

    def forward(self, input_data):
        """Forward pass through conv and pool layer"""
        # Convolution
        conv_out = conv2d(input_data, self.W, self.b)

        # Apply activation
        if self.non_linear == "tanh":
            conv_out = tanh(conv_out)
        elif self.non_linear == "relu":
            conv_out = relu(conv_out)
        # else: no activation (linear)

        # Max pooling
        output = max_pool2d(conv_out, self.poolsize)
        return output

    def predict(self, input_data, batch_size=None):
        """Forward pass for prediction (same as forward)"""
        return self.forward(input_data)
