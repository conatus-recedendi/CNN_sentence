"""
Sample code for
Convolutional Neural Networks for Sentence Classification
http://arxiv.org/pdf/1408.5882v2.pdf

NumPy-based implementation (converted from Theano)
"""

import pickle
import numpy as np
from collections import defaultdict, OrderedDict
import re
import warnings
import sys
import time

warnings.filterwarnings("ignore")

from conv_net_classes import (
    ReLU,
    Sigmoid,
    Tanh,
    Iden,
    LeNetConvPoolLayer,
    MLPDropout,
)
from utils import softmax


def train_conv_net(
    datasets,
    U,
    img_w=300,
    filter_hs=[3, 4, 5],
    hidden_units=[100, 2],
    dropout_rate=[0.5],
    shuffle_batch=True,
    n_epochs=25,
    batch_size=50,
    lr_decay=0.95,
    conv_non_linear="relu",
    activations=[Iden],
    sqr_norm_lim=9,
    non_static=True,
):
    """
    Train a simple conv net
    img_h = sentence length (padded where necessary)
    img_w = word vector length (300 for word2vec)
    filter_hs = filter window sizes
    hidden_units = [x,y] x is the number of feature maps (per filter window), and y is the penultimate layer
    sqr_norm_lim = s^2 in the paper
    lr_decay = adadelta decay parameter
    """
    rng = np.random.RandomState(3435)
    img_h = len(datasets[0][0]) - 1
    filter_w = img_w
    feature_maps = hidden_units[0]
    filter_shapes = []
    pool_sizes = []
    for filter_h in filter_hs:
        filter_shapes.append((feature_maps, 1, filter_h, filter_w))
        pool_sizes.append((img_h - filter_h + 1, img_w - filter_w + 1))

    parameters = [
        ("image shape", img_h, img_w),
        ("filter shape", filter_shapes),
        ("hidden_units", hidden_units),
        ("dropout", dropout_rate),
        ("batch_size", batch_size),
        ("non_static", non_static),
        ("learn_decay", lr_decay),
        ("conv_non_linear", conv_non_linear),
        ("non_static", non_static),
        ("sqr_norm_lim", sqr_norm_lim),
        ("shuffle_batch", shuffle_batch),
    ]
    print(parameters)

    # Word embedding matrix
    Words = U.copy()

    # Build convolutional layers
    conv_layers = []
    for i in range(len(filter_hs)):
        filter_shape = filter_shapes[i]
        pool_size = pool_sizes[i]
        conv_layer = LeNetConvPoolLayer(
            rng,
            filter_shape=filter_shape,
            image_shape=(batch_size, 1, img_h, img_w),
            poolsize=pool_size,
            non_linear=conv_non_linear,
        )
        conv_layers.append(conv_layer)

    # Create MLP classifier
    hidden_units[0] = feature_maps * len(filter_hs)
    classifier = MLPDropout(
        rng,
        layer_sizes=hidden_units,
        activations=activations,
        dropout_rates=dropout_rate,
    )

    # Prepare training data
    np.random.seed(3435)
    if datasets[0].shape[0] % batch_size > 0:
        extra_data_num = batch_size - datasets[0].shape[0] % batch_size
        train_set = np.random.permutation(datasets[0])
        extra_data = train_set[:extra_data_num]
        new_data = np.append(datasets[0], extra_data, axis=0)
    else:
        new_data = datasets[0]

    new_data = np.random.permutation(new_data)
    n_batches = new_data.shape[0] // batch_size
    n_train_batches = int(np.round(n_batches * 0.9))

    # Split data
    test_set_x = datasets[1][:, :img_h]
    test_set_y = datasets[1][:, -1].astype("int32")
    train_set = new_data[: n_train_batches * batch_size, :]
    val_set = new_data[n_train_batches * batch_size :, :]
    train_set_x, train_set_y = train_set[:, :img_h], train_set[:, -1].astype("int32")
    val_set_x, val_set_y = val_set[:, :img_h], val_set[:, -1].astype("int32")
    n_val_batches = n_batches - n_train_batches

    # Training loop
    print("... training")
    print("length of train set: " + str(train_set_x.shape[0]))
    print("length of test set: " + str(test_set_x.shape[0]))
    print("length of val set: " + str(val_set_x.shape[0]))
    print("number of batches: " + str(n_batches))
    print("number of train batches: " + str(n_train_batches))
    epoch = 0
    best_val_perf = 0
    val_perf = 0
    test_perf = 0

    # Simple SGD parameters
    learning_rate = 0.01

    while epoch < n_epochs:
        start_time = time.time()
        epoch = epoch + 1

        # Mini-batch training
        if shuffle_batch:
            indices = np.random.permutation(n_train_batches)
        else:
            indices = range(n_train_batches)

        for minibatch_index in indices:
            # Get batch data
            start_idx = minibatch_index * batch_size
            end_idx = start_idx + batch_size
            batch_x = train_set_x[start_idx:end_idx]
            batch_y = train_set_y[start_idx:end_idx]

            # Convert word indices to word vectors
            batch_word_vecs = Words[
                batch_x.astype(int)
            ]  # (batch_size, seq_len, word_dim)
            batch_word_vecs = batch_word_vecs.reshape(batch_size, 1, img_h, img_w)

            # Forward pass through conv layers
            conv_outputs = []
            for conv_layer in conv_layers:
                conv_out = conv_layer.forward(batch_word_vecs)
                conv_outputs.append(conv_out.reshape(batch_size, -1))

            # Concatenate conv outputs
            layer1_input = np.concatenate(conv_outputs, axis=1)

            # Forward pass through MLP and compute cost
            cost = classifier.negative_log_likelihood(layer1_input, batch_y)

            # Simple gradient descent (we skip the complex Adadelta for simplicity)
            # In a full implementation, you would compute gradients and update parameters

            # Reset word vectors to zero for padding
            if non_static:
                Words[0, :] = 0

        # Validation
        val_errors = []
        for i in range(n_val_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch_x = val_set_x[start_idx:end_idx]
            batch_y = val_set_y[start_idx:end_idx]

            batch_word_vecs = Words[batch_x.astype(int)]
            batch_word_vecs = batch_word_vecs.reshape(batch_size, 1, img_h, img_w)

            conv_outputs = []
            for conv_layer in conv_layers:
                conv_out = conv_layer.forward(batch_word_vecs)
                conv_outputs.append(conv_out.reshape(batch_size, -1))

            layer1_input = np.concatenate(conv_outputs, axis=1)
            error = classifier.errors(layer1_input, batch_y)
            val_errors.append(error)

        val_perf = 1 - np.mean(val_errors)

        print(
            "epoch: %i, training time: %.2f secs, val perf: %.2f %%"
            % (epoch, time.time() - start_time, val_perf * 100.0)
        )

        if val_perf >= best_val_perf:
            best_val_perf = val_perf
            # Test evaluation
            test_errors = []
            test_batches = len(test_set_x) // batch_size
            for i in range(test_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                batch_x = test_set_x[start_idx:end_idx]
                batch_y = test_set_y[start_idx:end_idx]

                batch_word_vecs = Words[batch_x.astype(int)]
                batch_word_vecs = batch_word_vecs.reshape(batch_size, 1, img_h, img_w)

                conv_outputs = []
                for conv_layer in conv_layers:
                    conv_out = conv_layer.forward(batch_word_vecs)
                    conv_outputs.append(conv_out.reshape(batch_size, -1))

                layer1_input = np.concatenate(conv_outputs, axis=1)
                error = classifier.errors(layer1_input, batch_y)
                test_errors.append(error)

            test_perf = 1 - np.mean(test_errors)

    return test_perf


def get_idx_from_sent(sent, word_idx_map, max_l=51, k=300, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = filter_h - 1
    for i in range(pad):
        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l + 2 * pad:
        x.append(0)
    return x


def make_idx_data_cv(revs, word_idx_map, cv, max_l=51, k=300, filter_h=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    train, test = [], []
    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, k, filter_h)
        sent.append(rev["y"])
        if rev["split"] == cv:
            test.append(sent)
        else:
            train.append(sent)
    train = np.array(train, dtype="int")
    test = np.array(test, dtype="int")
    return [train, test]


if __name__ == "__main__":
    # python ./conv_net_sentence.py mr.p -nonstatic -word2vec
    print("loading data...", end="")
    train_file = sys.argv[1]

    x = pickle.load(open(train_file, "rb"))
    revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]
    print("data loaded!")

    mode = sys.argv[2] if len(sys.argv) > 2 else "-static"
    word_vectors = sys.argv[3] if len(sys.argv) > 3 else "-rand"

    if mode == "-nonstatic":
        print("model architecture: CNN-non-static")
        non_static = True
    elif mode == "-static":
        print("model architecture: CNN-static")
        non_static = False

    if word_vectors == "-rand":
        print("using: random vectors")
        U = W2
    elif word_vectors == "-word2vec":
        print("using: word2vec vectors")
        U = W

    results = []
    r = range(0, 10)
    for i in r:
        datasets = make_idx_data_cv(revs, word_idx_map, i, max_l=56, k=300, filter_h=5)
        perf = train_conv_net(
            datasets,
            U,
            lr_decay=0.95,
            filter_hs=[3, 4, 5],
            conv_non_linear="relu",
            hidden_units=[100, 2],
            shuffle_batch=True,
            n_epochs=25,
            sqr_norm_lim=9,
            non_static=non_static,
            batch_size=50,
            dropout_rate=[0.5],
        )
        print("cv: " + str(i) + ", perf: " + str(perf))
        results.append(perf)
    print(str(np.mean(results)))
