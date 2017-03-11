import numpy as np
import theano.tensor as T
from theano.tensor.signal.pool import pool_2d

activation = T.nnet.relu


def conv_weight_init(output_size, input_size, rfield_size):
    factor = np.sqrt(2.0 / (input_size * rfield_size**2))
    return factor * np.random.randn(output_size,
                                    input_size,
                                    rfield_size,
                                    rfield_size)


def build_stack(P, name, conv_filter_counts, conv_filter_sizes,
                conv_pool_factors, activations=None):
    if activations is None:
        activations = [activation] * len(conv_filter_sizes)
    conv_layers = [None] * (len(conv_filter_counts) - 1)
    for i in xrange(len(conv_filter_counts) - 1):
        conv_layers[i] = build_conv_and_pool(
            P,
            name='%s_%d' % (name, i),
            input_size=conv_filter_counts[i],
            output_size=conv_filter_counts[i+1],
            filter_size=conv_filter_sizes[i],
            pool_factor=conv_pool_factors[i],
            activation=activations[i]
        )

    def extract(X):
        batch_size, feature_size, img_size_1, img_size_2 = X.shape
        layers = []
        prev_layer = X
        for c in conv_layers:
            prev_layer = c(prev_layer)
            layers.append(prev_layer)
        return layers
    return extract


def build_conv_and_pool(P, name, input_size, output_size,
                        filter_size, pool_factor, activation):
    P['W_conv_%s' % name] = conv_weight_init(output_size,
                                             input_size,
                                             filter_size)
    P['b_conv_%s' % name] = np.zeros(output_size)
    W = P['W_conv_%s' % name]
    b = P['b_conv_%s' % name].dimshuffle('x', 0, 'x', 'x')

    def convolve(X):
        conved = T.nnet.conv2d(X, W, border_mode='half') + b
        pooled = pool_2d(activation(conved),
                         (pool_factor, pool_factor),
                         ignore_border=True,
                         mode='average_exc_pad')
        return pooled
    return convolve
