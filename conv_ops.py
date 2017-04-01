import numpy as np
import theano.tensor as T
from theano.tensor.signal.pool import pool_2d

activation = T.nnet.relu


def conv_weight_init(output_size, input_size, rfield_size):
    # factor = 4 * np.sqrt(
    #     6. / (input_size * rfield_size**2 + output_size))
    # return np.asarray(
    #     np.random.uniform(
    #        low=-factor,
    #        high=factor,
    #        size=(output_size, input_size, rfield_size, rfield_size)
    #     ),
    #     dtype=theano.config.floatX
    # )

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
            rfield_size=conv_filter_sizes[i],
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


def build_conv_layer(P, name, input_size, output_size, rfield_size,
                     activation=activation,
                     weight_init=conv_weight_init,
                     batch_norm=True):
    P['W_conv_%s' % name] = weight_init(output_size,
                                        input_size,
                                        rfield_size)
    P['b_conv_%s' % name] = np.zeros(output_size)
    P['g_conv_%s' % name] = np.ones(output_size)
    W = P['W_conv_%s' % name]
    b = P['b_conv_%s' % name].dimshuffle('x', 0, 'x', 'x')
    if batch_norm:
        g = P['g_conv_%s' % name].dimshuffle('x', 0, 'x', 'x')

    def convolve(X, same=True):
        border_mode = 'half' if same else 'valid'
        lin_out = T.nnet.conv2d(X, W, border_mode=border_mode)
        if batch_norm:
            norm_out = T.nnet.bn.batch_normalization(
                inputs=lin_out,
                gamma=g,
                beta=b,
                mean=T.mean(lin_out, axis=(0, 2, 3), keepdims=True),
                std=T.std(lin_out, axis=(0, 2, 3), keepdims=True),
                mode='low_mem'
            )
        else:
            norm_out = lin_out + b
        return activation(norm_out)
    return convolve


def build_conv_and_pool(P, name, input_size, output_size,
                        rfield_size, pool_factor, activation):
    conv = build_conv_layer(P, name,
                            input_size, output_size, rfield_size,
                            activation)

    def convolve(X):
        conved = conv(X)
        pooled = pool_2d(conved,
                         (pool_factor, pool_factor),
                         ignore_border=True,
                         mode='max')
        return pooled
    return convolve
