import numpy as np
import theano.tensor as T
from conv_ops import activation, conv_weight_init, build_conv_layer


def build_stack(P, name,
                conv_filter_counts, conv_filter_sizes,
                conv_pool_factors, activations=None):

    if activations is None:
        activations = [activation] * len(conv_filter_sizes)

    conv_layers = [None] * (len(conv_filter_counts) - 1)

    for i in xrange(len(conv_filter_counts) - 1):
        conv_layers[i] = build_upsample_and_conv(
            P, name='%s_stack_%d' % (name, i),
            input_size=conv_filter_counts[i],
            output_size=conv_filter_counts[i+1],
            rfield_size=conv_filter_sizes[i],
            pool_factor=conv_pool_factors[i],
            activation=activations[i]
        )

    def extract(X):
        batch_size, feature_size, img_size_1, img_size_2 = X.shape
        prev_layer = X
        for c in conv_layers:
            prev_layer = c(prev_layer)
        return prev_layer
    return extract


def build_upsample_and_conv(P, name, input_size, output_size,
                            rfield_size, pool_factor,
                            activation=activation):
    conv = build_conv_layer(
        P, name,
        input_size, output_size, rfield_size,
        activation
    )

    def upsample(X):
        upsamp_X = T.nnet.abstract_conv.bilinear_upsampling(X,
                                                            pool_factor)
        Y = conv(upsamp_X)
        return Y
    return upsample

if __name__ == "__main__":
    from theano_toolkit.parameters import Parameters
    P = Parameters()
    image_size = 10
    rfield_size = 5
    input_size = 1
    X = T.as_tensor_variable(
        np.random.randn(1, input_size, 4, 4).astype(np.float32))
    P.W = conv_weight_init(2, 1, 5)
    P.b = np.zeros(2)
    W = P.W
    b = P.b.dimshuffle('x', 0, 'x', 'x')

    upsamp_X_1 = T.zeros((X.shape[0],
                          X.shape[1],
                          2 * X.shape[2],
                          2 * X.shape[3]))
    upsamp_X_1 = T.set_subtensor(upsamp_X_1[:, :, ::2, ::2], X)
    upsamp_X_1 = T.inc_subtensor(upsamp_X_1[:, :, 1:-1:2, ::2],
                                 0.5 * (X[:, :, :-1] + X[:, :, 1:]))
    upsamp_X_1 = T.inc_subtensor(upsamp_X_1[:, :, ::2, 1:-1:2],
                                 0.5 * (X[:, :, :, :-1] + X[:, :, :, 1:]))
    upsamp_X_1 = T.inc_subtensor(upsamp_X_1[:, :, 1:-1:2, 1:-1:2],
                                 0.25 * (X[:, :, :-1, :-1] +
                                         X[:, :, 1:, :-1] +
                                         X[:, :, :-1, 1:] +
                                         X[:, :, 1:, 1:]))

    Y_1 = activation(T.nnet.conv2d(upsamp_X_1, W, border_mode='half') + b)

    upsamp_X_2 = T.nnet.abstract_conv.bilinear_upsampling(X, 2)
    upsamp_X_2 = T.set_subtensor(upsamp_X_2[:, :, -1, :], 0)
    upsamp_X_2 = T.set_subtensor(upsamp_X_2[:, :, :, -1], 0)
    Y_2 = activation(T.nnet.conv2d(upsamp_X_2, W, border_mode='half') + b)

    print X.eval()
    print upsamp_X_1.eval()
    print upsamp_X_2.eval()
    delta = (upsamp_X_1 - upsamp_X_2).eval()
    close = np.isclose(delta, 0)
    print delta[~close]

    cost_1 = T.sum(Y_1**2)
    cost_2 = T.sum(Y_2**2)
    grad_1 = T.grad(cost_1, wrt=W)
    grad_2 = T.grad(cost_2, wrt=W)

    grad_1_val = grad_1.eval()
    grad_2_val = grad_2.eval()
    close = np.isclose(grad_1_val, grad_2_val)

    print (grad_1_val - grad_2_val)[~close]
