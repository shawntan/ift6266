import numpy as np
import theano.tensor as T
import conv_ops
import feedforward
from theano.tensor.signal.pool import pool_2d

FEATURE_MAPS = 32


def build_gated_downsample(P, i):
    conv = build_conv_layer(
        P, name="downsample_%d" % i,
        input_size=FEATURE_MAPS,
        output_size=2 * FEATURE_MAPS,
        rfield_size=3,
        weight_init=tanh_weight_init,
        activation=lambda x: x
    )

    def ds(X):
        lin = conv(X)
        gate = T.nnet.sigmoid(lin[:, :FEATURE_MAPS, :, :])
        output = (gate * T.tanh(lin[:, FEATURE_MAPS:, :, :]) +
                  (1 - gate) * X)

        return pool_2d(output, (2, 2),
                       ignore_border=True,
                       mode='max')
    return ds


def build_gated_upsample(P, i):
    conv = build_conv_layer(
        P, name="upsample_%d" % i,
        input_size=FEATURE_MAPS,
        output_size=2 * FEATURE_MAPS,
        rfield_size=5,
        weight_init=tanh_weight_init,
        activation=lambda x: x
    )

    def us(X):
        upsamp_X = T.nnet.abstract_conv.bilinear_upsampling(X, 2)
        upsamp_X = T.set_subtensor(upsamp_X[:, :, -1, :], 0)
        upsamp_X = T.set_subtensor(upsamp_X[:, :, :, -1], 0)
        lin = conv(upsamp_X)
        gate = T.nnet.sigmoid(lin[:, :FEATURE_MAPS, :, :])
        output = (gate * T.tanh(lin[:, FEATURE_MAPS:, :, :]) +
                  (1 - gate) * upsamp_X)
        return output
    return us


def build_conv_layer(P, name, input_size, output_size, rfield_size,
                     activation=T.nnet.relu,
                     weight_init=conv_ops.conv_weight_init):
    P['W_conv_%s' % name] = weight_init(output_size,
                                        input_size,
                                        rfield_size)
    P['b_conv_%s' % name] = np.zeros(output_size)
    W = P['W_conv_%s' % name]
    b = P['b_conv_%s' % name].dimshuffle('x', 0, 'x', 'x')

    def convolve(X):
        return activation(
            T.nnet.conv2d(X, W, border_mode='half') + b)
    return convolve


def tanh_weight_init(output_size, input_size, rfield_size):
    W = feedforward.initial_weights(input_size * rfield_size**2,
                                    output_size)
    W = W.reshape(output_size, input_size, rfield_size, rfield_size)
    return W


def build_gated_conv_layer(P, name, input_size, rfield_size,
                           activation=T.tanh):
    conv = build_conv_layer(P, name, input_size,
                            2 * input_size, rfield_size,
                            weight_init=tanh_weight_init,
                            activation=lambda x: x)

    def convolve(X):
        lin = conv(X)
        gate = T.nnet.sigmoid(lin[:, :input_size, :, :])
        output = (gate * activation(lin[:, input_size:, :, :]) +
                  (1 - gate) * X)
        return output
    return convolve


def build(P):
    FEATURE_STACK = 2

    norm_transform = build_conv_layer(
        P, name="normalising_transform",
        input_size=3,
        output_size=FEATURE_MAPS,
        rfield_size=3,
        weight_init=tanh_weight_init,
        activation=T.tanh
    )

    downsample = [None] * 5
    for i in xrange(FEATURE_STACK):
        downsample[i] = build_gated_downsample(P, i)

    inpaint_iterator = build_gated_conv_layer(
        P, name="inpaint_iterator",
        input_size=FEATURE_MAPS,
        rfield_size=3,
    )

    output_transform = build_conv_layer(
        P, name="output",
        input_size=FEATURE_MAPS,
        output_size=3 * 256,
        rfield_size=1
    )

    upsample = [None] * FEATURE_STACK
    for i in xrange(FEATURE_STACK):
        upsample[i] = build_gated_upsample(P, i)

    def inpaint(X, training=True, iteration_steps=8):
        batch_size, channels, img_size_1, img_size_2 = X.shape
        down_X = X / 255.
        down_X = norm_transform(down_X)
        down_X = T.set_subtensor(down_X[:, :, 16:48, 16:48], 0)
        down_X = downsample[0](down_X)
        down_X = T.set_subtensor(down_X[:, :, 8:24, 8:24], 0)
        down_X = downsample[1](down_X)
        down_X = T.set_subtensor(down_X[:, :, 4:12, 4:12], 0)
        fill_X = down_X
        # batch_size, 32, 16, 16
        fill_X = T.set_subtensor(fill_X[:, :, 4:12, 4:12], 0)
        fill_X = inpaint_iterator(fill_X)
        fill_X = T.set_subtensor(fill_X[:, :, 5:11, 5:11], 0)
        fill_X = inpaint_iterator(fill_X)
        fill_X = T.set_subtensor(fill_X[:, :, 6:10, 6:12], 0)
        fill_X = inpaint_iterator(fill_X)
        fill_X = T.set_subtensor(fill_X[:, :, 7:9, 7:9], 0)

        def fill_step(prev_fill):
            fill = inpaint_iterator(prev_fill)
            # batch_size, 32, 8, 8
            up_Y = fill[:, :, 4:12, 4:12]
            up_Y = upsample[0](up_Y)
            up_Y = upsample[1](up_Y)
            output = output_transform(up_Y)
            return fill, output

        outputs = []
        for i in xrange(iteration_steps):
            fill_X, output = fill_step(fill_X)
            outputs.append(output.dimshuffle('x', 0, 1, 2, 3))

        if training:
            return outputs[-1]
        else:
            return T.concatenate(outputs, axis=0)

    return inpaint


def cost(recon, X):
    true = X[:, :, 16:48, 16:48].dimshuffle(0, 2, 3, 1)
    batch_size, img_size_1, img_size_2, _ = true.shape
    true = true.flatten()
    recon = recon.dimshuffle(0, 2, 3, 1)
    recon = recon.reshape((batch_size * img_size_1 * img_size_2 * 3, 256))
    per_colour_loss = T.nnet.categorical_crossentropy(T.nnet.softmax(recon),
                                                      true)
    per_colour_loss = per_colour_loss.reshape((batch_size,
                                               img_size_1, img_size_2, 3))
    per_image_loss = T.sum(per_colour_loss, axis=(1, 2, 3))
    return T.mean(per_image_loss)


def predict(recon):
    new_recon_shape = T.concatenate([recon.shape[:-3], (3, 256,
                                                        recon.shape[-2],
                                                        recon.shape[-1])])
    recon = recon.reshape(new_recon_shape, ndim=recon.ndim + 1)
    return T.argmax(recon, axis=-3)
