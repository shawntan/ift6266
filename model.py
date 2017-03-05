import numpy as np
import theano.tensor as T
import conv_ops
import feedforward
from theano.tensor.signal.pool import pool_2d
FMAP_SIZES = [32, 32, 64, 128, 256, 256]
FEATURE_MAP_SIZE = FMAP_SIZES[0]
REV_FMAP_SIZES = FMAP_SIZES[::-1]


def build_downsample(P, i,
                     input_feature_map,
                     output_feature_map):
    conv = build_conv_layer(
        P, name="downsample_%d" % i,
        input_size=input_feature_map,
        output_size=output_feature_map,
        rfield_size=3,
        weight_init=conv_ops.conv_weight_init,
        activation=T.nnet.relu,
        gated_bias_init=False
    )

    def ds(X):
        return pool_2d(conv(X), (2, 2),
                       ignore_border=True,
                       mode='average_exc_pad')
    return ds


def build_upsample(P, i,
                   input_feature_map,
                   output_feature_map):
    conv = build_conv_layer(
        P, name="upsample_%d" % i,
        input_size=input_feature_map,
        output_size=output_feature_map,
        rfield_size=3,
        weight_init=conv_ops.conv_weight_init,
        activation=lambda x: x,
        gated_bias_init=False
    )

    def us(X):
        upsamp_X = T.nnet.abstract_conv.bilinear_upsampling(X, 2)
        return conv(upsamp_X)
    return us


def build_conv_layer(P, name, input_size, output_size, rfield_size,
                     activation=T.nnet.relu,
                     weight_init=conv_ops.conv_weight_init,
                     gated_bias_init=False):
    P['W_conv_%s' % name] = weight_init(output_size,
                                        input_size,
                                        rfield_size)
    b_ = np.zeros(output_size)
    if gated_bias_init:
        b_[output_size:] = 5
    P['b_conv_%s' % name] = b_
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
                            activation=lambda x: x,
                            gated_bias_init=True)

    def convolve(X):
        lin = conv(X)
        gate = T.nnet.sigmoid(lin[:, :input_size, :, :])
        output = ((1 - gate) * activation(lin[:, input_size:, :, :]) +
                  gate * X)
        return output
    return convolve


def build(P):

    input_transform = build_conv_layer(
        P, name="input_transform",
        input_size=3,
        output_size=FMAP_SIZES[0],
        rfield_size=3,
        weight_init=tanh_weight_init,
        activation=T.tanh
    )

    pre_output_transform = build_conv_layer(
        P, name="output_transform",
        input_size=FMAP_SIZES[0],
        output_size=FMAP_SIZES[0],
        rfield_size=3,
        weight_init=tanh_weight_init,
        activation=T.tanh
    )

    downsample = [None] * (len(FMAP_SIZES) - 1)
    fmap_size = 64
    for i in xrange(len(FMAP_SIZES) - 1):
        fmap_size = fmap_size // 2
        print i, FMAP_SIZES[i], FMAP_SIZES[i+1], fmap_size
        downsample[i] = build_downsample(
            P, i,
            input_feature_map=FMAP_SIZES[i],
            output_feature_map=FMAP_SIZES[i+1]
        )
    downsample_fmap_size = fmap_size
    print
    upsample = [None] * (len(FMAP_SIZES) - 1)
    fmap_size = 1
    for i in xrange(len(FMAP_SIZES) - 1):
        fmap_size = fmap_size * 2
        print i, REV_FMAP_SIZES[i], REV_FMAP_SIZES[i+1], fmap_size
        upsample[i] = build_upsample(
            P, i,
            input_feature_map=REV_FMAP_SIZES[i],
            output_feature_map=REV_FMAP_SIZES[i+1]
        )

    P.W_dense_in = feedforward.relu_init(
            FMAP_SIZES[-1] * downsample_fmap_size**2,
            512)
    P.b_dense_in = np.zeros(512)
    P.W_dense_out = feedforward.relu_init(
        512, REV_FMAP_SIZES[0])
    P.b_dense_out = np.zeros(REV_FMAP_SIZES[0])

    inpaint_iterator = build_gated_conv_layer(
        P, name="inpaint_iterator",
        input_size=32,
        rfield_size=3,
    )

    output_transform = build_conv_layer(
        P, name="output",
        input_size=FMAP_SIZES[0],
        output_size=3 * 256,
        rfield_size=1
    )

    def inpaint(X, training=True, iteration_steps=16):
        batch_size, channels, img_size_1, img_size_2 = X.shape
        down_X = X / 255.
        down_X = T.set_subtensor(down_X[:, :, 16:48, 16:48], 0)
        down_X = input_transform(down_X)
        fill_X = down_X
        down_X = downsample[0](down_X)
        down_X = T.set_subtensor(down_X[:, :, 8:24, 8:24], 0)
        down_X = downsample[1](down_X)
        down_X = T.set_subtensor(down_X[:, :, 4:12, 4:12], 0)
        down_X = downsample[2](down_X)
        down_X = T.set_subtensor(down_X[:, :, 2:6, 2:6], 0)
        down_X = downsample[3](down_X)
        down_X = T.set_subtensor(down_X[:, :, 1:3, 1:3], 0)
        down_X = downsample[4](down_X)

        z = down_X.flatten(2)
        z = T.nnet.relu(T.dot(z, P.W_dense_in) + P.b_dense_in)
        z = T.nnet.relu(T.dot(z, P.W_dense_out) + P.b_dense_out)

        up_Y = z.reshape((z.shape[0], REV_FMAP_SIZES[0], 1, 1))
        up_Y = upsample[0](up_Y)
        up_Y = upsample[1](up_Y)
        up_Y = upsample[2](up_Y)
        up_Y = upsample[3](up_Y)
        up_Y = upsample[4](up_Y)

        up_Y = pre_output_transform(up_Y)
        # batch_size, 32, 16, 16
        fill_X = T.set_subtensor(fill_X[:, :, 16:48, 16:48], up_Y)

        def fill_step(prev_fill):
            fill = inpaint_iterator(prev_fill)
            # batch_size, 32, 8, 8
            output = output_transform(fill)[:, :, 16:48, 16:48]
            return fill, output

        outputs = []
        for i in xrange(iteration_steps):
            fill_X, output = fill_step(fill_X)
            outputs.append(output.dimshuffle('x', 0, 1, 2, 3))
        if training:
            return T.concatenate(outputs, axis=0)
        else:
            return outputs[-1]
    return inpaint


def cost(recon, X, validation=False):
    true = X[:, :, 16:48, 16:48].dimshuffle(0, 2, 3, 1)
    iteration_steps, batch_size, channels, img_size_1, img_size_2 = recon.shape
    true = T.extra_ops.repeat(
        true[None, :, :, :, :],
        repeats=iteration_steps,
        axis=0
    )
    true = true.reshape((iteration_steps *
                         batch_size *
                         img_size_1 * img_size_2 * 3,))
    recon = recon.dimshuffle(0, 1, 3, 4, 2)
    recon = recon.reshape((iteration_steps *
                           batch_size *
                           img_size_1 * img_size_2 * 3, channels // 3))
    per_colour_loss = T.nnet.categorical_crossentropy(
        T.nnet.softmax(recon), true
    )
    per_colour_loss = per_colour_loss.reshape((iteration_steps, batch_size,
                                               img_size_1, img_size_2, 3))
    per_pixel_loss = T.sum(per_colour_loss, axis=-1)
    per_pixel_loss = T.max(per_pixel_loss, axis=0)

    per_image_loss = T.sum(per_pixel_loss, axis=(1, 2))
    return T.mean(per_image_loss, axis=0)


def predict(recon):
    iteration_steps, batch_size, channels, img_size_1, img_size_2 = recon.shape
    new_recon_shape = (iteration_steps,
                       batch_size,
                       3, 256,
                       img_size_1,
                       img_size_2)
    return T.argmax(recon.reshape(new_recon_shape), axis=-3)
