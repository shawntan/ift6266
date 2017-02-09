import numpy as np
import theano.tensor as T
import conv_ops
import deconv_ops
import feedforward


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


def build_gated_conv_layer(P, name, input_size, output_size, rfield_size,
                           activation=T.tanh):
    conv = build_conv_layer(P, name, input_size,
                            2 * output_size, rfield_size,
                            weight_init=tanh_weight_init,
                            activation=lambda x: x)

    def convolve(X):
        lin = conv(X)
        gate = T.nnet.sigmoid(lin[:, :output_size, :, :])
        output = (gate * activation(lin[:, output_size:, :, :]) +
                  (1 - gate) * X)
        return output
    return convolve


def build(P):
    FEATURE_STACK = 2
    input_size = 3
    downsample = [None] * FEATURE_STACK
    for i in xrange(FEATURE_STACK):
        downsample[i] = conv_ops.build_conv_and_pool(
            P, name="downsample_%d" % i,
            input_size=input_size,
            output_size=32,
            filter_size=5,
            pool_factor=2,
            activation=T.nnet.relu
        )
        input_size = 32

    norm_transform = build_conv_layer(
        P, name="normalising_transform",
        input_size=32,
        output_size=32,
        rfield_size=3,
        weight_init=tanh_weight_init,
        activation=T.tanh
    )

    inpaint_iterator = build_gated_conv_layer(
        P, name="inpaint_iterator",
        input_size=32,
        output_size=32,
        rfield_size=5,
    )

    output_transform = build_conv_layer(
        P, name="output",
        input_size=32,
        output_size=3 * 256,
        rfield_size=1
    )

    upsample = [None] * FEATURE_STACK
    for i in xrange(FEATURE_STACK):
        upsample[i] = deconv_ops.build_upsample_and_conv(
            P, name="upsample_%d" % i,
            input_size=input_size,
            output_size=32,
            filter_size=5,
            pool_factor=2,
            activation=T.nnet.relu
        )
        input_size = 32

    def inpaint(X):
        batch_size, channels, img_size_1, img_size_2 = X.shape
        down_X = T.set_subtensor(X[:, :, 16:48, 16:48], 0)
        down_X = downsample[0](down_X)
        down_X = downsample[1](down_X)
        down_X = norm_transform(down_X)
        # batch_size, 32, 16, 16

        fill_X = T.set_subtensor(down_X[:, :, 4:12, 4:12], 0)
        fill_X = inpaint_iterator(fill_X)
        fill_X = T.set_subtensor(down_X[:, :, 5:11, 5:11], 0)
        fill_X = inpaint_iterator(fill_X)
        fill_X = T.set_subtensor(down_X[:, :, 6:10, 6:12], 0)
        fill_X = inpaint_iterator(fill_X)
        fill_X = T.set_subtensor(down_X[:, :, 7:9, 7:9], 0)
        fill_X = inpaint_iterator(fill_X)

        # batch_size, 32, 8, 8
        up_Y = fill_X[:, :, 4:12, 4:12]
        up_Y = upsample[0](up_Y)
        up_Y = upsample[1](up_Y)

        output = output_transform(up_Y)

        return output

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
    recon = recon.reshape((recon.shape[0],
                           3, 256,
                           recon.shape[2],
                           recon.shape[3]))
    return T.argmax(recon, axis=2)
