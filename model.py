import numpy as np
import theano.tensor as T
import conv_ops
import deconv_ops
import feedforward


def build(P):
    conv_stack = conv_ops.build_stack(
        P,
        conv_filter_counts=[3, 32, 32, 64, 64],
        conv_filter_sizes=[3, 3, 3, 3],
        conv_pool_factors=[2, 2, 2, 2]
    )

    transconv_stack = deconv_ops.build_stack(
        P, name="upsample",
        conv_filter_counts=[64, 64, 32, 32, 3 * 256],
        conv_filter_sizes=[5, 5, 5, 5],
        conv_pool_factors=[2, 2, 2, 2],
        activations=3 * [T.nnet.relu] + [lambda x: x]
    )

    mask = T.as_tensor_variable(
        np.array([[1, 1, 1, 1],
                  [1, 0, 0, 1],
                  [1, 0, 0, 1],
                  [1, 1, 1, 1]], dtype=np.int8)
        )

    transform_1 = feedforward.build_combine_transform(
        P, "fully_connected_1",
        input_sizes=[16 * 64],
        output_size=4 * 64,
        initial_weights=feedforward.relu_init,
        activation=T.nnet.relu,
        batch_norm=False)

    def inpaint(X):
        batch_size, channels, img_size_1, img_size_2 = X.shape
        X = T.set_subtensor(X[:, :, 16:48, 16:48], 0)
        in_feature_maps = T.switch(mask[None, None, :, :],
                                   conv_stack(X), 0)
        out_feature_maps = transform_1([in_feature_maps.flatten(2)])
        out_feature_maps = out_feature_maps.reshape((batch_size, 64, 2, 2))
        output = transconv_stack(out_feature_maps)
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
