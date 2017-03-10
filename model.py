import numpy as np
import theano.tensor as T
import conv_ops
import deconv_ops
import feedforward
from theano.tensor.signal.pool import pool_2d
from theano_toolkit import utils as U


FMAP_SIZES = [8, 16, 32, 64, 128, 256]
FEATURE_MAP_SIZE = FMAP_SIZES[0]
REV_FMAP_SIZES = FMAP_SIZES[::-1]
FINAL_FMAP_SIZE = 64 // 2**(len(FMAP_SIZES)-1)


def build_conv_layer(P, name, input_size, output_size, rfield_size,
                     activation=T.nnet.relu,
                     weight_init=conv_ops.conv_weight_init):
    P['W_conv_%s' % name] = weight_init(output_size,
                                        input_size,
                                        rfield_size)
    b_ = np.zeros(output_size)
    P['b_conv_%s' % name] = b_
    W = P['W_conv_%s' % name]
    b = P['b_conv_%s' % name].dimshuffle('x', 0, 'x', 'x')

    def convolve(X, same=True):
        border_mode = 'half' if same else 'valid'
        return activation(
            T.nnet.conv2d(X, W, border_mode=border_mode) + b)
    return convolve


def build_conv_gaussian_output(P, name, input_size, output_size):
    P["W_%s_mean" % name] = np.zeros((input_size, output_size))
    P["b_%s_mean" % name] = np.zeros((output_size,))
    P["W_%s_std" % name] = np.zeros((input_size, output_size))
    P["b_%s_std" % name] = np.zeros((output_size,)) + np.log(np.exp(1) - 1)
    W_mean = P["W_%s_mean" % name].dimshuffle(1, 0, 'x', 'x')
    b_mean = P["b_%s_mean" % name].dimshuffle('x', 0, 'x', 'x')
    W_std = P["W_%s_std" % name].dimshuffle(1, 0, 'x', 'x')
    b_std = P["b_%s_std" % name].dimshuffle('x', 0, 'x', 'x')

    def gaussian_params(X):
        mean = T.nnet.conv2d(X, W_mean, border_mode='half') + b_mean
        std = T.nnet.softplus(T.nnet.conv2d(X, W_std, border_mode='half') +
                              b_std)
        eps = U.theano_rng.normal(size=std.shape)
        latent = mean + eps * std
        return latent, mean, std
    return gaussian_params


def build_layer_posteriors(P, layer_sizes, output_sizes):
    gaussian_transforms = [None] * len(layer_sizes)
    for i in xrange(len(layer_sizes)):
        gaussian_transforms[i] = build_conv_gaussian_output(
            P, name="posterior_%d" % i,
            input_size=layer_sizes[i],
            output_size=output_sizes[i]
        )

    def infer(hiddens):
        return [gaussian(h)
                for h, gaussian in zip(hiddens, gaussian_transforms)]


    return infer, gaussian_transforms[-1]


def build_layer_priors(P, layer_sizes, latent_sizes):
    layers = [None] * (len(layer_sizes) - 1)
    gaussian_transforms = [None] * (len(layer_sizes) - 1)
    for i in xrange(len(layer_sizes) - 1):
        print layer_sizes[i], layer_sizes[i+1]
        layers[i] = deconv_ops.build_upsample_and_conv(
            P, name='upsample_%d' % i,
            input_size=layer_sizes[i] + latent_sizes[i],
            output_size=layer_sizes[i+1],
            filter_size=3,
            pool_factor=2
        )

    for i in xrange(len(layers)):
        gaussian_transforms[i] = build_conv_gaussian_output(
            P, name="prior_%d" % i,
            input_size=layer_sizes[i+1],
            output_size=layer_sizes[i+1]
        )

    def infer(hiddens, prev_latents):
        results = []
        for i in xrange(len(layers)):
            upsamp = layers[i](
                T.concatenate([hiddens[i], prev_latents[i]], axis=1)
            )
            results.append(gaussian_transforms[i](upsamp))
        return results

    def ancestral_sample(hiddens, top_latent):
        curr_latent = top_latent
        for i in xrange(len(layers)):
            curr_latent = layers[i](
                T.concatenate([hiddens[i], curr_latent], axis=1)
            )
        return curr_latent

    return infer, ancestral_sample


def build(P):
    input_transform = build_conv_layer(
        P, name="input",
        input_size=4,
        output_size=FMAP_SIZES[0],
        rfield_size=1
    )

    stack = conv_ops.build_stack(
        P, name="conv",
        conv_filter_counts=FMAP_SIZES,
        conv_filter_sizes=[3] * (len(FMAP_SIZES) - 1),
        conv_pool_factors=[2] * (len(FMAP_SIZES) - 1)
    )

    dense_layer = feedforward.build_transform(
        P, name="dense",
        input_size=FINAL_FMAP_SIZE**2 * FMAP_SIZES[-1],
        output_size=512,
        initial_weights=feedforward.relu_init,
        activation=T.nnet.relu
    )

    def extract(X):
        first_layer = input_transform(X)
        hiddens = stack(first_layer)
        final = dense_layer(hiddens[-1].flatten(2))
        return [first_layer] + hiddens + [final[:, :, None, None]]

    posterior_transforms, final_gaussian_transform =\
        build_layer_posteriors(P, FMAP_SIZES + [512], FMAP_SIZES + [256])

    prior_transforms, ancestral_sample = build_layer_priors(
        P,
        [512] + REV_FMAP_SIZES,
        [256] + REV_FMAP_SIZES
    )

    output_transform = build_conv_layer(
        P, name="output",
        input_size=FMAP_SIZES[0],
        output_size=3 * 256,
        rfield_size=3,
        activation=lambda x: x
    )

    def autoencoder(X):
        X = T.concatenate([X, T.ones_like(X[:, :1, :, :])], axis=1)
        X_masked = T.set_subtensor(X[:, :, 16:-16, 16:-16], 0)
        hiddens = extract(X)
        hiddens_masked = extract(X_masked)
        posteriors = posterior_transforms(hiddens)
        posterior_samples = [p[0] for p in posteriors]
        reverse_hiddens = hiddens_masked[::-1]
        reverse_posteriors = posterior_samples[::-1]
        priors = prior_transforms(reverse_hiddens[:-1],
                                  reverse_posteriors[:-1])
        priors = [final_gaussian_transform(hiddens_masked[-1])] + priors
        lin_output = output_transform(posterior_samples[0])

        return (lin_output,
                [(p[1].flatten(2), p[2].flatten(2)) for p in posteriors[::-1]],
                [(p[1].flatten(2), p[2].flatten(2)) for p in priors])

    def inpaint(X):
        X = T.concatenate([X, T.ones_like(X[:, :1, :, :])], axis=1)
        X_masked = T.set_subtensor(X[:, :, 16:-16, 16:-16], 0)
        hiddens_masked = extract(X_masked)
        reverse_hiddens = hiddens_masked[::-1]
        final_latent = final_gaussian_transform(hiddens_masked[-1])[0]
        lowest_latent = ancestral_sample(reverse_hiddens[:-1], final_latent)
        lin_output = output_transform(lowest_latent)
        return lin_output

    return autoencoder, inpaint


def cost(recon, X, validation=False):
    batch_size, channels, img_size_1, img_size_2 = recon.shape

    true = X.dimshuffle(0, 2, 3, 1)
    true = true.reshape((batch_size * img_size_1 * img_size_2 * 3,))

    recon = recon.dimshuffle(0, 2, 3, 1)
    recon = recon.reshape((batch_size * img_size_1 * img_size_2 * 3, channels // 3))
    per_colour_loss = T.nnet.categorical_crossentropy(
        T.nnet.softmax(recon), true
    )
    per_colour_loss = per_colour_loss.reshape((batch_size, img_size_1, img_size_2, 3))
    per_pixel_loss = T.sum(per_colour_loss, axis=-1)
    per_image_loss = T.sum(per_pixel_loss, axis=(1, 2))
    missing_part_loss = T.sum(per_pixel_loss[:, 16:-16, 16:-16], axis=(1, 2))
    return T.mean(per_image_loss, axis=0), T.mean(missing_part_loss, axis=0)


def predict(recon):
    batch_size, channels, img_size_1, img_size_2 = recon.shape
    new_recon_shape = (batch_size,
                       3, 256,
                       img_size_1,
                       img_size_2)
    return T.argmax(recon.reshape(new_recon_shape), axis=2)
