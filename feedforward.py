import theano
import theano.tensor as T
import numpy as np


def initial_weights(input_size, output_size, factor=4):
    return np.asarray(
      np.random.uniform(
         low=-factor * np.sqrt(6. / (input_size + output_size)),
         high=factor * np.sqrt(6. / (input_size + output_size)),
         size=(input_size, output_size)
      ),
      dtype=theano.config.floatX
    )


def relu(X):
    return T.switch(X > 0, X, 0)


def relu_init(input_size, output_size):
    return (np.random.randn(input_size, output_size) *
            np.sqrt(2.0 / input_size)).astype(np.float32)


def build_classifier(
        P, name,
        input_sizes, hidden_sizes, output_size,
        batch_norm,
        initial_weights=initial_weights,
        activation=T.nnet.sigmoid,
        output_activation=T.nnet.softmax,
        output_weights=lambda x, y: np.zeros((x, y))):

    combine_inputs = build_combine_transform(
            P, "%s_input" % name,
            input_sizes, hidden_sizes[0],
            initial_weights=initial_weights,
            activation=activation,
            batch_norm=batch_norm
        )

    transforms = build_stacked_transforms(P, name, hidden_sizes,
                                          initial_weights=initial_weights,
                                          activation=activation)

    output = build_transform(
        P, "%s_output" % name, hidden_sizes[-1], output_size,
        initial_weights=output_weights,
        activation=output_activation,
        batch_norm=batch_norm
    )

    def classify(Xs):
        hidden_0 = combine_inputs(Xs)
        hiddens = transforms(hidden_0)
        return hiddens, output(hiddens[-1])

    return classify


def build_stacked_transforms(
        P, name, sizes,
        activation,
        initial_weights):
    if len(sizes) == 1:
        return lambda X: [X]
    else:
        transform_stack = build_stacked_transforms(
                P, name, sizes[:-1],
                initial_weights=initial_weights,
                activation=activation)
        transform = build_transform(
                P, "%s_%d" % (name, len(sizes)-1),
                sizes[-2], sizes[-1],
                initial_weights=initial_weights,
                activation=activation
            )

        def t(X):
            layers = transform_stack(X)
            return layers + [transform(layers[-1])]
        return t


def build_transform(
        P, name, input_size, output_size,
        initial_weights,
        activation):
    P["W_%s" % name] = initial_weights(input_size, output_size)
    P["b_%s" % name] = np.zeros((output_size,), dtype=np.float32)
    W = P["W_%s" % name]
    b = P["b_%s" % name]
    def transform(X):
        Z = T.dot(X, W) + b
        output = activation(Z)
        if hasattr(output, "name"):
            output.name = name
        return output
    return transform


def build_combine_transform(
        P, name, input_sizes, output_size,
        initial_weights,
        activation):
    weights = initial_weights(sum(input_sizes), output_size)

    Ws = []
    acc_size = 0
    for i, size in enumerate(input_sizes):
        P["W_%s_%d" % (name, i)] = weights[acc_size:acc_size+size]
        Ws.append(P["W_%s_%d" % (name, i)])
        acc_size += size
    P["b_%s" % name] = np.zeros((output_size,), dtype=np.float32)
    b = P["b_%s" % name]

    def transform(Xs):
        acc = 0.
        for X, W in zip(Xs, Ws):
            if X.dtype.startswith('int32'):
                acc += W[X]
            else:
                acc += T.dot(X, W)
        Z = acc + b
        output = activation(Z)
        output.name = name
        return output
    return transform
