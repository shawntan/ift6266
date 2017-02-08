import theano
import theano.tensor as T
import numpy as np
from theano_toolkit.parameters import Parameters
from theano_toolkit import updates
import data_io
import model

if __name__ == "__main__":
    chunk_size = 1024
    batch_size = 32
    P = Parameters()
    inpaint = model.build(P)

    X = T.itensor4('X')
    loss = model.cost(inpaint(T.cast(X, 'float32')), X)
    parameters = P.values()
    gradients = T.grad(loss, wrt=parameters)
    chunk_X = theano.shared(np.empty((1, 3, 64, 64), dtype=np.int32))
    idx = T.iscalar('idx')
    train = theano.function(
        inputs=[idx],
        outputs=loss,
        updates=updates.adam(parameters, gradients, learning_rate=3e-4),
        givens={X: chunk_X[idx * batch_size:(idx + 1) * batch_size]}
    )

    stream = data_io.stream_file("data/train2014.pkl.gz")
    stream = data_io.buffered_random(stream)
    stream = data_io.randomised_chunks((x[0] for x in stream),
                                       buffer_items=chunk_size)
    stream = data_io.async(stream, queue_size=100)
    for chunk in stream:
        chunk_X.set_value(chunk)
        for i in xrange(chunk_size / batch_size):
            print train(i)
        P.save('model.pkl')
