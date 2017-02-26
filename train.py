import theano
import theano.tensor as T
import numpy as np
from theano_toolkit.parameters import Parameters
from theano_toolkit import updates
import data_io
import model
import math
from pprint import pprint

if __name__ == "__main__":
    chunk_size = 5000
    batch_size = 20
    P = Parameters()
    inpaint = model.build(P)

    X = T.itensor4('X')
    loss = model.cost(inpaint(T.cast(X, 'float32')), X) / (
        32 * 32)
    parameters = P.values()
    pprint(parameters)
    gradients = updates.clip_deltas(T.grad(loss, wrt=parameters), 5)
    chunk_X = theano.shared(np.empty((1, 3, 64, 64), dtype=np.int32))
    idx = T.iscalar('idx')
    train = theano.function(
        inputs=[idx],
        outputs=loss,
        updates=updates.adam(parameters, gradients, learning_rate=1e-4),
        givens={X: chunk_X[idx * batch_size:(idx + 1) * batch_size]}
    )

    test = theano.function(inputs=[X], outputs=loss)

    def data_stream():
        stream = data_io.stream_file("data/train2014.pkl.gz")
        stream = data_io.buffered_random(stream)
        stream = data_io.chunks((x[0] for x in stream),
                                buffer_items=chunk_size)
        stream = data_io.async(stream, queue_size=100)
        return stream

    def validation():
        stream = data_io.stream_file("data/val2014.pkl.gz")
        stream = data_io.chunks((x[0] for x in stream), buffer_items=512)
        total = 0
        count = 0
        for chunk in stream:
            total += chunk.shape[0] * test(chunk)
            count += chunk.shape[0]
        return total / count

    best_cost = np.inf
    for epoch in xrange(20):
        print "Epoch %d" % epoch,
        for chunk in data_stream():
            chunk_X.set_value(chunk)
            batches = int(math.ceil(chunk.shape[0] / float(batch_size)))
            for i in xrange(batches):
                loss_val = train(i)
                print loss_val
                # pprint({p.name: g for p, g in zip(parameters, grad_norms)})
        cost = validation()
        print cost,
        if cost < best_cost:
            print "Saving...."
            P.save('model.pkl')
            best_cost = cost
        else:
            print
