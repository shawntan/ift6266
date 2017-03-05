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
    batch_size = 32
    P = Parameters()
    inpaint = model.build(P)

    parameters = P.values()
    X = T.itensor4('X')
    loss = (model.cost(inpaint(T.cast(X, 'float32')), X) +
            1e-4 * sum(T.sum(T.sqr(w)) for w in parameters)) / (32 * 32)
    val_loss = model.cost(
        inpaint(T.cast(X, 'float32')), X
    ) / (32 * 32)
    display_loss = val_loss
    pprint(parameters)
    gradients = updates.clip_deltas(T.grad(loss, wrt=parameters), 5)
    chunk_X = theano.shared(np.empty((1, 3, 64, 64), dtype=np.int32))
    idx = T.iscalar('idx')
    # [T.sum(T.sqr(w)) for w in gradients],
    train = theano.function(
        inputs=[idx],
        outputs=display_loss,
        updates=updates.adam(parameters, gradients, learning_rate=1e-4),
        givens={X: chunk_X[idx * batch_size:(idx + 1) * batch_size]}
    )

    test = theano.function(inputs=[X], outputs=val_loss)
    print "Done compilation."

    def data_stream():
        stream = data_io.stream_file("data/train2014.pkl.gz")
        stream = data_io.buffered_random(stream)
        stream = data_io.chunks((x[0] for x in stream),
                                buffer_items=chunk_size)
        stream = data_io.async(stream, queue_size=2)
        return stream

    def validation():
        stream = data_io.stream_file("data/val2014.pkl.gz")
        stream = data_io.chunks((x[0] for x in stream), buffer_items=128)
        stream = data_io.async(stream, queue_size=3)
        total = 0
        count = 0
        for chunk in stream:
            total += chunk.shape[0] * test(chunk)
            count += chunk.shape[0]
        return total / count

    best_cost = np.inf
    for epoch in xrange(20):
        print "Epoch %d" % epoch,
        cost = validation()
        print cost,
        if cost < best_cost:
            print "Saving...."
            P.save('model.pkl')
            best_cost = cost
        else:
            print
        for chunk in data_stream():
            chunk_X.set_value(chunk)
            batches = int(math.ceil(chunk.shape[0] / float(batch_size)))
            for i in xrange(batches):
                loss = train(i)
                print loss
                # pprint({p.name: g for p, g in zip(parameters, grad_norms)})
