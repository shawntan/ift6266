import theano
import theano.tensor as T
import numpy as np
from theano_toolkit.parameters import Parameters
from theano_toolkit import updates
import data_io
import model
import math
from pprint import pprint
import vae

if __name__ == "__main__":
    chunk_size = 512
    batch_size = 64
    P = Parameters()
    autoencoder, _ = model.build(P)

    parameters = P.values()
    X = T.itensor4('X')
    X_hat, posteriors, priors = autoencoder(T.cast(X, 'float32') / 255.)
    latent_kls = [T.mean(vae.kl_divergence(po_m, po_s, pr_m, pr_s), axis=0)
                  for (po_m, po_s), (pr_m, pr_s) in zip(posteriors, priors)]
    latent_kl = sum(latent_kls)
    recon_loss = model.cost(
        X_hat,
        X[:, :, 16:-16, 16:-16]
    )
    pprint(parameters)
    l2 = 1e-4 * sum(T.sum(T.sqr(w)) for w in parameters
                    if w.name.startswith('W'))
    loss = (recon_loss + latent_kl + l2) / (32**2)

    print "Calculating gradient...",
    gradients = updates.clip_deltas(T.grad(loss, wrt=parameters), 5)
    print "Done with gradients."
    chunk_X = theano.shared(np.empty((1, 3, 64, 64), dtype=np.int32))
    idx = T.iscalar('idx')
    print "Compiling functions...",
    train = theano.function(
        inputs=[idx],
        outputs=[loss] + latent_kls,  # [T.sum(T.sqr(w)) for w in gradients],
        updates=updates.adam(parameters, gradients, learning_rate=3e-4),
        givens={X: chunk_X[idx * batch_size:(idx + 1) * batch_size]}
    )

    test = theano.function(inputs=[X], outputs=loss)
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
        stream = data_io.chunks((x[0] for x in stream), buffer_items=256)
        stream = data_io.async(stream, queue_size=3)
        total = 0
        count = 0
        for chunk in stream:
            total += chunk.shape[0] * test(chunk)
            count += chunk.shape[0]
        return total / count

    best_cost = np.inf
    P.save('model.pkl')
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
                print ' '.join(str(v) for v in loss)
                # pprint({p.name: g for p, g in zip(parameters, grad_norms)})
