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
    autoencoder, inpaint = model.build(P)

    parameters = P.values()
    X = T.itensor4('X')
    X_hat, posteriors, priors = autoencoder(T.cast(X, 'float32') / 255.)
    latent_kls = [T.mean(vae.kl_divergence(po_m, po_s, pr_m, pr_s), axis=0)
                  for (po_m, po_s), (pr_m, pr_s) in zip(posteriors, priors)]

    beta_start = 500 * (np.arange(len(latent_kls)) + 1)
    beta_lin = theano.shared(np.float32(0))
    betas_ = (beta_lin - beta_start) / np.float32(500)
    betas_ = T.switch(betas_ < 0, 0, betas_)
    betas = T.switch(betas_ > 1, 1, betas_)[::-1]
    print betas.eval()
    train_latent_kl = sum(betas[i] * kl for i, kl in enumerate(latent_kls))
    latent_kl = sum(latent_kls)
    recon_loss = model.cost(
        X_hat, X[:, :, 16:-16, 16:-16]
    )
    pprint(parameters)

    l2 = sum(T.sum(T.sqr(w)) for w in parameters)

    pretrain_loss = model.cost(
        inpaint(T.cast(X, 'float32') / np.float32(255.)),
        X[:, :, 16:-16, 16:-16]
    ) + 1e-2 * l2

    loss = (recon_loss + train_latent_kl + 1e-3 * l2) / (32**2)
    val_loss = (recon_loss + latent_kl) / (32**2)

    print "Calculating gradient...",
    gradients = updates.clip_deltas(T.grad(loss, wrt=parameters), 5)
    print "Done with gradients."
    chunk_X = theano.shared(np.empty((1, 3, 64, 64), dtype=np.int32))
    idx = T.iscalar('idx')
    print "Compiling functions...",
    pretrain = theano.function(
        inputs=[idx],
        outputs=[recon_loss / (32 * 32)] + latent_kls,
        updates=updates.adam(
            parameters,
            T.grad(pretrain_loss, wrt=parameters),
            learning_rate=1e-3
        ),
        givens={X: chunk_X[idx * batch_size:(idx + 1) * batch_size]}
    )

    train = theano.function(
        inputs=[idx],
        outputs=[recon_loss / (32 * 32)] + latent_kls,
        updates=updates.adam(parameters, gradients, learning_rate=1e-3) + [
            (beta_lin, beta_lin + 1)],
        givens={X: chunk_X[idx * batch_size:(idx + 1) * batch_size]}
    )

    test = theano.function(
        inputs=[X],
        outputs=[val_loss, recon_loss / (32 * 32)] + latent_kls
    )
    show_betas = theano.function(inputs=[], outputs=betas)
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
            vals = test(chunk)
            loss = vals[0]
            total += chunk.shape[0] * loss
            count += chunk.shape[0]
        print ' '.join(str(v) for v in vals),
        return total / count

    best_cost = np.inf
    for epoch in xrange(50):
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
                if epoch == 0:
                    loss = pretrain(i)
                else:
                    loss = train(i)
                print ' '.join(str(v) for v in loss)  # , show_betas()
                # pprint({p.name: g for p, g in zip(parameters, grad_norms)})
        P.save('unval_model.pkl')
