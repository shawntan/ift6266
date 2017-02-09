import theano
import theano.tensor as T
import numpy as np
from theano_toolkit.parameters import Parameters
import data_io
import model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if __name__ == "__main__":
    P = Parameters()
    inpaint = model.build(P)

    X = T.itensor4('X')
    Y = model.predict(inpaint(T.cast(X, 'float32')))
    fill = theano.function(inputs=[X], outputs=Y)
    P.load('model.pkl')
    stream = data_io.stream_file("data/val2014.pkl.gz")
    stream = data_io.buffered_random(stream)
    stream = data_io.randomised_chunks((x[0] for x in stream),
                                       buffer_items=10)
    for chunk in stream:
        output = fill(chunk)
        break
    chunk_filled = chunk.copy()
    chunk = chunk.transpose(2, 0, 3, 1)
    chunk = chunk / 256.

    chunk_filled[:, :, 16:48, 16:48] = output
    chunk_filled = chunk_filled.transpose(2, 0, 3, 1)
    chunk_filled = chunk_filled / 256.
    plot_data = np.concatenate((chunk, chunk_filled), axis=0)

    plt.figure(figsize=(30, 150))
    plt.imshow(plot_data.reshape(chunk.shape[0] + chunk_filled.shape[0],
                                 chunk.shape[1] * chunk.shape[2],
                                 chunk.shape[3]), interpolation='None')
    plt.savefig('sample.png', bbox_inches='tight')
