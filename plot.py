import theano
import theano.tensor as T
import numpy as np
from theano_toolkit.parameters import Parameters
import data_io
import model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
ITERATION_STEPS = 40
if __name__ == "__main__":
    P = Parameters()
    inpaint = model.build(P)

    X = T.itensor4('X')
    X_hat, _ = inpaint(T.cast(X, 'float32'),
                    iteration_steps=ITERATION_STEPS)
    val_loss = model.cost(X_hat[-1:], X)
    Y = model.predict(X_hat)
    fill = theano.function(inputs=[X], outputs=[Y, val_loss])
    P.load('model.pkl')
    stream = data_io.stream_file("data/val2014.pkl.gz")
    stream = data_io.buffered_random(stream)
    stream = data_io.chunks((x[0] for x in stream), buffer_items=10)
    for chunk in stream:
        output, loss = fill(chunk)
        break
    print loss
    fig = plt.figure(figsize=(20, 5))
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1,
                        wspace=None, hspace=None)

    def plot(i):
        global chunk
        i = ITERATION_STEPS - 1 if i >= ITERATION_STEPS else i
        chunk_filled = chunk.copy()
        chunk_temp = chunk.copy()

        chunk_temp = chunk_temp.transpose(2, 0, 3, 1)
        chunk_temp = chunk_temp / 256.

        chunk_filled[:, :, 16:48, 16:48] = output[i]
        chunk_filled = chunk_filled.transpose(2, 0, 3, 1)
        chunk_filled = chunk_filled / 256.
        plot_data = np.concatenate((chunk_temp, chunk_filled), axis=0)

        plt.imshow(
            plot_data.reshape(chunk_temp.shape[0] + chunk_filled.shape[0],
                              chunk_temp.shape[1] * chunk_temp.shape[2],
                              chunk_temp.shape[3]), interpolation='None')

    anim = FuncAnimation(fig, plot, frames=np.arange(0, ITERATION_STEPS + 5), interval=200)
    anim.save('sample.gif', dpi=80, writer='imagemagick')
