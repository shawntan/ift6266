import gzip
import cPickle as pickle
import numpy as np
import random


def stream_file(filename, open_method=gzip.open):
    with open_method(filename, 'rb') as fd:
        try:
            while True:
                x = pickle.load(fd)
                yield x
        except EOFError:
            pass


def async(stream, queue_size):
    import threading
    import Queue
    queue = Queue.Queue(maxsize=queue_size)
    end_marker = object()

    def producer():
        for item in stream:
            queue.put(item)
        queue.put(end_marker)

    thread = threading.Thread(target=producer)
    thread.daemon = True
    thread.start()
    # run as consumer
    item = queue.get()
    while item is not end_marker:
        yield item
        queue.task_done()
        item = queue.get()


def buffered_random(stream, buffer_items=512, leak_percent=0.9):
    item_buffer = [None] * buffer_items
    leak_count = int(buffer_items * leak_percent)
    item_count = 0
    for item in stream:
        item_buffer[item_count] = item
        item_count += 1
        if buffer_items == item_count:
            random.shuffle(item_buffer)
            for item in item_buffer[leak_count:]:
                yield item
            item_count = leak_count
    if item_count > 0:
        item_buffer = item_buffer[:item_count]
        random.shuffle(item_buffer)
        for item in item_buffer:
            yield item


def chunks(stream, buffer_items=512):
    chunk = None
    buffer_idx = 0
    for item in stream:
        if chunk is None:
            chunk = np.empty((buffer_items,) + item.shape, dtype=np.int32)
        chunk[buffer_idx] = item
        buffer_idx += 1
        if buffer_idx == buffer_items:
            yield chunk
            buffer_idx = 0
    if buffer_idx > 0:
        yield chunk[:buffer_idx]

if __name__ == "__main__":
    stream = stream_file("data/train2014.pkl.gz")
    stream = buffered_random(stream)
    stream = async(randomised_chunks(x[0] for x in stream), queue_size=100)
    print sum(s.shape[0] for s in stream)
