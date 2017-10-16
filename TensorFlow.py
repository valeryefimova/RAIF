import tensorflow as tf


def read_my_file(filename_queue):
    reader = tf.TextLineReader(skip_header_lines=1)
    unused, file_lines = reader.read(filename_queue)
    # 76 default values:
    default_tensors = [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
                       [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
                       [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
                       [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
                       [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
                       [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]

    columns = tf.decode_csv(file_lines, record_defaults=default_tensors, field_delim=";")
    features = tf.stack(columns[0:75])
    label = tf.stack(columns[75])
    return features, label


filename_queue = tf.train.string_input_producer(["data/transformed.csv"])

with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    features, label = read_my_file(filename_queue=filename_queue)
    X, Y = tf.train.batch([features, label], batch_size=8519)
    tf.set_random_seed(42)

    # TODO: actual model training should be here:
    # ...



    # stop and join all threads
    coord.request_stop()
    coord.join(threads)
