import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell, DropoutWrapper, MultiRNNCell


def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()


def sequence2class_lstm_train_graph(embedding_matrix, batch_size=25,
                                    train_embeddings=False, num_layers=2,
                                    num_classes=2, state_size=100,
                                    learning_rate=1e-4):

    input_sequence = tf.placeholder(tf.int32, [batch_size, None],
                                    name='input_sequence')

    labels = tf.placeholder(tf.int32, [batch_size, 2], name='labels')

    embeddings = tf.constant(embedding_matrix)

    if not train_embeddings:
        embeddings = tf.stop_gradient(embeddings)

    rnn_input = tf.nn.embedding_lookup(embeddings, input_sequence)

    lstm = BasicLSTMCell
    lstms_list = [lstm(state_size, state_is_tuple=True)
                  for _ in range(0, num_layers - 1)]

    input_dropout_probability = tf.placeholder(tf.float32,
                                               name='input_dropout_prob')

    output_dropout_probability = tf.placeholder(tf.float32,
                                                name='output_dropout_prob')

    lstm_dropout_list = [
        DropoutWrapper(
            basic_cell,
            input_keep_prob=input_dropout_probability,
            output_keep_prob=output_dropout_probability)

        for basic_cell in lstms_list
        ]
    cell = MultiRNNCell(lstm_dropout_list, state_is_tuple=True)

    initial_state = cell.zero_state(batch_size, tf.float32)

    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_input,
                                                 initial_state=initial_state)

    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_classes], dtype=tf.float32)
        b = tf.get_variable('b', [num_classes],
                            initializer=tf.constant_initializer(0.0),
                            dtype=tf.float32)

    logits = tf.matmul(rnn_outputs[:, -1, :], W) + b

    predictions = tf.nn.softmax(logits)
    compare_score_label = tf.equal(tf.argmax(predictions, axis=1),
                                   tf.argmax(labels, axis=1))

    accuracy = tf.reduce_mean(tf.cast(compare_score_label, tf.float32))

    total_loss = tf.reduce_mean(tf.nn.
                                softmax_cross_entropy_with_logits
                                (logits=logits, labels=labels))

    train_step = (tf.train.
                  AdamOptimizer(learning_rate=learning_rate).
                  minimize(total_loss))

    return dict(input_sequence=input_sequence,
                labels=labels,
                total_loss=total_loss,
                accuracy=accuracy,
                train_step=train_step,
                initial_state=initial_state,
                final_state=final_state,
                predictions=predictions,
                input_dropout_probability=input_dropout_probability,
                output_dropout_probability=output_dropout_probability)


def sequence2class_lstm_inference_graph(embedding_matrix,
                                        batch_size=25,
                                        num_layers=2,
                                        num_classes=2,
                                        state_size=100):

    input_sequence = tf.placeholder(tf.int32, [batch_size, None],
                                    name='input_sequence')

    labels = tf.placeholder(tf.int32, [batch_size, 2], name='labels')

    embeddings = tf.constant(embedding_matrix)

    embeddings = tf.stop_gradient(embeddings)

    rnn_input = tf.nn.embedding_lookup(embeddings, input_sequence)

    lstm = BasicLSTMCell
    lstms_list = [lstm(state_size, state_is_tuple=True)
                  for _ in range(0, num_layers - 1)]

    cell = MultiRNNCell(lstms_list, state_is_tuple=True)

    initial_state = cell.zero_state(batch_size, tf.float32)

    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_input,
                                                 initial_state=initial_state)

    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_classes], dtype=tf.float32)
        b = tf.get_variable('b', [num_classes],
                            initializer=tf.constant_initializer(0.0),
                            dtype=tf.float32)

    logits = tf.matmul(rnn_outputs[:, -1, :], W) + b

    predictions = tf.nn.softmax(logits)
    compare_score_label = tf.equal(tf.argmax(predictions, axis=1),
                                   tf.argmax(labels, axis=1))

    accuracy = tf.reduce_mean(tf.cast(compare_score_label, tf.float32))

    return dict(input_sequence=input_sequence,
                labels=labels,
                accuracy=accuracy,
                initial_state=initial_state,
                final_state=final_state,
                predictions=predictions)
