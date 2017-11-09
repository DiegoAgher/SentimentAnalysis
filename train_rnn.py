import numpy as np
import tensorflow as tf
from utils.data import get_IMDB_data
from utils.nlp import bucket_generator
from utils.training import build_embedding_ids_sequences, embedding_matrix,\
    hash_vector_map
from utils.tf_graphs import sequence2class_lstm_train_graph


def build_one_hot_encoding_target_variable(dataframe):
    negative_encoding = np.array([1, 0])
    positive_encoding = np.array([0, 1])
    numeric_response = (dataframe['sentiment'].
                        apply(lambda x: negative_encoding if x == 'neg'
                        else positive_encoding).values)
    return numeric_response.tolist()

train_data, test_data = get_IMDB_data()

train_embedding_sequences, exceptioned_sequences =\
    build_embedding_ids_sequences(train_data['text'].values)

test_embedding_sequences, test_exceptioned_sequences =\
    build_embedding_ids_sequences(test_data['text'].values)


en_oov_indicator = len(hash_vector_map)
en_bos_indicator = en_oov_indicator + 1
en_eos_indicator = en_bos_indicator + 1
en_pad_indicator = en_eos_indicator + 1


y_seq = build_one_hot_encoding_target_variable(train_data)

batch_size = 32
train_generator = bucket_generator(list(train_embedding_sequences.values()),
                                   y_seq, samples_threshold=100,
                                   batch_size=batch_size, pad=en_pad_indicator)

test_generator = bucket_generator(list(test_embedding_sequences.values()),
                                  y_seq, samples_threshold=100,
                                  batch_size=batch_size, pad=en_pad_indicator)

num_classes = 2
g = sequence2class_lstm_train_graph(batch_size=batch_size,
                                    train_embeddings=False,
                                    num_layers=2, num_classes=num_classes,
                                    state_size=100, learning_rate=1e-4,
                                    embedding_matrix=embedding_matrix)
saver = tf.train.Saver()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

num_epochs = 2
max_iterations = train_data.shape[0] / (num_epochs - 0.5)
training_state = None
training_loss = 0
accuracy_t = 0

inference_dict = {
    g['input_dropout_probability']: 1,
    g['output_dropout_probability']: 1,
                 }

for i, (seq_length, batch) in enumerate(train_generator):
    x, y = batch[:, 0, :], batch[:, 1, -num_classes:]
    x = np.reshape(x, (batch_size, seq_length))
    y = np.reshape(y, (batch_size, num_classes))
    feed_dict = {g['input_sequence']: x, g['labels']: y,
                 g['input_dropout_probability']: 0.5,
                 g['output_dropout_probability']: 0.5}

    if training_state is not None:
        feed_dict[g['initial_state']] = training_state

    training_loss_, _, training_state, accuracy_ = sess.run([g['total_loss'],
                                                             g['train_step'],
                                                             g['final_state'],
                                                             g['accuracy']],
                                                            feed_dict)
    training_loss += training_loss_
    accuracy_t += accuracy_
    if i % 20 == 0:
        _, test_batch = next(test_generator)
        x_test, y_test = test_batch[:, 0, :], test_batch[:, 1, -num_classes:]
        inference_dict[g['input_sequence']] = x_test
        inference_dict[g['labels']] = y_test

        test_accuracy, test_loss = sess.run([g['accuracy'], g['total_loss']])
        print("Train Metrics")
        print("Average loss", i, ":", training_loss / 20)
        print("Average accuracy ", i, ":", accuracy_t / 20)
        print("Test Metrics")
        print("Average loss", i, ":", test_loss / 20)
        print("Average accuracy ", i, ":", test_accuracy/ 20)
        print("=====")

        training_loss = 0
        accuracy_t = 0
        saver.save(sess, 'sentiment_analysis_tsm')

    if i > max_iterations:
        break

