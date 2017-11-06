import numpy as np
import tensorflow as tf
from utils.data import get_IMDB_data
from utils.nlp import build_hashed_documents
from utils.nlp import read_vocabulary_vectors, get_vector_id_from_hash
from utils.nlp import bucket_generator
from utils.tf_graphs import sequence2class_lstm_train_graph


def build_one_hot_encoding_target_variable(dataframe):
    negative_encoding = np.array([1, 0])
    positive_encoding = np.array([0, 1])
    numeric_response = (dataframe['sentiment'].
                        apply(lambda x: negative_encoding if x == 'neg'
                        else positive_encoding).values)
    return numeric_response.tolist()

train_data, test_data = get_IMDB_data()

hashed_sentences, exceptioned_docs =\
    build_hashed_documents(train_data['text'].values)

embedding_matrix, hash_vector_map, vector_hash_map =\
    read_vocabulary_vectors('/data/glove/glove.6B.300d.txt')

en_oov_indicator = len(hash_vector_map)
en_bos_indicator = en_oov_indicator + 1
en_eos_indicator = en_bos_indicator + 1
en_pad_indicator = en_eos_indicator + 1

hash_id_sequences = {sent_id:
                     [get_vector_id_from_hash(x, hash_vector_map)
                      for x in sentence]
                     for sent_id, sentence in hashed_sentences.items()
                     }

y_seq = build_one_hot_encoding_target_variable(train_data)

batch_size = 32
my_generator = bucket_generator(list(hash_id_sequences.values()),
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

max_iterations = 3000
training_state = None
training_loss = 0
accuracy_t = 0

for i, (seq_length, batch) in enumerate(my_generator):
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
        print("Average training loss for example", i, ":", training_loss / 20)
        print("Average training loss for example", i, ":", accuracy_t / 20)
        print("=====")

        training_loss = 0
        accuracy_t = 0
        saver.save(sess, 'sentiment_analysis_tsm')

    if i > max_iterations:
        break

