import numpy as np
import tensorflow as tf
from utils.training import build_embedding_ids_sequences, embedding_matrix
from utils.tf_graphs import sequence2class_lstm_inference_graph


class SentimentAnalyzer(object):
    def __init__(self, model, embedding_matrix):
        graph = tf.Graph()
        with graph.as_default() as g:
            self.graph_elements = model(batch_size=1, num_layers=2,
                                        num_classes=2,
                                        embedding_matrix=embedding_matrix)
            self.saver = tf.train.Saver()
            self.graph = g
            self.positive_encoding = np.array([1, 0])
            self.negative_encoding = np.array([0, 1])

    def predict_text(self, text):
        mapped_text_dict, _ = build_embedding_ids_sequences(text)
        embedding_sequence = mapped_text_dict[0]
        mapped_text = np.reshape(embedding_sequence,
                                 (1, len(embedding_sequence)))
        inference_dict = {
            self.graph_elements['input_sequence']: mapped_text
                         }

        with tf.Session(graph=self.graph) as sess:
            self.saver.restore(sess=sess, save_path='sentiment_analysis_tsm')

            probabilities = sess.run(self.graph_elements['predictions'],
                                     inference_dict)
            prediction = np.argmax(probabilities)
            feeling_prob = probabilities[0][prediction]
            print("probabilities: ", probabilities)
            feeling = 'positive' if prediction == 1 else 'negative'
            print("""The text {0} has a {1} feeling with probability {2}""".
                  format(text, feeling, feeling_prob))


Analyzer = SentimentAnalyzer(sequence2class_lstm_inference_graph,
                             embedding_matrix)
