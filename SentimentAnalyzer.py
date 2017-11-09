import numpy as np
import tensorflow as tf
from utils.training import build_embedding_ids_sequences, embedding_matrix
from utils.tf_graphs import sequence2class_lstm_inference_graph


class SentimentAnalyzer(object):
    def __init__(self, model, embedding_matrix):
        with tf.Graph.as_default() as graph:
            self.graph_elements = model(batch_size=1, num_layers=2, num_classes=2,
                                        embedding_matrix=embedding_matrix)
            self.saver = tf.train.Saver()
            self.graph = graph
            self.positive_encoding = np.array([1, 0])
            self.negative_encoding = np.array([0, 1])

    def predict_text(self, text):
        mapped_text, _ = build_embedding_ids_sequences(text)
        inference_dict = {
            self.graph_elements['input_sequence']: mapped_text
                         }

        with tf.Session(graph=self.graph) as sess:
            self.saver.restore(sess=sess, save_path='sentiment_analysis_tsm')

            probabilities = sess.run(g['predictions'], inference_dict)
            predictions = np.argmax(probabilities)
            print(predictions)
            if predictions == self.positive_encoding:
                print("Text: ", text, """ has a positive feeling with
                 probability """, probabilities[np.argmax(probabilities)])


Analyzer = SentimentAnalyzer(sequence2class_lstm_inference_graph,
                             embedding_matrix)