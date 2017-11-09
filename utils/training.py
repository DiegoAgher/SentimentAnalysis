from utils.nlp import build_hashed_documents
from utils.nlp import read_vocabulary_vectors, get_vector_id_from_hash

embedding_matrix, hash_vector_map, vector_hash_map = \
    read_vocabulary_vectors('data/glove/glove.6B.300d.txt')


def build_embedding_ids_sequences(text_sequences):
    hashed_sequences, exceptioned_seqs =\
        build_hashed_documents(text_sequences)

    embedding_id_sequences = {sequence_id:
                              [get_vector_id_from_hash(x, hash_vector_map)
                               for x in sequence]
                              for sequence_id, sequence in hashed_sequences.items()
                              }

    return embedding_id_sequences, exceptioned_seqs
