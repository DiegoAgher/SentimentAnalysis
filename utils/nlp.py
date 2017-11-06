import spacy
import numpy as np
from collections import defaultdict

en_nlp = spacy.util.get_lang_class('en')()

COMA_TOKEN = en_nlp(',')[0].lower
NUMBER_TOKEN = en_nlp('0')[0].lower

def try_to_number(string):
    try:
        float(string)
        return True
    except:
        return False

def get_clean_token_and_hash(token):
    token_hash = token.lower
    token_string = token.lower_

    if try_to_number(token_string):
        token_hash = NUMBER_TOKEN
        token_string = '0'

    if token.is_punct and token_string != '.':
        token_hash = COMA_TOKEN
        token_string = ','
    
    return token_string, token_hash

def build_hashed_documents(list_of_documents):
    english_id_sequences = {}
    sentence_exceptions_ids = {}

    for sent_id, sentence in enumerate(list_of_documents):
        try:
            doc = en_nlp(sentence)
        except Exception as e:
            sentence_exceptions_ids[sent_id] = sentence
            continue

        current_hash_sequence = []
        for token in doc:
            _, token_hash = get_clean_token_and_hash(token)
            current_hash_sequence.append(token_hash)

        english_id_sequences[sent_id] = current_hash_sequence
    return english_id_sequences, sentence_exceptions_ids

def get_string_or_unknown_from_hash(hash_):
    return en_nlp.vocab.strings.get(hash_, en_oov_indicator)

def read_vocabulary_vectors(vectors_path):
    with open(vectors_path) as glove_file:
        header = glove_file.readline()
        dims = len(header.split()) - 1
    
        num_vectors = 1
        for _ in glove_file:
            num_vectors += 1

        glove_file.seek(0)
    
        hash_embedding_map = {}
        embedding_matrix = np.ndarray((num_vectors + 4, dims), dtype=np.float32)
    
        for i, line in enumerate(glove_file):
            word, *vector_elements = line.split()
            vector = np.array([float(f) for f in vector_elements])
            embedding_matrix[i] = vector
            hash_ = spacy.strings.hash_string(word)
            hash_embedding_map[hash_] = i
        
        ## Inverse vocabulary
        embedding_hash_map = {v: k for k,v in hash_embedding_map.items()}
        
        ## Adding vector for oov/unknown token
        last_word_id = num_vectors - 1
        embedding_matrix[last_word_id + 1, :] = np.random.random(size=(1, dims))
        
        ## Adding vector for bos
        embedding_matrix[last_word_id + 2, :] = np.random.random(size=(1,dims))
        
        ## Adding vector for eos
        embedding_matrix[last_word_id + 3, :] = np.random.random(size=(1,dims))
        
        ## Adding vector for padd
        embedding_matrix[last_word_id + 4, :] = np.random.random(size=(1,dims))
        
    return embedding_matrix, hash_embedding_map, embedding_hash_map


def get_vector_id_from_hash(_hash, hash_vector_map):
    en_oov_indicator = len(hash_vector_map)
    return hash_vector_map.get(_hash, en_oov_indicator)

def get_string_from_hash(hash_, hash_vector_map):
    en_oov_indicator = len(hash_vector_map)
    en_bos_indicator = en_oov_indicator + 1
    en_eos_indicator = en_bos_indicator + 1
    en_pad_indicator = en_eos_indicator + 1
    
    if hash_ in en_nlp.vocab.strings:
        return en_nlp.vocab.strings[hash_]
    
    if hash_ == en_oov_indicator:
        return "UNKNWN"
    if hash_ == en_bos_indicator:
        return "BOS"
    if hash_ == en_eos_indicator:
        return "EOS"
    if hash_ == en_pad_indicator:
        return "PAD"


def get_hash_from_id(embedding_id):
    if embedding_id in vector_hash_map:
        return vector_hash_map[embedding_id]
    return embedding_id

def embed_subsequence_right(sequence, subsequence):
    """
    Insert subsequence in the rightmost part of sequence.
    """
    if len(subsequence) == 0:
        return sequence
    sequence[-len(subsequence):] = subsequence
    return sequence


class bucket_generator:
    def __init__(self, *lists_of_sequences,
                 labels=None, batch_size=64, samples_threshold=2000, pad=0):
        self.batch_size = batch_size
        self.num_sequences = len(lists_of_sequences)
        assert(self.num_sequences > 0)

        self.num_samples = len(lists_of_sequences[0])
        for sequence in lists_of_sequences:
            assert(len(sequence) == self.num_samples)
        if labels is not None:
            assert(len(labels) == self.num_samples)
            self.have_labels = True
        else:
            self.have_labels = False

        self.bucketize(lists_of_sequences, list_of_labels=labels)
        self.group_buckets(samples_threshold)
        self.create_bucket_matrices(pad)

        self.bucket_keys = list(self.buckets.keys())
        self.bucket_probs = [self.buckets[key]['size']/self.num_samples for key in self.bucket_keys]

        for bucket_key in self.bucket_keys:
            self.shuffle(bucket_key)

    def bucketize(self, lists_of_sequences, list_of_labels):
        buckets = defaultdict(list)
        for i in range(self.num_samples):
            sample_sequences = [list_[i] for list_ in lists_of_sequences]
            max_length = max([len(seq) for seq in sample_sequences])
            if self.have_labels:
                sample_sequences.append(list_of_labels[i])
            buckets[max_length].append(sample_sequences)
        self.buckets = buckets

    def group_buckets(self, samples_threshold):
        new_buckets = defaultdict(list)
        current_items = []
        for length in sorted(self.buckets.keys()):
            current_items.extend(self.buckets[length])
            self.buckets[length] = None
            if len(current_items) > samples_threshold:
                new_buckets[length] = current_items
                current_items = []
        new_buckets[length].extend(current_items)
        self.buckets = new_buckets

    def create_bucket_matrices(self, pad):
        bucket_matrices = {}
        for bucket_length in self.buckets.keys():
            bucket_elements = len(self.buckets[bucket_length])
            matrix = np.ndarray((bucket_elements, self.num_sequences, bucket_length), dtype=np.int32)
            if self.have_labels:
                labels = np.ndarray((bucket_elements), dtype=np.int32)
            for i, sequences in enumerate(self.buckets[bucket_length]):
                if self.have_labels:
                    label = sequences.pop()
                    labels[i] = label
                for j, sequence in enumerate(sequences):
                    padded_sequence = [pad] * bucket_length
                    embed_subsequence_right(padded_sequence, sequence)
                    matrix[i][j] = padded_sequence

            bucket_matrices[bucket_length] = {'matrix': matrix,
                                              'index': 0, 
                                              'size': bucket_elements}
            if self.have_labels:
                bucket_matrices[bucket_length]['labels'] = labels
        self.buckets = bucket_matrices

    def shuffle(self, bucket_length):
        permutation = np.random.permutation(self.buckets[bucket_length]['size'])
        matrix = self.buckets[bucket_length]['matrix']
        shuffled_matrix = np.empty(matrix.shape, dtype=matrix.dtype)
        if self.have_labels:
            labels = self.buckets[bucket_length]['labels']
            shuffled_labels = np.empty(labels.shape, dtype=labels.dtype)
        # Real shuffle
        for index, sindex in enumerate(permutation):
            shuffled_matrix[sindex] = matrix[index]
            if self.have_labels:
                shuffled_labels[sindex] = labels[index]
        self.buckets[bucket_length]['matrix'] = shuffled_matrix
        if self.have_labels:
            self.buckets[bucket_length]['labels'] = shuffled_labels
        self.buckets[bucket_length]['index'] = 0

    def __next__(self):
        bucket_length = np.random.choice(self.bucket_keys, p=self.bucket_probs)
        size = self.buckets[bucket_length]['size']
        begin = self.buckets[bucket_length]['index']
        end = begin + self.batch_size
        if end > size:
            self.shuffle(bucket_length)
            begin = self.buckets[bucket_length]['index']
            end = begin + self.batch_size
        self.buckets[bucket_length]['index'] = end
        if self.have_labels:
            return (bucket_length,
                    self.buckets[bucket_length]['matrix'][begin:end],
                    self.buckets[bucket_length]['labels'][begin:end])
        else:
            return (bucket_length, 
                    self.buckets[bucket_length]['matrix'][begin:end])

    def __iter__(self):
        return self

