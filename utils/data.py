import os
import pandas as pd


def get_IMDB_data():

    train_data = get_train_data()
    test_data = get_test_data()


def get_train_data():
    data_path = '/data/aclImdb/train'
    return _get_parsed_data_from_reviews(data_path)


def get_test_data():
    data_path = '/data/aclImdb/test'
    return _get_parsed_data_from_reviews(data_path)


def _get_parsed_data_from_reviews(data_path):
    train_pos = data_path + '/pos'
    train_neg = data_path + '/neg'

    data = pd.DataFrame(columns=['id', 'sentiment', 'text', 'rating'])
    data = _parse_reviews_from_text_file(train_pos, data)
    data = _parse_reviews_from_text_file(train_neg, data)

    data.reset_index(inplace=True, drop=True)
    return data.sample(frac=1)


def _parse_reviews_from_text_file(reviews_path, dataframe):
    sentiment = reviews_path.split('/')[-1]
    files_list = os.listdir(reviews_path)
    for enum, file_name in enumerate(files_list):
        path_to_file = reviews_path + file_name
        with open(path_to_file, 'rb') as file:
            review_text = (file.readlines()[0]).decode('utf-8')
            review_text = review_text.replace('<br />', ' ')
        sep_position = file_name.index('_')
        review_id = file_name[0:sep_position]
        rating_val = float(file_name[sep_position + 1: sep_position + 3])
        entry_values = pd.DataFrame(
            {'id': review_id, 'sentiment': sentiment, 'text': review_text,
             'rating': rating_val}, index=[0])
        dataframe = dataframe.append(entry_values)
    return dataframe

