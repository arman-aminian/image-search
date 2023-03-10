import numpy as np
import pandas as pd
import pickle

import io
import itertools
import csv
import yaml

from src.base_model.train.word2vec.word2vec import Word2Vec
from src.base_model.train.word2vec.utils import generate_training_data, text_preprocessing, vectorize_vocabs


import tensorflow as tf
from keras import Model, Sequential
from keras.layers import Activation, Dense, Dot, Embedding, Flatten, GlobalAveragePooling1D, Reshape
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from sklearn.model_selection import train_test_split


def train_word2vec(
    dataframe,
    params,
    embedding_dim = 512,
    SEED = 42,
    AUTOTUNE = tf.data.AUTOTUNE,
    BATCH_SIZE = 512,
    BUFFER_SIZE = 10000,
    vocab_size = 2000,
    sequence_length = 5,
    num_ns = 4
    ):
    '''Traing Word2Vec'''
    col_name = train_params['texts_col_name']
    df_stand = dataframe[col_name]

    # Vectorisation and Inverse Vocabulary
    inverse_vocab, sequences = vectorize_vocabs(
                    df_stand = df_stand,
                    vocab_size = 2000,
                    sequence_length = 5,
                    AUTOTUNE = tf.data.AUTOTUNE,
                    )
    

    targets, contexts, labels = generate_training_data(
        sequences=sequences, 
        window_size=4, 
        num_ns=num_ns, 
        vocab_size=vocab_size, 
        seed = 42)


    dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)


    word2vec = Word2Vec(vocab_size, embedding_dim, num_ns=num_ns)
    word2vec.compile(optimizer='adam',
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    word2vec.fit(dataset, epochs=params['epochs']) 


    w2v_embedding = word2vec.get_layer('w2v_embedding').get_weights()[0]
    ctxt_embedding = word2vec.get_layer('ctxt_embedding').get_weights()[0]

    # Save weights
    np.savez(params['result_path']+'w2v_embedding.npz', w2v_embedding)
    np.savez(params['result_path']+'ctxt_embedding.npz', ctxt_embedding)

    
    vocab = {}
    for i in range(len(inverse_vocab)):
        vocab[inverse_vocab[i]] = i
    
    # Save vocabs
    with open(params['result_path']+'vocabs.pkl', 'wb') as fp:
        pickle.dump(vocab, fp)




def compute_text_embedding(query: str, params, w2v_weights, w2v_vocabs, embedding_dim=512):
    '''Computting an embedding for a text'''
    query_embedding = None

    v = [0. for i in range(embedding_dim)]
    l = 0
    for word in (query.numpy()).decode('utf-8').split():
        word = '[UNK]' if word not in w2v_vocabs.keys() else word
        v += w2v_weights[w2v_vocabs[word]]
        l += 1
    query_embedding = v / l

    return query_embedding

    
def compute_texts_embedding(dataframe, params):
    '''Computting texts' embedding of a Dataframe'''
    col_name = params['texts_col_name']
    w2v_weights = np.load(open(params['result_path'] + 'w2v_embedding.npz','rb'))['arr_0']
    w2v_vocabs = pickle.load(open(params['result_path'] + 'vocabs.pkl','rb'))

    df['vec'] = df[col_name].apply(compute_text_embedding, args=[
                                                            params,
                                                            w2v_weights,
                                                            w2v_vocabs
                                                            ])
    return df


def read_dataset(dataset_path):
    '''Reading data from desired path as DataFrame'''
    df = pd.read_csv(dataset_path)
    return df
    
def split_dataset(dataframe, params):
    '''Splitting data to test and train '''
    train, test = train_test_split(dataframe, test_size=1 - params['train_size'] , random_state=41)
    return train.reset_index(), test.reset_index()

def save_df(dataframe, save_as: str, params):
    '''Saving dataframe as pickle file'''
    dataframe.to_pickle(params['dataset_path']+save_as)



if __name__ == '__main__':
    with open("params.yaml", "r") as stream:
        params = yaml.safe_load(stream)
    train_params = params['train']
    col_name = train_params['texts_col_name']

    df = read_dataset(train_params['dataset_path'])
    df[col_name] = text_preprocessing(df, col_name)

    train_df, test_df = split_dataset(df, train_params)
    train_word2vec(train_df, train_params)

    train_df = compute_texts_embedding(train_df, train_params)
    test_df = compute_texts_embedding(test_df, train_params)

    save_df(train_df, 'train_data', train_params)
    save_df(test_df, 'test_data', train_params)


