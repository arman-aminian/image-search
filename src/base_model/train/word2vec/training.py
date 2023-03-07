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


def train_doc2vec(
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

    df_stand = text_preprocessing(dataframe)
    
    #df_stand =  pd.DataFrame(list(df_stand))
        

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

    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")
    word2vec.fit(dataset, epochs=params['epochs'])  #, callbacks=[tensorboard_callback]


    weights1 = word2vec.get_layer('w2v_embedding').get_weights()[0]
    weights2 = word2vec.get_layer('ctxt_embedding').get_weights()[0]

    # Save weights
    np.savez(params['result_path']+'w2v_embedding.npz', weights1)
    np.savez(params['result_path']+'ctxt_embedding.npz', weights2)

    
    vocab = {}
    for i in range(len(inverse_vocab)):
        vocab[inverse_vocab[i]] = i
    
    # Save vocabs
    with open(params['result_path']+'vocabs.pkl', 'wb') as fp:
        pickle.dump(vocab, fp)


    # Text embedding
    lst = []
    
    for i in range(df_stand.size):
        v = np.array([0. for i in range(embedding_dim)])
        l = 0
        for word in (df_stand[i].numpy()).decode('utf-8').split():
            word = '[UNK]' if word not in vocab.keys() else word
            v += weights1[vocab[word]]
            l += 1
        lst.append(v / l)

    lst = [str(l) for l in lst]

    # Writing texts embedds to CSV
    with open(params['result_path']+"text_datas.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(lst)

    return lst


def read_dataset(dataset_path):
    df = pd.read_csv(dataset_path)
    return df
    
def split_dataset(dataframe, params):
    train, test = train_test_split(dataframe, test_size=1 - params['train_size'] , random_state=42)
    return train.reset_index(), test.reset_index()

def join_df(train_df, embedded_docs, params):
    # joining embedding of texts to corresponded text
    final_df = train_df[['image']]
    final_df['vec'] = embedded_docs
    final_df.to_csv(params['result_path']+'train_datas.csv')


if __name__ == '__main__':
    with open("params.yaml", "r") as stream:
        params = yaml.safe_load(stream)
    train_params = params['train']
    df = read_dataset(train_params['dataset_path'])
    train_df, _ = split_dataset(df, train_params)
    embedded_docs = train_doc2vec(train_df, train_params)
    join_df(train_df, embedded_docs, train_params)

