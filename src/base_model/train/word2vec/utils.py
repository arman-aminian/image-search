import numpy as np
import pandas as pd
import pickle

import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from src.base_model.train.word2vec.preprocessing import Preprocessor


def generate_training_data(sequences, window_size, num_ns, vocab_size, seed):

    '''
    Generates skip-gram pairs with negative sampling for a list of sequences(int-encoded sentences) 
    based on window size, number of negative samples and vocabulary size.
    '''

    # Elements of each training example are appended to these lists.
    targets, contexts, labels = [], [], []

    # Build the sampling table for vocab_size tokens.
    sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

    # Iterate over all sequences (sentences) in dataset.
    for sequence in sequences:

        # Generate positive skip-gram pairs for a sequence (sentence).
        positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
            sequence, 
            vocabulary_size=vocab_size,
            sampling_table=sampling_table,
            window_size=window_size,
            negative_samples=0)

        # Iterate over each positive skip-gram pair to produce training examples 
        # with positive context word and negative samples.
        for target_word, context_word in positive_skip_grams:
            context_class = tf.expand_dims(
                tf.constant([context_word], dtype="int64"), 1)
            negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
                true_classes=context_class,
                num_true=1, 
                num_sampled=num_ns, 
                unique=True, 
                range_max=vocab_size, 
                seed=seed, 
                name="negative_sampling")

            # Build context and label vectors (for one target word)
            negative_sampling_candidates = tf.expand_dims(
                negative_sampling_candidates, 1)

            context = tf.concat([context_class, negative_sampling_candidates], 0)
            label = tf.constant([1] + [0]*num_ns, dtype="int64")

            # Append each element from the training example to global lists.
            targets.append(target_word)
            contexts.append(context)
            labels.append(label)

    return targets, contexts, labels

def text_standardization(input_data, preprocessor):
    '''Text preprocessing'''
    cleaned_input = preprocessor.cleaning(input_data)
    return tf.convert_to_tensor(cleaned_input, dtype=tf.string) 

def text_preprocessing(input_dataframe,col_name):
    '''Preprocess Texts of a Dataframe's column'''
    preprocessor = Preprocessor()
    return input_dataframe[col_name].apply(text_standardization, args=[preprocessor])


def vectorize_vocabs(
                    df_stand,
                    vocab_size = 2000,
                    sequence_length = 5,
                    AUTOTUNE = tf.data.AUTOTUNE,
                    ):
    vectorize_layer = TextVectorization(
        max_tokens=vocab_size,
        output_mode='int',
        output_sequence_length=sequence_length)

    text_dataset = tf.data.Dataset.from_tensor_slices(list(df_stand))
    
    t = text_dataset.batch(1024)
    vectorize_layer.adapt(t)
    inverse_vocab = vectorize_layer.get_vocabulary()

    text_vector_ds = text_dataset.batch(1024).prefetch(AUTOTUNE).map(vectorize_layer).unbatch()
    sequences = list(text_vector_ds.as_numpy_iterator())
    return inverse_vocab, sequences


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


