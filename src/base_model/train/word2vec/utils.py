import numpy as np
import pandas as pd

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

def text_standardization(input_data, preprocessor = Preprocessor()):
        cleaned_input = preprocessor.cleaning(input_data)
        return tf.convert_to_tensor(cleaned_input, dtype=tf.string) 

def text_preprocessing(input_dataframe):
    def text_standardization(input_data):
        cleaned_input = preprocessor.cleaning(input_data)
        return tf.convert_to_tensor(cleaned_input, dtype=tf.string) 
  
    preprocessor = Preprocessor()
    return input_dataframe['caption'].apply(text_standardization)


def vectorize_vocabs(
                    df_stand,
                    vocab_size = 2000,
                    sequence_length = 5,
                    AUTOTUNE = tf.data.AUTOTUNE,
                    ):
    vectorize_layer = TextVectorization(
        #standardize=custom_standardization,
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