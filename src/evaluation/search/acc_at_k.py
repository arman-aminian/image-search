import numpy as np
import copy
from numpy import dot
from numpy.linalg import norm
import matplotlib.pyplot as plt
import seaborn as sns


def cosine_similarity(emb1, emb2):
    cos_sim = dot(emb1, emb2) / (norm(emb1) * norm(emb2))
    return cos_sim


def calculate_pairwise_similarity(text_embeddings, image_embeddings):
    cosine_matrix = []
    for text_embedding in tqdm(text_embeddings):
        cosine_matrix.append(
            [cosine_similarity(text_embedding, image_embedding) for image_embedding in image_embeddings])
    return cosine_matrix


def is_value_in_top_k(k, value, array):
    array.sort(reverse=True)
    rank = array.index(value) + 1
    return True if rank <= k else False


def accuracy_at_k(k, cosine_matrix):
    is_in_top_k_count = 0
    for i in range(len(cosine_matrix)):
        if is_value_in_top_k(k, cosine_matrix[i][i], copy.deepcopy(cosine_matrix[i])):
            is_in_top_k_count += 1
    return is_in_top_k_count / len(cosine_matrix)


def calculate_accuracy_at_k(cosine_matrix):
    accuracy_at = {}
    for k in list(range(1, 50)) + list(range(50, 1001, 20)):
        accuracy_at[k] = accuracy_at_k(k, cosine_matrix)
        print(f"accuracy @ {k:<3}: {accuracy_at[k]}")
    return accuracy_at


def plot_accuracy_at_k(accuracy_at):
    sns.set(rc={'figure.figsize': (11.7, 8.27)})
    sns.lineplot(x=accuracy_at.keys(), y=accuracy_at.values())
    plt.plot()

    plt.xscale('log')
    sns.lineplot(x=accuracy_at.keys(), y=accuracy_at.values())
    plt.plot()
