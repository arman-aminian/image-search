from typing import List, Dict
from sentence_transformers import SentenceTransformer
from numpy import dot
from numpy.linalg import norm
import seaborn as sns
from tqdm import tqdm


def _calc_similarity(sent1, sent2, model):
    """
    Returns cosine similarity of embeddings of to given sentences.
    """
    emb1, emb2 = model.encode([sent1, sent2])
    cos_sim = dot(emb1, emb2) / (norm(emb1)*norm(emb2))
    return cos_sim


def calculate_translation_score(data: List[Dict]) -> List[Dict]:
    """
    Adds a field to each row named 'translation_score' for similarity between 'comment' and its 'translation'.
    """
    # load model: a multilingual bert-based model trained for sentence similarity
    model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
    model = model.to(device="cuda:0")

    for i in tqdm(range(len(data))):
        data[i]['translation_score'] = _calc_similarity(data[i]['comment'], data[i]['translation'], model)

    return data


def draw_translation_scores_diagram(data: List[Dict]):
    """
    Draw's a histogram on translation scores for rows of a given dataset
    """
    sns.histplot([row['translation_score'] for row in data])


def filter_dataset_by_translation_score(data: List[Dict], translation_score_threshold: float) -> List[Dict]:
    """
    Filter rows with 'translation_score' of at least the given threshold.
    """
    filtered_data = [row for row in data if row['translation_score'] >= translation_score_threshold]
    return filtered_data
