from sentence_transformers import SentenceTransformer
import math
import operator
import numpy as np

"""This module uses sentence-transformers https://github.com/UKPLab/sentence-transformers to compute sentence embeddings from 
state of the art transformer models using BERT. Cosine similarity between two sentence embeddings is returned which is used as the weight of
the edges when building the graph.

"""

# initialize embedding model
embedding_model = SentenceTransformer('bert-base-nli-mean-tokens')

def _dot_product2(v1, v2):
    return sum(map(operator.mul, v1, v2))

def _fast_cosine_similarity(v1, v2):
    prod = _dot_product2(v1, v2)
    len1 = math.sqrt(_dot_product2(v1, v1))
    len2 = math.sqrt(_dot_product2(v2, v2))
    return prod / (len1 * len2)

def _get_sentence_embedding(sentence):
    embeddings = embedding_model.encode([sentence])
    return embeddings[0]

def get_similarity_score(sentence_1, sentence_2):
    vec_1 = _get_sentence_embedding(sentence_1)
    vec_2 = _get_sentence_embedding(sentence_2)
    return _fast_cosine_similarity(vec_1, vec_2) 
