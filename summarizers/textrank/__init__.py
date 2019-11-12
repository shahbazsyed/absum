import sys
sys.path.insert(1, '.')

import editdistance
import io
import itertools
import networkx as nx
import syntok.segmenter as sentence_segmenter
import os
from collections import OrderedDict
from math import log10
from summarizers.embeddings import sentence_embeddings


def _count_common_words(words_sentence_one, words_sentence_two):
    return len(set(words_sentence_one) & set(words_sentence_two))


def _build_graph(nodes, weight_function):
    """Return a networkx graph instance.

    :param nodes: List of hashables that represent the nodes of a graph.
    """
    gr = nx.Graph()  # initialize an undirected graph
    gr.add_nodes_from(nodes)
    nodePairs = list(itertools.combinations(nodes, 2))

    # add edges to the graph (weighted by weight_function)
    for pair in nodePairs:
        firstString = pair[0]
        secondString = pair[1]
        edge_weight = weight_function(firstString, secondString)
        gr.add_edge(firstString, secondString, weight=edge_weight)

    return gr


def _edit_distance(sentence_1, sentence_2):
    return editdistance.eval(sentence_1, sentence_2)

def _lexical_overlap(sentence_1, sentence_2):
    print(type(sentence_1))
    print(type(sentence_2))
    print('\n')
    words_sentence_one = sentence_1.split()
    words_sentence_two = sentence_2.split()
    common_word_count = _count_common_words(words_sentence_one, words_sentence_two)

    log_s1 = log10(len(words_sentence_one))
    log_s2 = log10(len(words_sentence_two))

    if log_s1 + log_s2 == 0:
        return 0
    return common_word_count / (log_s1 + log_s2)

def _get_sentence_embedding_similarity(sentence_1, sentence_2):
    score = sentence_embeddings.get_similarity_score(sentence_1, sentence_2)
    return round(score,3)


def summarize(text, ratio=0.2, weight_function="edit_distance"):
    """Return a paragraph style summary of the source text
    
    Args:
        text (string): Text to be summarized
        ratio (float, optional): Summary length in terms of ratio. Defaults to 0.2.
        weight_function (str, optional): Weight function to compute edge weights for graph. Defaults to "edit_distance". Options [edit_distance, lexical_overlap. embedding_similarity]
    """
    processed_segments = sentence_segmenter.process(text)
    sentences = []
    for paragraph in processed_segments:
        for sentence in paragraph:
            sentences.append("".join(map(str, sentence)).lstrip())
    
    if weight_function == "edit_distance":
        graph = _build_graph(sentences, _edit_distance)
    if weight_function == "lexical_overlap":
        graph = _build_graph(sentences, _lexical_overlap)
    if weight_function == "embedding_similarity":
        graph = _build_graph(sentences, _get_sentence_embedding_similarity)
    calculated_page_rank = nx.pagerank(graph, weight='weight')

    # most important sentences in ascending order of importance
    key_sentences = sorted(calculated_page_rank, key=calculated_page_rank.get,
                       reverse=True)
    # calculate ratio of important sentences to be returned as summary
    length = len(key_sentences) * ratio
    _temp_sentences = key_sentences[:int(length)]
    _summary_sentences = [(i,item) for i, item in enumerate(_temp_sentences)]
    _summary_sentences.sort(key=lambda x: sentences.index(x[1]))
    summary_sentences = [item for i, item in _summary_sentences]
    return ' '.join(summary_sentences)
