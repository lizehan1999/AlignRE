import torch
import numpy as np


def calculate_similarity(query_vector, target_vectors):
    similarity = torch.cosine_similarity(query_vector, target_vectors, dim=-1)
    return similarity


def weight_mean(sentence_vectors):
    def cosine_similarity_(v1, v2):
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        similarity = dot_product / (norm_v1 * norm_v2)
        return similarity

    num_sentences = len(sentence_vectors)

    similarity_matrix = np.zeros((num_sentences, num_sentences))
    for i in range(num_sentences):
        for j in range(num_sentences):
            similarity_matrix[i, j] = cosine_similarity_(sentence_vectors[i], sentence_vectors[j])

    sentence_weights = np.mean(similarity_matrix, axis=1)

    merged_vector = np.average(sentence_vectors, axis=0, weights=sentence_weights)

    return merged_vector
