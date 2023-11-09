import numpy as np
import matplotlib.pyplot as plt

def generate_gaussian_vectors(num_points, dimension):
    # num_points: number of uniformly distributed points, the more the better;
    # dimension: hidden dimention of our representations.
    # You only need to run this function once, before training.
    
    # Generate random Gaussian vectors
    gaussian_vectors = np.random.normal(size=(num_points, dimension))
    
    # Normalize each vector to have a norm of 1
    norms = np.linalg.norm(gaussian_vectors, axis=1, keepdims=True)
    unit_vectors = gaussian_vectors / norms

    return unit_vectors

def uniformity_loss(embeddings, num_points, epsilon):
    # Generate the Gaussian vectors
    points = generate_gaussian_vectors(num_points, embeddings.shape[1])
    
    # Calculate cosine similarity (dot product since vectors are normalized)
    cosine_similarity = np.dot(embeddings, points.T)
    
    # Check for similarities less than epsilon
    close_points = np.any(cosine_similarity > (1 - epsilon), axis=1)
    
    # Calculate percentage of embeddings with at least one point close enough
    percentage = np.mean(close_points) * 100
    
    return percentage

def smooth_loss(embeddings): # assume the embeddings are already ordered by the label
    
    differences = embeddings[1:] - embeddings[:-1]
    
    distances = np.linalg.norm(differences, axis=1)
    
    total_distance = np.sum(distances) / (2 * (len(embeddings) - 1))
    
    return total_distance

def disparity_loss(embeddings_A, labels_A, embeddings_B, labels_B):# assume two set of embeddings are already ordered by the label
    """
    Calculates the normalized sum of Euclidean distances between embeddings in set B and their matching embeddings in set A.
    """
    
    # Get unique labels in 'labels_B' and their inverse mapping to reconstruct the original 'labels_B' order
    unique_labels_B, inverse_indices = np.unique(labels_B, return_inverse=True)

    # Find indices in 'labels_A' for each unique label in 'labels_B'
    indices_for_unique = np.searchsorted(labels_A, unique_labels_B)

    # Use the inverse mapping to get the corresponding indices for the original 'labels_B'
    corresponding_indices = indices_for_unique[inverse_indices]

    # Take the embeddings from 'embeddings_A' at the found indices
    matched_embeddings_A = embeddings_A[corresponding_indices]

    # Compute the Euclidean distances
    distances = np.linalg.norm(embeddings_B - matched_embeddings_A, axis=1)

    # Sum the distances
    total_distance = np.sum(distances)

    # Normalize the sum to the range of 0-1 by dividing by the maximum possible sum of distances
    normalized_total_distance = total_distance / len(embeddings_B)

    return normalized_total_distance