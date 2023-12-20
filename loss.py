import torch

def generate_gaussian_vectors(num_points, dimension):
    """
    Generates random Gaussian vectors, filters out those with norms greater than 1, 
    and then normalizes the remaining vectors to have a norm of 1.

    :param num_points: Number of points to generate.
    :param dimension: Dimension of each vector.
    :return: A tensor of normalized Gaussian vectors with norm = 1.
    """
    # Generate random Gaussian vectors
    gaussian_vectors = torch.randn(num_points, dimension)
    
    # Calculate the norms of the vectors
    norms = torch.norm(gaussian_vectors, dim=1, keepdim=True)

    normalized_vectors = gaussian_vectors / norms

    return normalized_vectors

def uniformity_loss(embeddings, num_points, epsilon):
    # Generate the Gaussian vectors
    points = generate_gaussian_vectors(num_points, embeddings.shape[1])
    
    # Calculate cosine similarity (dot product since vectors are normalized)
    cosine_similarity = torch.matmul(embeddings, points.T)
    
    # Check for similarities less than epsilon
    close_points = torch.any(cosine_similarity > (1 - epsilon), dim=1)
    
    # Calculate percentage of embeddings with at least one point close enough
    percentage = torch.mean(close_points.float()) * 100
    
    return percentage

def smooth_loss(embeddings):
    differences = embeddings[1:] - embeddings[:-1]
    distances = torch.norm(differences, dim=1)
    total_distance = torch.sum(distances) / (2 * (embeddings.size(0) - 1))
    return total_distance

def disparity_loss(embeddings_A, labels_A, embeddings_B, labels_B):
    # Find matching embeddings in A for each label in B
    matched_embeddings_A = embeddings_A[labels_A.argsort()[labels_B.argsort().argsort()]]
    
    # Compute the Euclidean distances
    distances = torch.norm(embeddings_B - matched_embeddings_A, dim=1)

    # Sum and normalize the distances
    normalized_total_distance = torch.sum(distances) / embeddings_B.size(0)

    return normalized_total_distance


if __name__ == '__main__':
    # test
    points = generate_gaussian_vectors(10000, 512)
    print(points.shape)