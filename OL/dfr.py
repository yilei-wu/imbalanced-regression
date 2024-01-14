import torch
import torch.nn.functional as F
import numpy as np

def generate_gaussian_vectors(num_points, dimension):
    """
    Generates random Gaussian vectors, filters out those with norms greater than 1,
    and then normalizes the remaining vectors to have a norm of 1.

    :param num_points: Number of points to generate.
    :param dimension: Dimension of each vector.
    :return: A tensor of normalized Gaussian vectors with norm = 1.
    """
    # Generate random Gaussian vectors
    # torch.manual_seed(2024)
    
    gaussian_vectors = torch.randn(num_points, dimension)

    # Calculate the norms of the vectors
    norms = torch.norm(gaussian_vectors, dim=1, keepdim=True)

    normalized_vectors = gaussian_vectors / norms

    return normalized_vectors

def uniformity_loss(embeddings, points):
    # Generate the Gaussian vectors (assuming generate_gaussian_vectors is a valid function)
    # points = generate_gaussian_vectors(num_points, embeddings.shape[1]).to(embeddings.device)

    points = F.normalize(points, dim=1)
    embeddings = F.normalize(embeddings, dim=1)

    # Calculate cosine similarity (dot product since vectors are normalized)
    cosine_similarity = torch.matmul(embeddings, points.T)
    # print(torch.max(cosine_similarity.view(-1)), torch.min(cosine_similarity.view(-1))) [-1, 1]

    # Finding the indices of the maximum values in each column
    _, max_indices = torch.max(cosine_similarity, dim=0)

    # Creating a mask of the same size as cosine_similarity
    mask = torch.zeros_like(cosine_similarity, dtype=torch.bool)

    # Using the indices to set the maximum values in the mask
    mask.scatter_(0, max_indices.unsqueeze(0), True)

    # Average the maximum cosine similarity values for each column
    max_values = cosine_similarity[mask]
    # print(max_values.shape)
    average_max_similarity = torch.mean(max_values)

    return 1 - average_max_similarity

def smooth_loss(embeddings, alpha=1.0, beta=0.1):
    # Calculating the smoothness term
    embeddings = F.normalize(embeddings, dim=1)
    lengths = torch.matmul(embeddings[1:], embeddings[:-1].T)
    # get tthe diagonal elements
    lengths = torch.diag(lengths)

    # Regularization term (optional)
    regularization = torch.var(lengths)
    length = torch.sum(lengths)

    # Combined loss
    loss = alpha * length + beta * regularization
    return loss

def regression_contrastive_loss(embeddings, labels, surrogates, temperature=0.1, use_weight=False):
    """
    embeddings: Tensor of shape [N, d], where N is the number of samples, and d is the dimensionality of the embeddings
    labels: Tensor of shape [N], where each element is an integer label corresponding to the index in surrogates
    surrogates: Tensor of shape [C, d], where C is the number of classes
    temperature: A temperature scaling parameter
    """
    N, d = embeddings.shape
    C, _ = surrogates.shape

    # Normalize embeddings and surrogates to unit vectors
    embeddings_norm = F.normalize(embeddings, p=2, dim=1)
    surrogates_norm = F.normalize(surrogates, p=2, dim=1)

    # Compute dot product between embeddings and surrogates
    similarities = torch.matmul(embeddings_norm, surrogates_norm.T) # [N, C] * [C, T] = [N, T]
    # print(torch.max(similarities.view(-1)), torch.min(similarities.view(-1)))    
    similarities /= temperature

    mask = torch.zeros_like(similarities).to(similarities.device)
    label_range_temp = torch.tensor(np.linspace(-1.8, 1.8, 100)).cuda() 
    label_range_temp = label_range_temp.unsqueeze(1)
    labels = labels.unsqueeze(0)
    label_ind = torch.argmin(torch.abs(labels.detach().cpu() - label_range_temp.cpu().unsqueeze(0)), dim=1)
    mask[torch.arange(N), label_ind] = 1

    # compute log_prob
    p = mask / mask.sum(1, keepdim=True).clamp(min=1.0)

    logits_max, _ = torch.max(similarities, dim=-1, keepdim=True)
    similarities = similarities - logits_max.detach()
    
    if use_weight:
        label_range = torch.arange(200).to(labels.device)
        weights = torch.abs(labels.view(-1, 1) - label_range.view(1, -1)) + 1
        weights = weights / torch.max(weights) # [0, 1]
        # print("weights shape 1:", weights.shape)
    else:
        weights = torch.ones_like(similarities).to(similarities.device)
        # print("weights shape 2:", weights.shape)

    loss = compute_cross_entropy(p, similarities, weights)
    return loss

def compute_cross_entropy(p, q, weights):
    # q = F.log_softmax(q, dim=-1)
    # spell out the above formula q = F.log_softmax(q, dim=-1)
    q = q - torch.log(torch.sum(torch.exp(q) * weights, dim=-1, keepdim=True))
    loss = torch.sum(p * q, dim=-1)
    return - loss.mean()

def get_centroid(features, labels):
    
    unique_labels = torch.unique(labels)
    unique_labels, _ = torch.sort(unique_labels)
    centroids_list = []
    for label in unique_labels:
        cluster_features = features[labels == label]
        if cluster_features.size(0) > 0:
            centroids_list.append(cluster_features.mean(dim=0))
    centroids = torch.stack(centroids_list)
    return centroids, unique_labels.float()

def complete_centroid(centroids, centroids_label, label_range):
    # centroids: [C, D]
    # centroids_label: [C]
    # label_range: [L]
    # return: [L, D]

    device = centroids.device
    D = centroids.size(1)
    L = len(label_range)

    # Create a tensor of zeros with the size [L, D]
    complete_centroids = torch.zeros(L, D, device=device)

    # Scatter centroids into complete_centroids using centroids_label indices
    complete_centroids.scatter_(0, centroids_label.unsqueeze(1).expand(-1, D), centroids_label.unsqueeze(1).expand(-1, D).float())


    return complete_centroids

def dfr_simple(features, labels, points, label_range, surrogate, temperature=0.1, use_weight=False):
    # features: [N, D]
    # preds: [N]
    # labels: [N]
    # points: ranomly generated uniform points
    features = F.normalize(features, dim=1)
    centroids, centroids_label = get_centroid(features, labels) # lebels or preds
    
    centroids_complete = torch.zeros(len(label_range), features.size(1)).cuda()
    for j, i in enumerate(centroids_label):
        # the index of the label i in the label_range
        ind = torch.argmin(torch.abs(i - label_range)).long()
        centroids_complete[ind] = centroids[j]

    assert centroids_complete.shape == surrogate.shape, f"centroids_complete.shape: {centroids_complete.shape}, surrogate.shape: {surrogate.shape}"

    # replace the centroids with surrogate where the row is all zero 
    # print(torch.sum(centroids_complete, dim=1) == 0)
    # centroids_complete[torch.sum(centroids_complete, dim=1) == 0] = surrogate[torch.sum(centroids_complete, dim=1) == 0].detach()

    centroids = F.normalize(centroids, dim=1) 
    centroids_complete = F.normalize(centroids_complete, dim=1) 
    
    # label_long as the label for their nearset index in label_range
    # label_long = torch.argmin(torch.abs(centroids_complete - label_range), dim=1)

    # calcuate the loss
    loss_reg = None
    loss_con = regression_contrastive_loss(features, labels, centroids_complete.clone().detach(), temperature=temperature, use_weight=use_weight)
    loss_uni = uniformity_loss(centroids_complete, points)
    loss_smo = smooth_loss(centroids_complete)

    return loss_reg, loss_con, loss_uni, loss_smo

