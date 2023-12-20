import torch
import torch.nn as nn
import torch.nn.functional as F
from loss import uniformity_loss, smooth_loss, regression_contrastive_loss

# write a function for get the centroid of the representation
# patchify the label space (to do) 
def get_centroid(features, labels):
    # features: [N, D]
    # labels: [N]
    # return: [C, D]
    unique_labels = torch.unique(labels)
    # sort the unique labels
    unique_labels, _ = torch.sort(unique_labels)
    
    centroids = nn.Parameter(torch.zeros(len(unique_labels), features.size(1)).to(features.device))
    centroids = centroids.clone()  # Add this line before the loop
    for i, label in enumerate(unique_labels):
        cluster_features = features[labels == label]
        if cluster_features.size(0) > 0:  # Check if the cluster is not empty
            centroids[i] = cluster_features.mean(dim=0)
    return centroids, unique_labels.long()


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

    # Use advanced indexing to update complete_centroids
    # This operation maintains the gradient flow from centroids
    complete_centroids = complete_centroids.scatter_(0, centroids_label.unsqueeze(1).expand(-1, D), centroids)

    # Convert to nn.Parameter to include in the model's parameters
    complete_centroids = nn.Parameter(complete_centroids)

    return complete_centroids

def dfr(features, preds, labels, label_range, seq2seq):
    # features: [N, D]
    # preds: [N]
    # labels: [N]
    # seq2seq: a seq2seq model

    features = F.normalize(features, dim=1)
    centroids, centroids_label = get_centroid(features, labels) # lebels or preds
    centroids = complete_centroid(centroids, centroids_label, torch.tensor(label_range))

    # plain trasnfomer
    surrogate = seq2seq(centroids.unsqueeze(0).detach())
    if isinstance(surrogate, tuple):
        surrogate = surrogate[0]
    surrogate = surrogate.squeeze(0) # [L, D]

    # calcuate the loss
    loss_reg = F.l1_loss(preds, labels)

    loss_con = regression_contrastive_loss(features, labels, surrogate.detach(), temperature=0.1)
    # loss_con = F.l1_loss(centroids[centroids_label], surrogate[centroids_label].detach())
    loss_uni = uniformity_loss(surrogate, 100, 0.1)
    loss_smo = smooth_loss(surrogate)

    return loss_reg, loss_con, loss_uni, loss_smo

if __name__ == '__main__':
    # test
    features = torch.randn(20, 5).float()
    labels = torch.randint(0, 100, (20,)).float()
    preds = nn.Parameter(torch.randint(0, 100, (20,)).float())
    seq2seq = nn.Linear(5, 5)
    loss_reg, loss_con, loss_uni, loss_smo = dfr(features, preds, labels, label_range=range(100), seq2seq=seq2seq)
    
    print(loss_reg, loss_con, loss_uni, loss_smo)