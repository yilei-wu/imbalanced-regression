import torch
import torch.nn as nn
import torch.nn.functional as F
from loss import uniformity_loss, smooth_loss, disparity_loss

# write a function for get the centroid of the representation
# patchify the label space (to do) 
def get_centroid(features, labels):
    # features: [N, D]
    # labels: [N]
    # return: [C, D]
    unique_labels = torch.unique(labels).long()
    centroids = nn.Parameter(torch.zeros(len(unique_labels), features.size(1)).to(features.device))
    centroids = centroids.clone()  # Add this line before the loop
    for i, label in enumerate(unique_labels):
        centroids[i] = features[labels == label].mean(dim=0)
    return centroids, unique_labels


# complete the centroid to whole label space with empty representation where the label is not presented 
def complete_centroid(centroids, centroids_label, label_range):
    # centroids: [C, D]
    # centroids_lebl: [C]
    # label_range: [L]
    # return: [L, D]
    unique_labels = torch.unique(label_range)
    complete_centroids = torch.zeros(len(unique_labels), centroids.size(1)).to(centroids.device) # L, D    
    for i, label in enumerate(centroids_label):
        complete_centroids[label] = centroids[i]
    complete_centroids = nn.Parameter(complete_centroids)
    return complete_centroids


def dfr(features, preds, labels, label_range, seq2seq):
    # features: [N, D]
    # preds: [N]
    # labels: [N]
    # seq2seq: a seq2seq model
    centroids, centroids_label = get_centroid(features, preds) # lebels or preds
    centroids = complete_centroid(centroids, centroids_label, torch.tensor(label_range)) 
    # plain trasnfomer 
    surrogate = seq2seq(centroids) # [L, D]
    # calcuate the loss
    loss_reg = F.l1_loss(preds, labels)
    loss_con = F.l1_loss(surrogate, centroids)
    loss_uni = uniformity_loss(surrogate, 1000, 0.1)
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