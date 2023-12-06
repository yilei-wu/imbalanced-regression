# Copyright (c) 2023-present, Royal Bank of Canada.
# Copyright (c) 2021-present, Yuzhe Yang
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
########################################################################################
# Code is based on the LDS and FDS (https://arxiv.org/pdf/2102.09554.pdf) implementation
# from https://github.com/YyzHarry/imbalanced-regression/tree/main/imdb-wiki-dir 
# by Yuzhe Yang et al.
########################################################################################
import torch
import torch.nn.functional as F


def weighted_mse_loss(inputs, targets, weights=None):
    loss = (inputs - targets) ** 2
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_l1_loss(inputs, targets, weights=None):
    loss = F.l1_loss(inputs, targets, reduction='none')
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_focal_mse_loss(inputs, targets, activate='sigmoid', beta=.2, gamma=1, weights=None):
    loss = (inputs - targets) ** 2
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_focal_l1_loss(inputs, targets, activate='sigmoid', beta=.2, gamma=1, weights=None):
    loss = F.l1_loss(inputs, targets, reduction='none')
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_huber_loss(inputs, targets, beta=1., weights=None):
    l1_loss = torch.abs(inputs - targets)
    cond = l1_loss < beta
    loss = torch.where(cond, 0.5 * l1_loss ** 2 / beta, l1_loss - 0.5 * beta)
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss

# ConR loss function
def ConR(features,targets,preds,w=1,weights =1,t=0.07,e=0.01):
    


    q = torch.nn.functional.normalize(features, dim=1)
    k = torch.nn.functional.normalize(features, dim=1)

    l_k = targets.flatten()[None,:]
    l_q = targets

    p_k = preds.flatten()[None,:]
    p_q = preds
    

    l_dist= torch.abs(l_q - l_k)
    p_dist= torch.abs(p_q - p_k)

    
    pos_i = l_dist.le(w)
    neg_i = ((~ (l_dist.le(w)))*(p_dist.le(w)))

    for i in range(pos_i.shape[0]):
        pos_i[i][i] = 0
    
    prod = torch.einsum("nc,kc->nk", [q, k])/t
    pos = prod * pos_i
    neg = prod * neg_i
    
    pushing_w = weights*torch.exp(l_dist*e)
    neg_exp_dot=(pushing_w*(torch.exp(neg))*neg_i).sum(1)

    # For each query sample, if there is no negative pair, zero-out the loss.
    no_neg_flag = (neg_i).sum(1).bool()

    # Loss = sum over all samples in the batch (sum over (positive dot product/(negative dot product+positive dot product)))
    denom=pos_i.sum(1)    

    loss = ((-torch.log(torch.div(torch.exp(pos),(torch.exp(pos).sum(1) + neg_exp_dot).unsqueeze(-1)))*(pos_i)).sum(1)/denom)
    
    loss = (weights*(loss*no_neg_flag).unsqueeze(-1)).mean() 
    
    
    
    return loss

