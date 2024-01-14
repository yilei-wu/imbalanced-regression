"""
Experiments on linear/non-linear operators learning tasks, results as mean +- standard variance over 10 runs.
"""
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from OrdinalEntropy import ordinal_entropy
import scipy.io as scio
# from models import MLP
import time
from dfr import dfr_simple, generate_gaussian_vectors, get_centroid
from collections import defaultdict
from scipy.stats import gmean
import random 

def seed_everything(seed=2024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def DIR(X_train1, p):
    def updated_sampling_probability(x):
        if x < 0.1 or x > 0.9:
            return 0.1
        elif 0.1 <= x < 0.2 or 0.8 <= x < 0.9:
            return 0.3
        elif 0.2 <= x < 0.4 or 0.6 <= x < 0.8:
            return 0.6
        elif 0.4 <= x < 0.6:
            return 1
        else:
            return 0

    updated_probabilities = np.array([updated_sampling_probability(x) for x in X_train1])
    updated_probabilities /= updated_probabilities.sum()
    sample_size = int(len(X_train1) * p)
    sampled_indices = np.random.choice(np.arange(X_train1.shape[0]), size=sample_size, replace=False, p=updated_probabilities)
    imbalanced_subset = X_train1[sampled_indices]
    return imbalanced_subset, sampled_indices

def shot_metrics(preds, labels, many_shot_ind, median_shot_ind, low_shot_ind):
    shot_dict = defaultdict(dict)
    shot_dict['all']['mse'] = np.mean(np.abs(preds - labels) ** 2)
    shot_dict['all']['l1'] = np.mean(np.abs(preds - labels))
    shot_dict['all']['gmean'] = gmean(np.abs(preds - labels), axis=None).astype(float)
    shot_dict['many']['mse'] =  np.mean(np.abs(preds[many_shot_ind] - labels[many_shot_ind]) ** 2)
    shot_dict['many']['l1'] = np.mean(np.abs(preds[many_shot_ind] - labels[many_shot_ind]))
    shot_dict['many']['gmean'] = gmean(np.abs(preds[many_shot_ind] - labels[many_shot_ind]), axis=None).astype(float)
    shot_dict['median']['mse'] = np.mean(np.abs(preds[median_shot_ind] - labels[median_shot_ind]) ** 2)
    shot_dict['median']['l1'] = np.mean(np.abs(preds[median_shot_ind] - labels[median_shot_ind]))
    shot_dict['median']['gmean'] = gmean(np.abs(preds[median_shot_ind] - labels[median_shot_ind]), axis=None).astype(float)
    shot_dict['low']['mse'] = np.mean(np.abs(preds[low_shot_ind] - labels[low_shot_ind]) ** 2)
    shot_dict['low']['l1'] = np.mean(np.abs(preds[low_shot_ind] - labels[low_shot_ind]))
    shot_dict['low']['gmean'] = gmean(np.abs(preds[low_shot_ind] - labels[low_shot_ind]), axis=None).astype(float)

    return shot_dict

class MLP(nn.Module):

    def __init__(self, m=100, dim_x=1):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(m + dim_x, 100),
                                     nn.ReLU(),
                                     nn.Linear(100, 100),
                                     nn.ReLU())
        self.regression_layer = nn.Linear(100, 1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.encoder(x)
        pred = self.regression_layer(features)
        return pred, features

def main(Linear=True, oe=True, dir=False, dfr=False, print_performance=False, kwargs=None):
    if Linear:
        m = 100
        lr = 1e-3
        epochs = 20000
        dataset_train = "train.npz"
        dataset_test = "test.npz"
        Lambda_d = 1e-3
        description = 'linear'
        w1, w2, w3 = kwargs['w1'], kwargs['w2'], kwargs['w3']
    else:
        m = 240
        lr = 1e-3
        epochs = 50000
        dataset_train = "train_sde.npz"
        dataset_test = "test_sde.npz"
        Lambda_d = 1e-3
        description = 'nonlinear'
        w1, w2, w3 = kwargs['w1'], kwargs['w2'], kwargs['w3']
    model = MLP(m).cuda()

    d = np.load(dataset_train)
    X_train, y_train = (d["X_train0"], d["X_train1"]), d["y_train"]
    if dir:
        X_train1, indices = DIR(X_train[1], 0.6)
        X_train0 = X_train[0][indices]
        X_train = (X_train0, X_train1)
        y_train = y_train[indices]

    d = np.load(dataset_test)
    X_test, y_test = (d["X_test0"], d["X_test1"]), d["y_test"]
    X_train = np.hstack(X_train)
    X_test = np.hstack(X_test)
    X_train = Variable(torch.from_numpy(X_train), requires_grad=True).float().cuda()
    y_train = Variable(torch.from_numpy(y_train), requires_grad=True).float().cuda()
    X_test = Variable(torch.from_numpy(X_test), requires_grad=True).float().cuda()
    y_test = Variable(torch.from_numpy(y_test), requires_grad=True).float().cuda()

    mse_loss = nn.MSELoss().cuda()
    points = generate_gaussian_vectors(1000, m).cuda()
    label_range = torch.tensor(np.linspace(-1.8, 1.8, m)).cuda()  
    surrogate = torch.rand(len(label_range), 100).cuda()

    l_train = []
    l_test = []

    for times in range(1):   # run 10 times
        begin = time.time()
        model.init_weights()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        _mse_train = 9999
        _mse_test = 9999
        for epoch in range(epochs):
            X_train_small = X_train
            y_train_small = y_train

            model.train()
            optimizer.zero_grad()
            pred, feature = model(X_train_small)
            loss = mse_loss(pred, y_train_small)

            if oe:
                loss_oe = ordinal_entropy(feature, y_train_small) * Lambda_d
            else:
                loss_oe = loss * 0
            
            ind = torch.argmin(torch.abs(y_train_small.detach().cpu() - label_range.cpu().unsqueeze(0)), dim=1)
            y_train_small = label_range[ind.long()]
            running_mean, unique_labels = get_centroid(feature.detach(), y_train_small)
            for j, i in enumerate(unique_labels):
                # the index of the label i in the label_range
                ind = torch.argmin(torch.abs(i - label_range)).long()
                surrogate[ind] = running_mean[j]                    
        
            if dfr:
                loss_reg, loss_con, loss_uni, loss_smo = dfr_simple(feature, y_train_small, points.cuda(), label_range, surrogate, temperature=kwargs['temp'], use_weight=False)
            else:
                loss_reg, loss_con, loss_uni, loss_smo = loss * 0, loss * 0, loss * 0, loss * 0

            loss += w1 * loss_con + w2 * loss_uni + w3 * loss_smo


            loss_all = loss + loss_oe
            loss_all.backward()

            optimizer.step()
            if epoch % 1000 ==0:
                model.eval()
                pred, feature = model(X_test)
                loss_test = mse_loss(pred, y_test)
                # if print_performance:
                #     print('{0}, Epoch: [{1}]\t'
                #         'Loss_train: [{loss:.2e}]\t'
                #         'Loss_test: [{loss_test:.2e}]\t'
                #         'Loss_con: [{loss_con:.2e}]\t'
                #         'Loss_uni: [{loss_uni:.2e}]\t'
                #         'Loss_smo: [{loss_smo:.2e}]\t'.format(description, epoch, loss=loss.data, loss_test=loss_test.data, loss_con=loss_con.data, loss_uni=loss_uni.data, loss_smo=loss_smo.data))

                if loss_test < _mse_test:
                    _mse_test = loss_test
                    _mse_train = loss
                    # if print_performance:
                    #     print('best model, Loss_test: [{loss_test:.2e}]'.format(loss_test=_mse_test.data))

        l_test.append(_mse_test.cpu().detach().numpy())
        l_train.append(_mse_train.cpu().detach().numpy())
        end = time.time()
        print(end-begin)

    l_train = np.array(l_train)
    l_test = np.array(l_test)
    train_dict = {}
    train_dict['train_mse'] = l_train
    train_dict['test_mse'] = l_test
    if Linear:
        path = './Linear.mat'
    else:
        path = './nonlinear.mat'
    scio.savemat(path, train_dict)

    preds = model(X_test)[0].detach().cpu().numpy()
    labels = y_test.detach().cpu().numpy()

    x_test1 = np.array(d["X_test1"], dtype=float)
    many_shot_ind = np.where((x_test1 > 0.2) & (x_test1 < 0.8))[0]
    median_shot_ind = np.where(((x_test1 > 0.1) & (x_test1 < 0.2)) | ((x_test1 > 0.8) & (x_test1 < 0.9)))[0]
    low_shot_ind = np.where((x_test1 < 0.1) | (x_test1 > 0.9))[0]

    if print_performance:
        shot_dict = shot_metrics(preds, labels, many_shot_ind, median_shot_ind, low_shot_ind)
        print(f" * All: MSE {shot_dict['all']['mse']:.5f}\t" f"L1 {shot_dict['all']['l1']:.5f}\tG-Mean {shot_dict['all']['gmean']:.5f}")
        print(f" * Many: MSE {shot_dict['many']['mse']:.5f}\t"f"L1 {shot_dict['many']['l1']:.5f}\tG-Mean {shot_dict['many']['gmean']:.5f}")
        print(f" * Median: MSE {shot_dict['median']['mse']:.5f}\t" f"L1 {shot_dict['median']['l1']:.5f}\tG-Mean {shot_dict['median']['gmean']:.5f}")
        print(f" * Low: MSE {shot_dict['low']['mse']:.5f}\t" f"L1 {shot_dict['low']['l1']:.3f}\tG-Mean {shot_dict['low']['gmean']:.5f}")

    return np.mean(l_test), np.std(l_test)

if __name__ == "__main__":
    Linear = True  # choose the Linear/nonlinear task, i.e. True=Linear, False=nonlinear
    dir = True 
    oe = False  # using the ordinal entropy or not
    dfr = False  # using the DFR or not
    print_performance = True
    kwargs = {'w1': 0.0001, 'w2': 0.0001, 'w3': 0.0001, 'temp': 0.5}

    for i in range(10):
        seed_everything(seed=i)
        main(Linear, oe, dir, dfr, print_performance, kwargs)