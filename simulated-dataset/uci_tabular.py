from gaussian import Surrogate, generate_gaussian_vectors, uniformity_loss, smooth_loss, regression_contrastive_loss, compute_cross_entropy, get_centroid, complete_centroid, dfr
from uci_dataset.dataset import get_airfoildataset, get_concretedataset, get_housingdataset, get_abalonedataset, get_bostondataset
from uci_dataset.plot import visualize_model_performance
from lds import apply_label_density_smoothing, weighted_mae_loss
from fds import FDS
from ranksim import batchwise_ranking_regularizer
from scipy.stats import gmean

from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
import torch.nn as nn 
from collections import defaultdict
import torch.nn.functional as F
import torch
import numpy as np
import random
import os


def train(args):
    seed_everything(args.seed)
    
    if os.path.exists(args.path_to_save_figures) == False:
        os.makedirs(args.path_to_save_figures)

    if args.dataset == "airfoil":
        train_dataset, test_dataset = get_airfoildataset(args.fold)
    elif args.dataset == "concrete":
        train_dataset, test_dataset = get_concretedataset(args.fold)
    elif args.dataset == "housing":
        train_dataset, test_dataset = get_housingdataset(args.fold)
    elif args.dataset == "abalone":
        train_dataset, test_dataset = get_abalonedataset(args.fold)
    elif args.dataset == "boston":
        train_dataset, test_dataset = get_bostondataset(args.fold)
    else:
        raise ValueError("Invalid dataset name!")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    input_dim = train_dataset.X.shape[1]
    y_range = range(int(train_dataset.Y.min()), int(train_dataset.Y.max())+1)
    encoder = nn.Sequential(nn.Linear(input_dim, 20), nn.ReLU(), nn.Linear(20, 30), nn.ReLU(), nn.Linear(30, args.feature_dim), nn.ReLU()).to(args.device)
    regressor = nn.Sequential(nn.Linear(args.feature_dim, 1)).to(args.device)
    seq2seq = nn.Sequential(nn.Linear(args.feature_dim, args.feature_dim),nn.ReLU(),nn.Linear(args.feature_dim, args.feature_dim)).cuda()
    surrogate = Surrogate(len(y_range), args.feature_dim, momentum=args.momentum)
    
    if args.fds:
        fds = FDS(args.feature_dim, bucket_num=args.bucket_num, bucket_start=args.bucket_start, start_update=args.start_update, start_smooth=args.start_smooth, kernel='gaussian', ks=args.fds_ks, sigma=args.fds_sigma, momentum=args.fds_mmt).cuda()

    optimizer = AdamW([
        {'params': seq2seq.parameters(), 'lr': args.lr_seq2seq},
        {'params': regressor.parameters(), 'lr': args.lr},
        {'params': encoder.parameters(), 'lr': args.lr}
    ])

    saved_features = defaultdict(list)
    save_labels = defaultdict(list)
    loss_con_history = []
    loss_uni_history = []
    loss_smo_history = []
    loss_reg_history = []

    # train
    for epoch in range(args.epochs):
        features_list = []
        labels_list = []

        for i, (x, y) in enumerate(train_loader):
            x = x.to(args.device)
            y = y.to(args.device)

            optimizer.zero_grad()

            z = encoder(x)
            # fds
            
            if args.fds:
                z_smooth = fds.smooth(z, y.unsqueeze(1), epoch)
                y_hat = regressor(z_smooth)
            else:
                y_hat = regressor(z)

            loss_reg, loss_con, loss_uni, loss_smo = dfr(F.normalize(z, dim=1), y_hat, y, y_range, seq2seq, surrogate, args.temperature, args.n)
            
            if args.baseline:
                # lds
                if args.lds:
                    eff_label_dist = apply_label_density_smoothing(y, args.lds_ks, args.lds_sigma)
                    min_y = torch.min(y)
                    eff_num_per_label = [eff_label_dist[each_y-min_y] for each_y in y]
                    weights = torch.tensor([np.float32(1 / temp) for temp in eff_num_per_label]).to(x.device)
                    loss = weighted_mae_loss(y_hat.squeeze().float(), y.float(), weights=weights.float())
                else:
                    loss = weighted_mae_loss(y_hat.squeeze().float(), y.float(), weights=None)
            else:
                loss = loss_reg + args.loas_w1 * loss_con + args.loas_w2 * loss_uni + args.loas_w3 * loss_smo

            loss_con_history.append(loss_con.item())
            loss_uni_history.append(loss_uni.item())
            loss_smo_history.append(loss_smo.item())
            loss_reg_history.append(loss_reg.item())

            # ranksim_loss
            if args.regularization_weight > 0:
                loss_ranksim = args.regularization_weight * batchwise_ranking_regularizer(z, y.unsqueeze(1), args.interpolation_lambda)
                loss += loss_ranksim

            loss.backward()
            optimizer.step()

            if epoch % args.print_freq == 0 or args.fds:
                features_list.append(z.detach())
                labels_list.append(y.detach())

        
        if args.fds:
            fds.update_running_stats(torch.cat(features_list, dim=0).cuda(), torch.cat(labels_list, dim=0).cuda(), epoch)
            fds.update_last_epoch_stats(epoch)   
             
        if epoch % args.print_freq == 0:
            saved_features[epoch] = torch.cat(features_list, dim=0)
            save_labels[epoch] = torch.cat(labels_list, dim=0)


    #evaluate the regression performance
    y_hat_te = []
    y_te = []
    feat_te = []
    for i, (inputs_te, labels_te) in enumerate(test_loader):
        inputs_te = inputs_te.to(args.device)
        labels_te = labels_te.to(args.device)
        z_te = encoder(inputs_te)
        y_hat_te.append(regressor(z_te).detach().cpu())
        y_te.append(labels_te.cpu())
        feat_te.append(z_te.detach().cpu())
    y_hat_te = torch.cat(y_hat_te, dim=0).squeeze()
    y_te = torch.cat(y_te, dim=0)
    mse_te = F.mse_loss(y_hat_te, y_te).item()
    mae_te = F.l1_loss(y_hat_te, y_te).item()
        
    print(f"Test Set MSE: {mse_te}, MAE {mae_te}") 
    shot_metrics(y_hat_te, y_te, save_labels[0].cpu().numpy())

    # all train labels 
    labels = list(save_labels.values())[0].clone().detach().cpu()
    feat_te = torch.cat(feat_te, dim=0)

    if args.visualize:
        visualize_model_performance(saved_features, save_labels, feat_te, labels_te, labels, y_hat_te, args.path_to_save_figures)
   
    return mse_te

def seed_everything(seed):
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

    # Ensure that the operations are deterministic on GPU (if using CUDA)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.autograd.set_detect_anomaly(True)

def shot_metrics(preds, labels, train_labels):
    # many shot is the class with top one third class occurence
    # medium shot is the class with middle one third class occurence
    # few shot is the class with bottom one third class occurence
    from collections import Counter
    class_occurence = Counter(train_labels)
    class_occurence = sorted(class_occurence.items(), key=lambda x: x[1], reverse=True)
    # many_shot_thr for the cut-off of top one third class occurence
    many_shot_thr = class_occurence[len(np.unique(train_labels))//3][1] 
    low_shot_thr = class_occurence[-len(np.unique(train_labels))//3][1]

    train_labels = np.array(train_labels).astype(int)

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError(f'Type ({type(preds)}) of predictions not supported')

    train_class_count, test_class_count = [], []
    mse_per_class, l1_per_class, l1_all_per_class = [], [], []
    for l in np.unique(labels):
        train_class_count.append(len(train_labels[train_labels == l]))
        test_class_count.append(len(labels[labels == l]))
        mse_per_class.append(np.sum((preds[labels == l] - labels[labels == l]) ** 2))
        l1_per_class.append(np.sum(np.abs(preds[labels == l] - labels[labels == l])))
        l1_all_per_class.append(np.abs(preds[labels == l] - labels[labels == l]))

    many_shot_mse, median_shot_mse, low_shot_mse = [], [], []
    many_shot_l1, median_shot_l1, low_shot_l1 = [], [], []
    many_shot_gmean, median_shot_gmean, low_shot_gmean = [], [], []
    many_shot_cnt, median_shot_cnt, low_shot_cnt = [], [], []

    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot_thr:
            many_shot_mse.append(mse_per_class[i])
            many_shot_l1.append(l1_per_class[i])
            many_shot_gmean += list(l1_all_per_class[i])
            many_shot_cnt.append(test_class_count[i])
        elif train_class_count[i] < low_shot_thr:
            low_shot_mse.append(mse_per_class[i])
            low_shot_l1.append(l1_per_class[i])
            low_shot_gmean += list(l1_all_per_class[i])
            low_shot_cnt.append(test_class_count[i])
        else:
            median_shot_mse.append(mse_per_class[i])
            median_shot_l1.append(l1_per_class[i])
            median_shot_gmean += list(l1_all_per_class[i])
            median_shot_cnt.append(test_class_count[i])

    shot_dict = defaultdict(dict)
    shot_dict['many']['mse'] = np.sum(many_shot_mse) / np.sum(many_shot_cnt)
    shot_dict['many']['l1'] = np.sum(many_shot_l1) / np.sum(many_shot_cnt)
    shot_dict['many']['gmean'] = gmean(np.hstack(many_shot_gmean), axis=None).astype(float)
    shot_dict['median']['mse'] = np.sum(median_shot_mse) / np.sum(median_shot_cnt)
    shot_dict['median']['l1'] = np.sum(median_shot_l1) / np.sum(median_shot_cnt)
    shot_dict['median']['gmean'] = gmean(np.hstack(median_shot_gmean), axis=None).astype(float)
    shot_dict['low']['mse'] = np.sum(low_shot_mse) / np.sum(low_shot_cnt)
    shot_dict['low']['l1'] = np.sum(low_shot_l1) / np.sum(low_shot_cnt)
    shot_dict['low']['gmean'] = gmean(np.hstack(low_shot_gmean), axis=None).astype(float)

    print(f" * Many: MSE {shot_dict['many']['mse']:.3f}\t"
            f"L1 {shot_dict['many']['l1']:.3f}\tG-Mean {shot_dict['many']['gmean']:.3f}")
    print(f" * Median: MSE {shot_dict['median']['mse']:.3f}\t"
            f"L1 {shot_dict['median']['l1']:.3f}\tG-Mean {shot_dict['median']['gmean']:.3f}")
    print(f" * Low: MSE {shot_dict['low']['mse']:.3f}\t"
            f"L1 {shot_dict['low']['l1']:.3f}\tG-Mean {shot_dict['low']['gmean']:.3f}")

    return shot_dict

if __name__ == "__main__":
    import argparse
    from datetime import datetime
    current_date = datetime.now()
    formatted_date = current_date.strftime("%B-%d")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="airfoil", help="Dataset name")
    parser.add_argument("--feature_dim", type=int, default=10, help="Feature dimension")
    parser.add_argument("--fold", type=int, default=1, help="Fold number")
    parser.add_argument("--print_freq", type=int, default=200, help="Print frequency")
    parser.add_argument("--baseline", action="store_true", default=False, help="Whether to run baseline")
    parser.add_argument("--seq2seq", type=str, default="mlp", choices=["mlp", "lstm", "vit"], help="seq2seq model")

    # searched hyper-parameter for configuration
    parser.add_argument("--epochs", type=int, default=1601, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--lr_seq2seq", type=float, default=1e-4, help="Learning rate for seq2seq")

    # LDS
    parser.add_argument('--lds', action='store_true', default=False, help='whether to enable LDS')
    parser.add_argument('--lds_ks', type=int, default=5, help='LDS kernel size: should be odd number')
    parser.add_argument('--lds_sigma', type=float, default=1, help='LDS gaussian/laplace kernel sigma')
    # FDS
    parser.add_argument('--fds', action='store_true', default=False, help='whether to enable FDS')
    parser.add_argument('--fds_ks', type=int, default=5, help='FDS kernel size: should be odd number')
    parser.add_argument('--fds_sigma', type=float, default=3, help='FDS gaussian/laplace kernel sigma')
    parser.add_argument('--start_update', type=int, default=0, help='which epoch to start FDS updating')
    parser.add_argument('--start_smooth', type=int, default=1, help='which epoch to start using FDS to smooth features')
    parser.add_argument('--bucket_num', type=int, default=20, help='maximum bucket considered for FDS')
    parser.add_argument('--bucket_start', type=int, default=3, choices=[0, 3], help='minimum(starting) bucket for FDS, 0 for IMDBWIKI, 3 for AgeDB')
    parser.add_argument('--fds_mmt', type=float, default=0.9, help='FDS momentum')
    # RankSim, Batchwise Ranking Regularizer
    parser.add_argument('--regularization_weight', type=float, default=0, help='weight of the regularization term')
    parser.add_argument('--interpolation_lambda', type=float, default=1.0, help='interpolation strength')

    # hyper-parameter for our method
    parser.add_argument("--momentum", type=float, default=0.90, help="momentum for surrogate")
    parser.add_argument("--temperature", type=float, default=0.1, help="temperature for surrogate")
    parser.add_argument("--loas_w1", type=float, default=0.1, help="weight for contrastive loss")
    parser.add_argument("--loas_w2", type=float, default=0.5, help="weight for uniformity loss")
    parser.add_argument("--loas_w3", type=float, default=0.5, help="weight for uniformity loss")
    parser.add_argument("--n", type=int, default=10, help="number of uniformity points")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--visualize", action="store_true", default=False, help="Whether to visualize")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    combination = f"{args.dataset}-{args.fold}-{args.baseline}-{args.seed}"
    args.path_to_save_figures = f"/mnt/isilon/CSC4/HelenZhouLab/HZLHD1/Data4/Members/yileiwu/fragmented-regression/simulated-dataset/{formatted_date}/{combination}/"
    seed_everything(args.seed)    
    surrogate = train(args)
