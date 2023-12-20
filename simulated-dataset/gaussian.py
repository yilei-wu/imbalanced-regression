import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
import numpy as np
import os 
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from scipy.stats import vonmises
from scipy.special import ive

class Surrogate(nn.Module):

    def __init__(self, num_labels, feature_dim, momentum=None):
        super().__init__()
        self.num_labels = num_labels
        self.feature_dim = feature_dim
        self.momentum = momentum
        self.surrogate = torch.zeros(num_labels, feature_dim)

    def reset(self):
        self.surrogate = torch.zeros(self.num_labels, self.feature_dim)
    
    def get(self):
        return self.surrogate.clone().detach()

    def update(self, surrogate):
        # momentum update
        if self.momentum is None:
            self.surrogate = surrogate.clone().detach()
        else:
            self.surrogate = self.momentum * self.surrogate + (1 - self.momentum) * surrogate.clone().detach()

class SequenceTransformer(nn.Module):
    def __init__(self, seq_len, feature_dim, num_layers=6, num_heads=8, dim_feedforward=2048):
        super().__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim

        # Positional Encoding
        self.pos_encoder = nn.Embedding(seq_len, feature_dim)

        # Transformer Encoder
        encoder_layers = TransformerEncoderLayer(d_model=feature_dim, nhead=num_heads, dim_feedforward=dim_feedforward)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

        self.bank = torch.zeros(seq_len, feature_dim)

    def forward(self, src):
        # Generating Positional Indices
        positions = torch.arange(self.seq_len, dtype=torch.long, device=src.device).unsqueeze(0).expand(src.size(0), -1)

        # Adding Positional Encoding
        src = src + self.pos_encoder(positions)

        # Transformer Encoder
        output = self.transformer_encoder(src)

        # momemtum output from the bank
        output = 0.01 * output + 0.99 * self.bank.to(output.device).detach()
        self.bank = output

        return output

class CustomDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

def generate_gaussian_vectors(num_points, dimension):
    """
    Generates random Gaussian vectors, filters out those with norms greater than 1,
    and then normalizes the remaining vectors to have a norm of 1.

    :param num_points: Number of points to generate.
    :param dimension: Dimension of each vector.
    :return: A tensor of normalized Gaussian vectors with norm = 1.
    """
    # Generate random Gaussian vectors
    torch.manual_seed(2024)
    gaussian_vectors = torch.randn(num_points, dimension)

    # Calculate the norms of the vectors
    norms = torch.norm(gaussian_vectors, dim=1, keepdim=True)

    normalized_vectors = gaussian_vectors / norms

    return normalized_vectors

def uniformity_loss(embeddings, num_points, epsilon):
    # Generate the Gaussian vectors (assuming generate_gaussian_vectors is a valid function)
    points = generate_gaussian_vectors(num_points, embeddings.shape[1]).to(embeddings.device)

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
    diff = embeddings[1:] - embeddings[:-1]
    lengths = torch.norm(diff, dim=1)

    # Regularization term (optional)
    regularization = torch.var(lengths)
    length = torch.sum(lengths)

    # Combined loss
    loss = alpha * length + beta * regularization
    return loss

def regression_contrastive_loss(embeddings, labels, surrogates, temperature=0.1):
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
    similarities = torch.matmul(embeddings_norm, surrogates_norm.T) 
    print(torch.max(similarities.view(-1)), torch.min(similarities.view(-1)))    
    similarities /= temperature

    mask = torch.zeros_like(similarities).to(similarities.device)
    mask[torch.arange(N), labels] = 1

    # compute log_prob
    # exp_logits = torch.exp(similarities)
    # log_prob = similarities - torch.log(exp_logits.sum(1, keepdim=True))

    # # compute mean of log-likelihood over positive
    # mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
    # loss = - mean_log_prob_pos.mean()
    p = mask / mask.sum(1, keepdim=True).clamp(min=1.0)

    logits_max, _ = torch.max(similarities, dim=-1, keepdim=True)
    similarities = similarities - logits_max.detach()
    
    loss = compute_cross_entropy(p, similarities)
    return loss

def compute_cross_entropy(p, q):
    q = F.log_softmax(q, dim=-1)
    loss = torch.sum(p * q, dim=-1)
    return - loss.mean()

def get_centroid(features, labels):
    # features: [N, D]
    # labels: [N]
    # return: [C, D]
    unique_labels = torch.unique(labels)
    # sort the unique labels
    unique_labels, _ = torch.sort(unique_labels)

    centroids = torch.zeros(len(unique_labels), features.size(1)).to(features.device)
    centroids = centroids.clone()  # Add this line before the loop
    for i, label in enumerate(unique_labels):
        cluster_features = features[labels == label]
        if cluster_features.size(0) > 0:  # Check if the cluster is not empty
            centroids[i] = cluster_features.mean(dim=0).detach()
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
    # complete_centroids = nn.Parameter(complete_centroids)

    return complete_centroids

def dfr(features, preds, labels, label_range, seq2seq, Surrogate):
    # features: [N, D]
    # preds: [N]
    # labels: [N]
    # seq2seq: a seq2seq model
    # surrogate: [L, D]

    features = F.normalize(features, dim=1)
    centroids, centroids_label = get_centroid(features, labels) # lebels or preds
    centroids = complete_centroid(centroids, centroids_label, torch.tensor(label_range))

    # plain trasnfomer
    # update the surrogate where centroids are not zero
    surrogate = Surrogate.get().to(features.device)
    if torch.sum(centroids) != 0:
        surrogate[centroids_label] = centroids[centroids_label].detach()

    surrogate = seq2seq(nn.Parameter(surrogate.unsqueeze(0)))
    if isinstance(surrogate, tuple):
        surrogate = surrogate[0]
    surrogate = surrogate.squeeze(0) # [L, D]
    Surrogate.update(surrogate.cpu())

    # calcuate the loss
    loss_reg = F.l1_loss(preds, labels)

    loss_con = regression_contrastive_loss(features, labels, surrogate.clone().detach(), temperature=0.1)
    # loss_con = F.l1_loss(centroids[centroids_label], surrogate[centroids_label].detach())
    loss_uni = uniformity_loss(surrogate, 100, 0.1)
    loss_smo = smooth_loss(surrogate)

    return loss_reg, loss_con, loss_uni, loss_smo

def init_features(num_labels, num_samples, input_dim=10, std_dev_feature=10, few_shot_ratio=0.2, scalar=3, 
                  train_dist='normal', df=10, transform_func=lambda x: x.float() + 20 * torch.sin(x.float())):
    # Adjust num_samples to generate equal train and test sets
    adjusted_num_samples = num_samples * 2

    # Sampling labels for the training set based on the specified distribution
    if train_dist == 'chi-square':
        # Sample labels based on a chi-square distribution, then map to label range
        chi_samples = np.random.chisquare(df, num_samples)
        train_labels = np.floor(chi_samples / chi_samples.max() * num_labels).astype(int)
    elif train_dist == 'dfr':
        # uniform sampel with range missing, no traing sample from 10-15, 30-35
        train_labels = np.random.choice(np.arange(0, 10).tolist() + np.arange(20, 30).tolist() + np.arange(40, 50).tolist(), num_samples)
    else:
        # Normal distribution with imbalanced label occurrence
        mean = num_labels / 2
        std_dev = num_labels / 4
        train_labels = np.random.normal(mean, std_dev, num_samples)
        train_labels = np.clip(train_labels, 0, num_labels - 1).astype(int)

    train_labels = torch.tensor(train_labels).long()
    train_labels.clamp_(0, num_labels - 1)

    # Generating balanced label samples for testing
    test_labels = torch.tensor(np.repeat(np.arange(num_labels), num_samples / num_labels)).long()

    # Combine train and test labels
    labels = torch.cat((train_labels, test_labels), 0)
    
    # Feature settings
    feature_dim = input_dim

    # Initializing and generating features tensor
    features = torch.zeros((adjusted_num_samples, feature_dim))

    # Apply the specified non-linear transformation to label values for feature mean
    for i, label in enumerate(labels):
        # Use label directly if it's already a tensor
        non_linear_mean = transform_func(label).float()
        features[i] = torch.normal(mean=non_linear_mean, std=std_dev_feature, size=(feature_dim,))

    # Splitting into train and test sets
    train_features = features[:num_samples]
    test_features = features[num_samples:]

    return train_features, train_labels, test_features, test_labels

def train(args):
    if os.path.exists(args.path_to_save_figures) == False:
        os.makedirs(args.path_to_save_figures)

    # Initialize dataset and model
    inputs_ori, labels_ori, inputs_te_ori, labels_te_ori = init_features(num_labels=args.num_labels, num_samples=args.num_samples, input_dim=args.input_dim, std_dev_feature=args.std_dev_feature, few_shot_ratio=0.2, df=args.df, scalar=args.scalar, train_dist=args.train_dist, transform_func=args.transform_func)

    inputs = inputs_ori.clone().cuda()
    labels = labels_ori.clone().cuda()
    inputs_te = inputs_te_ori.clone().cuda()
    labels_te = labels_te_ori.clone().cuda()

    dataset = CustomDataset(inputs, labels)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    encoder = nn.Sequential(
        nn.Linear(args.input_dim, args.input_dim), 
        nn.BatchNorm1d(num_features=args.input_dim),
        nn.ReLU(), 
        nn.Linear(args.input_dim, args.input_dim),
        nn.BatchNorm1d(num_features=args.input_dim), 
        nn.ReLU(),
        nn.Linear(args.input_dim, args.feature_dim)
    ).cuda()

    regressor = nn.Linear(args.feature_dim, 1).cuda()
    if args.seq2seq == 'lstm':
        seq2seq = nn.LSTM(input_size=args.feature_dim, hidden_size=args.feature_dim, num_layers=2, batch_first=True).cuda()
    if args.seq2seq == 'vit':
        # vision transformer with positional encoding
        seq2seq = SequenceTransformer(seq_len=args.num_labels, feature_dim=args.feature_dim, num_layers=2, num_heads=2, dim_feedforward=args.feature_dim).cuda()
    else:
        # mlp
        seq2seq = nn.Sequential(
            nn.Linear(args.feature_dim, args.feature_dim),
            nn.ReLU(),
            nn.Linear(args.feature_dim, args.feature_dim)
        ).cuda()

    optimizer = optim.AdamW([
        {'params': seq2seq.parameters(), 'lr': 0.001},
        {'params': regressor.parameters(), 'lr': 0.001},
        {'params': encoder.parameters(), 'lr': 0.001}
    ])

    # Training loop
    saved_features = defaultdict(list)
    save_labels = defaultdict(list)
    loss_con_history = []
    loss_uni_history = []
    loss_smo_history = []
    loss_reg_history = []

    surrogate = Surrogate(args.num_labels, args.feature_dim, momentum=0.9)

    for step in range(args.num_iterations):
        features_list = []
        labels_list = []

        for inputs_batch, labels_batch in dataloader:
            optimizer.zero_grad()
            features = encoder(inputs_batch)
            # features = F.normalize(features, dim=1)
            preds = regressor(features).squeeze()
            loss_reg, loss_con, loss_uni, loss_smo = dfr(F.normalize(features, dim=1), preds, labels_batch, range(args.num_labels), seq2seq, surrogate)
            if args.use_reg:
                total_loss = loss_reg
            if args.use_con and step > args.warmup:
                total_loss += loss_con
                # pass
            if args.use_uni and step > args.warmup:
                total_loss += loss_uni
            if args.use_smo and step > args.warmup:
                total_loss += loss_smo
                # pass
                
            loss_con_history.append(loss_con.item())
            loss_uni_history.append(loss_uni.item())
            loss_smo_history.append(loss_smo.item())
            loss_reg_history.append(loss_reg.item())

            total_loss.backward()
            optimizer.step()

            if step % args.n == 0:
                features_list.append(features.detach())
                labels_list.append(labels_batch.detach())

        if step % args.n == 0:
            saved_features[step] = torch.cat(features_list, dim=0)
            save_labels[step] = torch.cat(labels_list, dim=0)

    # Evaluation of test set and Saving Loss History Plots
    feat_te = encoder(inputs_te)
    preds_te = regressor(feat_te).squeeze(1)
    mae_te = torch.sum(torch.abs(preds_te - labels_te)) / args.num_samples
    mse_te = torch.sum(torch.abs(preds_te - labels_te) * torch.abs(preds_te - labels_te)) / args.num_samples
    print(f"Test Set MSE: {mse_te}, MAE {mae_te}")

    # Visualization using TSNE and Saving the Figure
    num_rows = 2
    num_cols = 5
    if args.visualizaion == 'uni':
        num_rows = 4
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 8))
    else:  
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 6), subplot_kw={'projection': '3d'})
    axes = axes.ravel()

    # replace the last saved_features with the test set
    
    saved_features['test'] = feat_te.detach()
    save_labels['test'] = labels_te.detach()

    for count, (iteration, iter_features) in enumerate(saved_features.items()):
        if args.visualizaion == 'pca':
            pca = PCA(n_components=3)
            iter_features[torch.isnan(iter_features)] = 0
            reduced_features = pca.fit_transform(iter_features.cpu().numpy())

            ax = axes[count]
            scatter = ax.scatter(reduced_features[:, 0], reduced_features[:, 1], reduced_features[:, 2], c=save_labels[iteration].cpu().numpy(), cmap='viridis')
            ax.set_title(f'Iteration {iteration}')

        elif args.visualizaion == 'tsne':
            tsne = TSNE(n_components=3)
            iter_features[torch.isnan(iter_features)] = 0
            reduced_features = tsne.fit_transform(iter_features.cpu().numpy())

            ax = axes[count]
            scatter = ax.scatter(reduced_features[:, 0], reduced_features[:, 1], reduced_features[:, 2], c=save_labels[iteration].cpu().numpy(), cmap='viridis')
            ax.set_title(f'Iteration {iteration}')

        else:
            def plot_vmf_kde_radar(data, ax, kappa=50):
                """
                Plot the von Mises-Fisher KDE as a radar chart.
                data: data points (N x 2)
                kappa: concentration parameter
                ax: axis for plotting
                """

                # Calculate arctan values for data points
                arctan_values = np.arctan2(data[:, 1], data[:, 0])

                # Bin the arctan values
                num_bins = 100
                bins = np.linspace(-np.pi, np.pi, num_bins)
                density, _ = np.histogram(arctan_values, bins=bins)
                # smooth the density using KDE with gaussian kernel while preserving the shape

                density = np.convolve(density, np.array([0.1, 0.3, 0.5, 0.7, 1, 0.7, 0.5, 0.3, 0.1]), 'same')


                # Radar chart plot
                theta = (bins[:-1] + bins[1:]) / 2  # Bin midpoints for theta values
                # set the subplot scale 
                ax.plot(theta, density, 'b-', linewidth=1)
                # fill the region between the line and the x axis
                ax.fill_between(theta, 0, density, color='blue', alpha=0.3)
                ax.set_ylim(0, 100)
                ax.set_aspect(1/60)  # Set aspect ratio of the axes

            if iter_features.size(1) != 2:
                pca = PCA(n_components=2)
                iter_features[torch.isnan(iter_features)] = 0
                reduced_features = pca.fit_transform(iter_features.cpu().numpy())
                # normalize the features numpy array to [-1, 1]
                # Compute the L2 norm for each row
                norms = np.linalg.norm(reduced_features, axis=1)
                norms = norms.reshape(-1, 1)
                reduced_features = reduced_features / norms
            else:
                reduced_features = iter_features.cpu().numpy()

            # Calculate angles for KDE
            angles = np.arctan2(reduced_features[:, 1], reduced_features[:, 0])

            # Create main plot
            ax1 = axes[count]  # This is the main subplot axis
            scatter = ax1.scatter(reduced_features[:, 0], reduced_features[:, 1], 
                                c=save_labels[iteration].cpu().numpy(), cmap='viridis')
            ax1.set_title(f'Iteration {iteration}')
            ax1.set_aspect('equal', 'box')  # Circle plot
            ax1.set_xlim([-1.1, 1.1])
            ax1.set_ylim([-1.1, 1.1])
            ax1.set_xticks([-1, 0, 1])
            ax1.set_yticks([-1, 0, 1])

            # 获取子图的位置和大小
            # bbox = ax1.get_position()
            # width, height = bbox.width, bbox.height
            # left, bottom = bbox.x0 + width / 2 - width / 8, bbox.y0 + height / 2 - height / 8

            # # 设置极坐标图的位置和大小
            # polar_width, polar_height = width / 4, height / 4  # 极坐标图为子图大小的一半

            # # 添加极坐标图
            ax_polar = axes[count+10]
            # ax_polar = fig.add_axes([left, bottom, polar_width, polar_height], polar=True)
            plot_vmf_kde_radar(reduced_features, ax_polar)


    plt.suptitle('Feature Distribution Over Iterations')
    plt.tight_layout()
    plt.savefig(args.path_to_save_figures + "1.png")  # Save the figure
    plt.close()  # Close the figure


    # Create a 2x2 subplot
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # Plotting each loss history in its own subplot
    axs[0, 0].plot(loss_con_history, label='Loss Con')
    axs[0, 0].set_title('Loss Con')
    axs[0, 0].set_xlabel('Iteration')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    axs[0, 1].plot(loss_uni_history, label='Loss Uni')
    axs[0, 1].set_title('Loss Uni')
    axs[0, 1].set_xlabel('Iteration')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    axs[1, 0].plot(loss_smo_history, label='Loss Smo')
    axs[1, 0].set_title('Loss Smo')
    axs[1, 0].set_xlabel('Iteration')
    axs[1, 0].set_ylabel('Loss')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    axs[1, 1].plot(loss_reg_history, label='loss Reg')
    axs[1, 1].set_title(f'Loss Reg, MAE: {mae_te} MSE: {mse_te}')
    axs[1, 1].set_xlabel('Iteration')
    axs[1, 1].set_ylabel('Loss')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    # Adjust layout
    plt.savefig(args.path_to_save_figures + "2.png")
    plt.close()

    # Assuming inputs_te, labels_te, regressor, encoder, labels_tr are defined and the model is trained
    # Calculate MAE for each label in the test set and Save the Figure
    all_labels = torch.unique(torch.cat([labels_te, labels]))
    max_label = all_labels.max().item()
    label_counts_te = torch.zeros(max_label + 1)
    label_counts_train = torch.zeros(max_label + 1)

    label_counts_te[:len(torch.bincount(labels_te))] = torch.bincount(labels_te)
    label_counts_train[:len(torch.bincount(labels))] = torch.bincount(labels)

    mae_per_label = torch.zeros_like(all_labels, dtype=torch.float)
    for i, label in enumerate(all_labels):
        indices = labels_te == label
        if indices.sum() > 0:
            mae_per_label[i] = torch.mean(torch.abs(preds_te[indices] - labels_te[indices]))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.bar(all_labels.detach().cpu().numpy(), mae_per_label.detach().cpu().numpy())
    plt.xlabel('Label')
    plt.ylabel('MAE')
    plt.title('MAE per Label (Test Set)')

    plt.subplot(1, 3, 2)
    plt.bar(np.arange(max_label + 1), label_counts_te.numpy(), alpha=0.5, label='Test')
    plt.bar(np.arange(max_label + 1), label_counts_train.numpy(), alpha=0.5, label='Train')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.title('Label Occurrences')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.scatter(preds_te.detach().cpu().numpy(), labels_te.detach().cpu().numpy())
    plt.xlabel('Pred')
    plt.ylabel('Label')
    plt.title('Prediction vs Label Scatter Plot')
    plt.legend()

    plt.tight_layout()
    plt.savefig(args.path_to_save_figures + "4.png")
    plt.close()

    return mae_te, mse_te

def run_simulation(args):
    batch_size = args.batch_size
    num_iterations = args.num_iterations
    args.n = num_iterations // 9   # Interval for visualization
    func = args.transform_func
    baseline = args.baseline
    std_dev_feature = args.std_dev_feature
    train_dist = args.train_dist
    transform_func_dict = {"5x+3" : lambda x: 5* x.float() + 3 , "0.3x+5": lambda x: 0.3* x.float() + 5, "0.4x^2" : lambda x: 0.4 * (x.float() ** 2), "logx" : lambda x: torch.log(x.float() + 1)}
    func = args.transform_func
    args.transform_func = transform_func_dict[func]
    args.scalar = args.scalar
    args.df = args.df
    args.warmup = args.warmup
    seed = args.seed
    train_dist = args.train_dist

    torch.manual_seed(seed)
    np.random.seed(seed)

    if baseline:
        args.use_reg = True; args.use_con = False; args.use_uni = False; args.use_smo = False
    else:
        args.use_reg = True; args.use_con = True;  args.use_uni = True;  args.use_smo = True
    
    transform_func = transform_func_dict[func]
    combination = f"func-{func}-seed-{seed}-dist-{train_dist}-{baseline}-batch-{batch_size}-std_dev_feature-{std_dev_feature}-{args.seq2seq if not baseline else ''}"
    args.path_to_save_figures = f"/mnt/isilon/CSC4/HelenZhouLab/HZLHD1/Data4/Members/yileiwu/fragmented-regression/simulated-dataset/{args.savepath}/{combination}/"
    mae, mse = train(args)
    with open(f"/mnt/isilon/CSC4/HelenZhouLab/HZLHD1/Data4/Members/yileiwu/fragmented-regression/simulated-dataset/{args.savepath}/result.csv", "a") as f:
        f.write(f"{baseline}, {batch_size}, {args.seq2seq}, {train_dist}, {seed}, {std_dev_feature}, {func}, {mae}, {mse}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run simulations with various configurations")
    parser.add_argument("--input_dim", type=int, default=20, help="Input dimension")
    parser.add_argument("--feature_dim", type=int, default=10, help="Feature dimension")
    parser.add_argument("--num_labels", type=int, default=50, help="Number of labels")
    parser.add_argument("--num_samples", type=int, default=2000, help="Number of samples")
    parser.add_argument("--num_iterations", type=int, default=801, help="Number of iterations")
    parser.add_argument("--n", type=int, default=100, help="Interval for visualization")
    parser.add_argument("--scalar", type=int, default=3, help="Scalar value")
    parser.add_argument("--df", type=int, default=4, help="Degrees of freedom")
    parser.add_argument("--warmup", type=int, default=0, help="Warmup period")
    parser.add_argument("--batch_size", default=1000, type=int, choices=[1000, 500, 200, 100, 50], help="List of batch sizes")
    parser.add_argument("--std_dev_feature", type=float, default=0.1, choices=[0.1, 1.0])
    parser.add_argument("--seed", type=int, default=2024, help="Random seed")
    parser.add_argument("--train_dist", type=str, default="normal", choices=["normal", "chi-square", 'dfr'], help="Distribution of training labels")
    parser.add_argument("--transform_func", type=str, default="5x+3", choices=["5x+3", "0.3x+5", "0.4x^2", "logx"], help="Transformation function")
    parser.add_argument("--baseline", action="store_true", default=False, help="Whether to run baseline")
    parser.add_argument("--seq2seq", type=str, default="mlp", choices=["mlp", "lstm", "vit"], help="seq2seq model")
    parser.add_argument("--visualizaion", type=str, default="pca", choices=["pca", "tsne", 'uni'], help="Visualization method")
    parser.add_argument("--savepath", type=str, required=True, help="save path")
    args = parser.parse_args()
    run_simulation(args)
