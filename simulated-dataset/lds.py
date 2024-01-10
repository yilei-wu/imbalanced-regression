import torch
from scipy.ndimage import gaussian_filter1d, convolve1d
import numpy as np

def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian']
    half_ks = (ks - 1) // 2
    base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
    kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    return kernel_window

def apply_label_density_smoothing(labels, kernel_size, sigma):
    """
    Apply Label Density Smoothing for integer labels (as PyTorch tensor) using a Gaussian kernel.

    Args:
    labels (torch.Tensor): Tensor of integer labels.
    kernel_size (int): Size of the Gaussian kernel.
    sigma (float): Standard deviation of the Gaussian kernel.

    Returns:
    torch.Tensor: Smoothed effective label distribution.
    """

    # Find the range of labels
    min_label, max_label = torch.min(labels), torch.max(labels)
    range_of_labels = max_label - min_label + 1

    # Create empirical label distribution
    emp_label_dist = np.zeros(range_of_labels)
    for label in labels:
        emp_label_dist[label - min_label] += 1

    # Pad empirical label distribution
    pad_width = kernel_size // 2
    padded_emp_label_dist = np.pad(emp_label_dist, pad_width, mode='constant')

    # Create Gaussian kernel
    lds_kernel_window = get_lds_kernel_window('gaussian', kernel_size, sigma)

    # Apply convolution for smoothing
    smoothed = convolve1d(padded_emp_label_dist, weights=lds_kernel_window, mode='constant')

    # Remove padding
    eff_label_dist = smoothed[pad_width:-(pad_width) or None]

    # Convert back to PyTorch tensor
    eff_label_dist_tensor = torch.tensor(eff_label_dist, dtype=labels.dtype)

    return eff_label_dist_tensor



def weighted_mae_loss(inputs, targets, weights=None):
    loss = torch.abs(inputs - targets)
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


if __name__ == "__main__":
    labels = torch.tensor([1, 1, 1, 2, 2, 6, 2, 2, 2, 5, 5, 5, 5])
    print("Original labels:", labels)
    smoothed_labels = apply_label_density_smoothing(labels, 3, 1.0)
    print("Smoothed labels:", smoothed_labels)
