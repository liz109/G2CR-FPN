import numpy as np
from math import exp
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from skimage.filters import sobel
from skimage.metrics import structural_similarity


def compute_ESSIM(img1, img2, data_range=1.0):
    # Compute the edges using Sobel filter
    edge1 = sobel(img1)
    edge2 = sobel(img2)

    # Compute SSIM for edge regions
    ssim_edge, _ = structural_similarity(edge1, edge2, full=True, data_range=data_range)

    # Compute SSIM for non-edge regions
    ssim_no_edge, _ = structural_similarity(img1 - edge1, img2 - edge2, full=True, data_range=data_range)

    # Weighted combination of SSIM for edge and non-edge regions
    alpha = 0.5  # Weight for edge regions
    ssim_combined = alpha * ssim_edge + (1 - alpha) * ssim_no_edge

    return ssim_combined, ssim_edge, ssim_no_edge

def tensor2np(arr):
    if torch.is_tensor(arr):
        arr = arr.cpu().detach().numpy()
    return arr

def compute_measure(x, y, pred, data_range):
    original_rmse = compute_RMSE(x, y)
    original_psnr = compute_PSNR(x, y, data_range)
    original_ssim = compute_SSIM(x, y, data_range)
    original_essim = compute_ESSIM(x, y, data_range)
    
    pred_rmse = compute_RMSE(pred, y)
    pred_psnr = compute_PSNR(pred, y, data_range)
    pred_ssim = compute_SSIM(pred, y, data_range)
    pred_essim = compute_ESSIM(pred, y, data_range)

    return (original_rmse, original_psnr, original_ssim, original_essim), \
        (pred_rmse, pred_psnr, pred_ssim, pred_essim)


def compute_MSE(img1, img2):
    return ((img1 - img2) ** 2).mean()


def compute_RMSE(img1, img2):
    if type(img1) == torch.Tensor:
        return torch.sqrt(compute_MSE(img1, img2)).item()
    else:
        return np.sqrt(compute_MSE(img1, img2))


def compute_PSNR(img1, img2, data_range):
    if type(img1) == torch.Tensor:
        mse_ = compute_MSE(img1, img2)
        return 10 * torch.log10((data_range ** 2) / mse_).item()
    else:
        mse_ = compute_MSE(img1, img2)
        return 10 * np.log10((data_range ** 2) / mse_)

def compute_SSIM(img1, img2, data_range):
    if type(img1) == torch.Tensor:
        img1 = img1.cpu().detach().numpy()
    if type(img2) == torch.Tensor:
        img2 = img2.cpu().detach().numpy()
    ssim = structural_similarity(img1, img2, data_range=data_range)
    return ssim

# def compute_SSIM(img1, img2, data_range, window_size=11, channel=1, size_average=True):
#     # referred from https://github.com/Po-Hsun-Su/pytorch-ssim
#     if not torch.is_tensor(img1):
#         img1 = torch.tensor(img1)
#     if not torch.is_tensor(img2):
#         img2 = torch.tensor(img2)

#     if len(img1.shape) == 2:
#         shape_ = img1.shape[-1]
#         img1 = img1.view(1,1,shape_ ,shape_ )
#         img2 = img2.view(1,1,shape_ ,shape_ )
#     window = create_window(window_size, channel)
#     window = window.type_as(img1)

#     mu1 = F.conv2d(img1, window, padding=window_size//2)
#     mu2 = F.conv2d(img2, window, padding=window_size//2)
#     mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
#     mu1_mu2 = mu1*mu2

#     sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2) - mu1_sq
#     sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2) - mu2_sq
#     sigma12 = F.conv2d(img1*img2, window, padding=window_size//2) - mu1_mu2

#     C1, C2 = (0.01*data_range)**2, (0.03*data_range)**2
#     #C1, C2 = 0.01**2, 0.03**2

#     ssim_map = ((2*mu1_mu2+C1)*(2*sigma12+C2)) / ((mu1_sq+mu2_sq+C1)*(sigma1_sq+sigma2_sq+C2))
#     if size_average:
#         return ssim_map.mean().item()
#     else:
#         return ssim_map
#         # return ssim_map.mean(1).mean(1).mean(1).item()


def compute_nps(img1, img2):
    noise = img2 - img1  # residual noise
    fft_noise = np.fft.fft2(noise)
    fft_noise_shifted = np.fft.fftshift(fft_noise)
    power_spectrum = np.abs(fft_noise_shifted) ** 2
    return power_spectrum


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


get_mse = lambda x, y: torch.mean((x - y) ** 2)

    
def get_psnr(x, y):
    if torch.max(x) == 0 or torch.max(y) == 0:
        return torch.zeros(1)
    else:
        x_norm = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
        y_norm = (y - torch.min(y)) / (torch.max(y) - torch.min(y))
        mse = get_mse(x_norm, y_norm)
        psnr = -10. * torch.log(mse) / torch.log(torch.Tensor([10.]).to(x.device))

    return psnr[0]


def get_psnr_3d(arr1, arr2, size_average=True, PIXEL_MAX=1.0):
    """
    :param arr1:
        Format-[NDHW], OriImage [0,1]
    :param arr2:
        Format-[NDHW], ComparedImage [0,1]
    :return:
        Format-None if size_average else [N]
    """
    if torch.is_tensor(arr1):
        arr1 = arr1.cpu().detach().numpy()
    if torch.is_tensor(arr2):
        arr2 = arr2.cpu().detach().numpy()
    arr1 = arr1[np.newaxis, ...]
    arr2 = arr2[np.newaxis, ...]
    arr1 = arr1.astype(np.float64)
    arr2 = arr2.astype(np.float64)
    eps = 1e-10
    se = np.power(arr1 - arr2, 2)
    mse = se.mean(axis=1).mean(axis=1).mean(axis=1)
    zero_mse = np.where(mse == 0)
    mse[zero_mse] = eps
    psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
    # #zero mse, return 100
    psnr[zero_mse] = 100

    if size_average:
        return psnr.mean()
    else:
        return psnr



def get_ssim(arr1, arr2, data_range=1.0):
    if torch.is_tensor(arr1):
        arr1 = arr1.cpu().detach().numpy()
    if torch.is_tensor(arr2):
        arr2 = arr2.cpu().detach().numpy()
    ssim = structural_similarity(arr1, arr2, data_range=data_range)
    return ssim


def get_ssim_3d(arr1, arr2, size_average=True, PIXEL_MAX=1.0, data_range=1.0):
    """
    :param arr1:
        Format-[NDHW], OriImage [0,1]
    :param arr2:
        Format-[NDHW], ComparedImage [0,1]
    :return:
        Format-None if size_average else [N]
    """
    if torch.is_tensor(arr1):
        arr1 = arr1.cpu().detach().numpy()
    if torch.is_tensor(arr2):
        arr2 = arr2.cpu().detach().numpy()
    arr1 = arr1[np.newaxis, ...]
    arr2 = arr2[np.newaxis, ...]
    assert (arr1.ndim == 4) and (arr2.ndim == 4)
    arr1 = arr1.astype(np.float64)
    arr2 = arr2.astype(np.float64)

    N = arr1.shape[0]
    # Depth
    arr1_d = np.transpose(arr1, (0, 2, 3, 1))
    arr2_d = np.transpose(arr2, (0, 2, 3, 1))
    ssim_d = []
    for i in range(N):
        ssim = structural_similarity(arr1_d[i], arr2_d[i], data_range=data_range)
        ssim_d.append(ssim)
    ssim_d = np.asarray(ssim_d, dtype=np.float64)

    # Height
    arr1_h = np.transpose(arr1, (0, 1, 3, 2))
    arr2_h = np.transpose(arr2, (0, 1, 3, 2))
    ssim_h = []
    for i in range(N):
        ssim = structural_similarity(arr1_h[i], arr2_h[i], data_range=data_range)
        ssim_h.append(ssim)
    ssim_h = np.asarray(ssim_h, dtype=np.float64)

    # Width
    # arr1_w = np.transpose(arr1, (0, 1, 2, 3))
    # arr2_w = np.transpose(arr2, (0, 1, 2, 3))
    ssim_w = []
    for i in range(N):
        ssim = structural_similarity(arr1[i], arr2[i], data_range=data_range)
        ssim_w.append(ssim)
    ssim_w = np.asarray(ssim_w, dtype=np.float64)

    ssim_avg = (ssim_d + ssim_h + ssim_w) / 3

    if size_average:
        return ssim_avg.mean()
    else:
        return ssim_avg


# def cast_to_image(tensor, normalize=True):
#     # tensor range: [0, 1]
#     # Conver to PIL Image and then np.array (output shape: (H, W))
#     if torch.is_tensor(tensor):
#         img = tensor.cpu().detach().numpy()
#     else:
#         img = tensor
#     if normalize:
#         img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX)
#     return img[..., np.newaxis]