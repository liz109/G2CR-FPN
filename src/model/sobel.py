import numpy as np
import torch
import torch.nn as nn
import cv2

def get_gaussian_kernel(k=3, mu=0, sigma=1, normalize=True):
    # compute 1 dimension gaussian
    gaussian_1D = np.linspace(-1, 1, k)
    # compute a grid distance from center
    x, y = np.meshgrid(gaussian_1D, gaussian_1D)
    distance = (x ** 2 + y ** 2) ** 0.5

    # compute the 2 dimension gaussian
    gaussian_2D = np.exp(-(distance - mu) ** 2 / (2 * sigma ** 2))
    gaussian_2D = gaussian_2D / (2 * np.pi *sigma **2)

    # normalize part (mathematically)
    if normalize:
        gaussian_2D = gaussian_2D / np.sum(gaussian_2D)
    return gaussian_2D


def get_sobel_kernel(k=3):
    # get range
    range = np.linspace(-(k // 2), k // 2, k)
    # compute a grid the numerator and the axis-distances
    x, y = np.meshgrid(range, range)
    sobel_2D_numerator = x
    sobel_2D_denominator = (x ** 2 + y ** 2)
    sobel_2D_denominator[:, k // 2] = 1  # avoid division by zero
    sobel_2D = sobel_2D_numerator / sobel_2D_denominator
    return sobel_2D


class SobelFilter(nn.Module):
    def __init__(self,
                 k_gaussian=3,  # 5
                 mu=0,
                 sigma=1,
                 k_sobel=3):
        super(SobelFilter, self).__init__()
        # gaussian
        gaussian_2D = get_gaussian_kernel(k_gaussian, mu, sigma)    # [3, 3]
        gaussian_filter = nn.Conv2d(in_channels=1,
                                         out_channels=1,
                                         kernel_size=k_gaussian,
                                         padding=k_gaussian // 2,
                                         bias=False)
        gaussian_filter.weight.data = self._to_kernel(gaussian_2D)  # [1, 1, 3, 3]


        # sobel
        sobel_2D = get_sobel_kernel(k_sobel)                    # [3, 3]
        sobel_filter_x = nn.Conv2d(in_channels=1,
                                        out_channels=1,
                                        kernel_size=k_sobel,
                                        padding=k_sobel // 2,
                                        bias=False)
        sobel_filter_x.weight.data = self._to_kernel(sobel_2D)
        sobel_filter_y = nn.Conv2d(in_channels=1,
                                        out_channels=1,
                                        kernel_size=k_sobel,
                                        padding=k_sobel // 2,
                                        bias=False)
        sobel_filter_y.weight.data = self._to_kernel(sobel_2D.T)

        self.gaussian_filter = gaussian_filter
        self.sobel_filter_x = sobel_filter_x
        self.sobel_filter_y = sobel_filter_y
        
    def _to_kernel(self, val):
        # [h, w] -> [1, 1, h, w]
        val = torch.from_numpy(val).to(torch.float)
        return val.unsqueeze(0).unsqueeze(0) 

    def forward(self, img):
        # set the setps tensors
        B, C, H, W = img.shape
        device = img.device
        blurred = torch.zeros((B, C, H, W)).to(device)
        grad_x = torch.zeros((B, C, H, W)).to(device)
        grad_y = torch.zeros((B, C, H, W)).to(device)

        # gaussian

        for c in range(C):
            blurred[:, c:c+1] = self.gaussian_filter(img[:, c:c+1].clone())
            grad_x[:, c:c+1] = self.sobel_filter_x(blurred[:, c:c+1].clone())
            grad_y[:, c:c+1] = self.sobel_filter_y(blurred[:, c:c+1].clone())


        return grad_x, grad_y
    
    
