import torch
import torch.nn as nn
import torch.nn.functional as F


class SobelConv2d(nn.Module):

    def __init__(self, direction, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=True, requires_grad=True):
        assert direction is not None, 'Must specifiy direction in x or y'
        assert kernel_size % 2 == 1, 'SobelConv2d\'s kernel_size must be odd.'
        assert out_channels % groups == 0, 'SobelConv2d\'s out_channels must be a multiple of groups.'

        super(SobelConv2d, self).__init__()

        self.direction = direction
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # In non-trainable case, it turns into normal Sobel operator with fixed weight and no bias.
        self.bias = bias if requires_grad else False

        if self.bias:
            self.bias = nn.Parameter(torch.zeros(size=(out_channels,), dtype=torch.float32), requires_grad=True)
        else:
            self.bias = None

        # Initialize the Sobel kernal
        self.sobel_weight = nn.Parameter(torch.zeros(
            size=(out_channels, int(in_channels / groups), kernel_size, kernel_size)), requires_grad=False)

        kernel_mid = kernel_size // 2
        if self.direction == 'x':   # horizontal derivative approx. (x)
            self.sobel_weight[:, :, :, 0] = -1
            self.sobel_weight[:, :, kernel_mid, 0] = -2
            self.sobel_weight[:, :, :, -1] = 1
            self.sobel_weight[:, :, kernel_mid, -1] = 2
        elif self.direction == 'y': # vertical derivative aprox. (y)
            self.sobel_weight[:, :, 0, :] = -1
            self.sobel_weight[:, :, 0, kernel_mid] = -2
            self.sobel_weight[:, :, -1, :] = 1
            self.sobel_weight[:, :, -1, kernel_mid] = 2
        else:
            raise Exception("Unrecognized direction")

    
        # Define the trainable sobel factor
        if requires_grad:
            self.sobel_factor = nn.Parameter(torch.ones(size=(out_channels, 1, 1, 1), dtype=torch.float32),
                                             requires_grad=True)
        else:
            self.sobel_factor = nn.Parameter(torch.ones(size=(out_channels, 1, 1, 1), dtype=torch.float32),
                                             requires_grad=False)
            
    def forward(self, x):
        # if torch.cuda.is_available():
        #     self.sobel_factor = self.sobel_factor.cuda()
        #     if isinstance(self.bias, nn.Parameter):
        #         self.bias = self.bias.cuda()

        sobel_weight = self.sobel_weight * self.sobel_factor
        # if torch.cuda.is_available():
        #     sobel_weight = sobel_weight.cuda()

        out = F.conv2d(x, sobel_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        # weights = torch.randn(8, 4, 3, 3)   # (out_channel, in_channel/groups, kH, kW)
        # inputs = torch.randn(1, 4, 5, 5)    # (batch, in_channel, iH, iW)
        # print(F.conv2d(inputs, weights, padding=1).shape)   # (batch, out_channel, H, W)
        return out

# if __name__ == '__main__':
#         img = torch.randn(16, 512, 32, 32)      # [N, C, H, W]
#         B, C, H, W = img.shape
#         in_ch = sobel_ch = C                      # in_channels = out_channels
#         direction = 'x'
#         conv_sobel = SobelConv2d(direction, in_ch, sobel_ch, kernel_size=3, stride=1, padding=1, bias=True)
#         out_0 = conv_sobel(img)                 # [N, C, H, W]
#         print(out_0.shape)

   