'''FPN in PyTorch.

See the paper "Feature Pyramid Networks for Object Detection" for more details.

https://jonathan-hui.medium.com/understanding-feature-pyramid-networks-for-object-detection-fpn-45b227b9106c
src: https://github.com/kuangliu/pytorch-fpn/blob/master/fpn.py
'''
import sys
ROOT = '/home/lzhou/medical_img'
sys.path.append(ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable 

from src.model.elements import DoubleConv, OutConv, PatchEmbeddings, Embeddings 
from src.model.elements import ResNet, FusionBlocks
from src.model.sobel import SobelFilter
from src.model.sobel_new import SobelConv2d

from src.model import model_utils 

    
class FPN(nn.Module):
    def __init__(self, C, H, W, \
                 num_down_blocks, num_up_blocks, \
                 fu_mode='attention', upsample_mode='bilinear', detector=None, \
                 num_patches_h=32, num_patches_w=32, scale=2, planes=64):  
        super(FPN, self).__init__()
        assert len(num_down_blocks) == len(num_up_blocks), 'Down-Up layers NOT paired'
        assert H%num_patches_h == 0 and W%num_patches_w == 0, 'Patches division error'

        self.n = len(num_down_blocks)
        self.C, self.H, self.W = C, H, W
        self.scale = scale

        self.upsample_mode = upsample_mode
        self.planes = planes 
        
        # Input 
        self.inconv = DoubleConv(C, self.planes)

        # Downsample 
        self.down_layers = nn.ModuleList([])
        in_planes = planes
        for i in range(len(num_down_blocks)):
            self.down_layers.append(self._downsample_layer(in_planes, num_down_blocks[i]))
            in_planes *= scale

        # Upsample 
        self.up_layers = nn.ModuleList([])
        for i in range(len(num_up_blocks)):
            self.up_layers.insert(0, self._upsample_layer(in_planes, mode=self.upsample_mode))
            in_planes = in_planes // scale

        # Fusion by embedding + attention + de-embedding
        c, h, w = planes, H, W
        p = H // num_patches_h      # patch_size

        self.embed_v_layers = nn.ModuleList([])
        self.embed_q_layers = nn.ModuleList([])
        self.embed_k_layers = nn.ModuleList([])
        self.fusion_layers = nn.ModuleList([])
        self.debed_layers = nn.ModuleList([])

        for i in range(len(num_up_blocks)):
            if fu_mode == 'retention':
                embed_v_layer = PatchEmbeddings(image_size=h, num_channels=c, patch_size=p, embed_size=c)
                embed_q_layer = PatchEmbeddings(image_size=h, num_channels=c, patch_size=p, embed_size=c)
                embed_k_layer = PatchEmbeddings(image_size=h, num_channels=c, patch_size=p, embed_size=c)
            else:
                embed_v_layer = Embeddings(image_size=h, num_channels=c, patch_size=p, embed_size=c, emb_dropout=0.)
                embed_q_layer = Embeddings(image_size=h, num_channels=c, patch_size=p, embed_size=c, emb_dropout=0.)
                embed_k_layer = Embeddings(image_size=h, num_channels=c, patch_size=p, embed_size=c, emb_dropout=0.)
                
            if fu_mode:
                fu_layer = FusionBlocks(c, fu_mode=fu_mode, num_blocks=num_up_blocks[i])
            else:     # To be modified
                fu_layer = nn.Identity()
            debed_layer = nn.ConvTranspose2d(c, c, kernel_size=p, stride=p)


            self.embed_v_layers.append(embed_v_layer)
            self.embed_q_layers.append(embed_q_layer)
            self.embed_k_layers.append(embed_k_layer)
            self.fusion_layers.append(fu_layer)
            self.debed_layers.append(debed_layer)
            c *= scale
            h //= scale

        c //= scale
        # Edge Detector
        self.detector_x_layers = nn.ModuleList([])
        self.detector_y_layers = nn.ModuleList([])
        for i in range(len(num_up_blocks)):
            if detector == 'sobel':
                # layer = SobelFilter()
                layer_x = SobelConv2d(direction='x', in_channels=c, out_channels=c, kernel_size=3, stride=1, padding=1, bias=True)
                layer_y = SobelConv2d(direction='y', in_channels=c, out_channels=c, kernel_size=3, stride=1, padding=1, bias=True)
                self.detector_x_layers.append(layer_x)
                self.detector_y_layers.append(layer_y)
                c //= scale


        # Output -> (16,1,512,512)
        self.outconv = OutConv(64, C, H, W)



    def _downsample_layer(self, planes, num_down_blocks, stride=2):
        strides = [stride] + [1]*(num_down_blocks-1)
        layers = []
        in_planes = planes
        for stride in strides:
            layers.append(ResNet(in_planes, planes, stride))
            in_planes = planes * ResNet.scale
        return nn.Sequential(*layers)


    def _upsample_layer(self, planes, mode='bilinear'):
        in_planes = planes
        out_planes = planes // 2
        if mode == 'bilinear':
            up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            conv = DoubleConv(in_planes, out_planes)
        elif mode == 'conv':
            up = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2, stride=2)
            conv = DoubleConv(out_planes, out_planes)
        else:
            raise Exception("Invalid upsample mode")
        layers = [up, conv]
        return nn.Sequential(*layers)


    def forward_feature(self, x, y):
        pass

    def forward(self, x):       # x: [16, 1, 512, 512]
        # Input 
        x = self.inconv(x)     # c1: [16, 64, 512, 512]

        # Downsample
        down_samples = [x]      # [c1, c2, c3, c4, c5]
        for down_layer in self.down_layers:
            x = down_layer(x)
            down_samples.append(x)

        # Upsample -> Fusion -> add 
        fused_sample = down_samples[-1]     # c5
        for i in range(self.n-1, -1, -1):
            prev_sample = self.up_layers[i](fused_sample)
            cur_sample = down_samples[i]    # c4, c3, c2, c1
            cur_sample_x, cur_sample_y = cur_sample, cur_sample # [16, 512, 64, 64]
            if self.detector_x_layers and self.detector_y_layers:
                # grad_x, grad_y = self.detector_layers[i](cur_sample)    # [16, 512, 64, 64]
                # cur_sample_x = cur_sample * grad_x  # [16, 512, 64, 64]
                # cur_sample_y = cur_sample * grad_y  # [16, 512, 64, 64]
                cur_sample_x = self.detector_x_layers[self.n-1-i](cur_sample)    # [16, 512, 64, 64]
                cur_sample_y = self.detector_y_layers[self.n-1-i](cur_sample)    # [16, 512, 64, 64]

            v = self.embed_v_layers[i](prev_sample)
            q = self.embed_q_layers[i](cur_sample_x)
            k = self.embed_k_layers[i](cur_sample_y)
            fused_sample = self.fusion_layers[i](q, k, v)

            c, l, d = fused_sample.size()
            fused_sample = fused_sample.reshape(c, d, int(l**(1/2)), int(l**(1/2)))  
            fused_sample = self.debed_layers[i](fused_sample)
            fused_sample = F.relu(fused_sample + cur_sample)              

 
        # Output 
        output = self.outconv(fused_sample)    # [16, 1, 512, 512]
        return output

    
    
# def FPN101():
#     # return FPN(Bottleneck, [3,4,23,3])
#     return FPN(1, 512, 512, \
#         num_down_blocks=[2,2,2,2], num_up_blocks=[2,2,2,2], \
#         fu_mode='retention', detector='sobel'
#     )


# def test():
#     net = FPN101()
#     total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)   
#     print('model size (grad): {:.3f}MB'.format(total_params / 1024**2))
#     model_utils.model_size(net)     # 1322.043MB 

#     fms = net(Variable(torch.randn(16, 1, 512, 512)))      # [N, C, H, W]
#     print("Output", fms.shape)


# if __name__ == '__main__':
#     test()