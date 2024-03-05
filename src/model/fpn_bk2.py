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

from src.model.elements import DoubleConv, OutConv
from src.model.elements import ResNet
from src.model.attention import Embeddings, AttentionBlocks


    
class FPN(nn.Module):
    def __init__(self, C, H, W, \
                 num_down_blocks, num_up_blocks, \
                 upsample_mode='bilinear', \
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
        self.down1 = self._downsample_layer(64, num_down_blocks[0])
        self.down2 = self._downsample_layer(128, num_down_blocks[1])
        self.down3 = self._downsample_layer(256, num_down_blocks[2])
        self.down4 = self._downsample_layer(512, num_down_blocks[3])

        # Upsample 
        self.up5 = self._upsample_layer(1024, mode=self.upsample_mode)
        self.up4 = self._upsample_layer(512, mode=self.upsample_mode)
        self.up3 = self._upsample_layer(256, mode=self.upsample_mode)
        self.up2 = self._upsample_layer(128, mode=self.upsample_mode)

        # Fusion by embedding + attention + de-embedding
        c, p = 64, 16
        self.embed2_v = Embeddings(image_size=512, num_channels=c, patch_size=p, embed_size=c, emb_dropout=0.)
        self.embed1 = Embeddings(image_size=512, num_channels=c, patch_size=p, embed_size=c, emb_dropout=0.)
        self.fu1 = AttentionBlocks(c, num_blocks=num_up_blocks[1])
        self.de_embed1 = nn.ConvTranspose2d(c, c, kernel_size=p, stride=p)

        c, p = 128, 8
        self.embed3_v = Embeddings(image_size=256, num_channels=c, patch_size=p, embed_size=c, emb_dropout=0.)
        self.embed2 = Embeddings(image_size=256, num_channels=c, patch_size=p, embed_size=c, emb_dropout=0.)
        self.fu2 = AttentionBlocks(c, num_blocks=num_up_blocks[1])
        self.de_embed2 = nn.ConvTranspose2d(c, c, kernel_size=p, stride=p)

        c, p = 256, 4
        self.embed4_v = Embeddings(image_size=128, num_channels=c, patch_size=p, embed_size=c, emb_dropout=0.)
        self.embed3 = Embeddings(image_size=128, num_channels=c, patch_size=p, embed_size=c, emb_dropout=0.)
        self.fu3 = AttentionBlocks(c, num_blocks=num_up_blocks[1])
        self.de_embed3 = nn.ConvTranspose2d(c, c, kernel_size=p, stride=p)

        c, p = 512, 2
        self.embed5_v = Embeddings(image_size=64, num_channels=c, patch_size=p, embed_size=c, emb_dropout=0.)
        self.embed4 = Embeddings(image_size=64, num_channels=c, patch_size=p, embed_size=c, emb_dropout=0.)
        self.fu4 = AttentionBlocks(c, num_blocks=num_up_blocks[0])
        self.de_embed4 = nn.ConvTranspose2d(c, c, kernel_size=p, stride=p)


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
        c1 = self.inconv(x)     # c1: [16, 64, 512, 512]

        # Downsample
        c2 = self.down1(c1)     # c2: [16, 128, 256, 256]
        c3 = self.down2(c2)     # c3: [16, 256, 128, 128]
        c4 = self.down3(c3)     # c4: [16, 512, 64, 64]
        c5 = self.down4(c4)     # c5: [16, 1024, 32, 32]

        # Upsample -> Fusion -> add 
        m5 = self.up5(c5)               # c5: [16, 512, 64, 64]
        m5_embed = self.embed5_v(m5)    # c5: [16, 1024, 512] 
        c4_embed = self.embed4(c4)      # c5: [16, 1024, 512]
        c4_fu = self.fu4(m5_embed, c4_embed)  # [16, 1024, 512]
        c, l, d = c4_fu.size()
        c4_fu = c4_fu.reshape(c, d, int(l**(1/2)), int(l**(1/2)))   # [16, 512, 32, 32]
        c4_fu = self.de_embed4(c4_fu)
        c4_fu = F.relu(c4_fu + c4)      # [16, 512, 64, 64]

        m4 = self.up4(c4_fu)            # [16, 256, 128, 128]
        m4_embed = self.embed4_v(m4)    # [16, 1024, 256]
        c3_embed = self.embed3(c3)      # [16, 1024, 256]
        c3_fu = self.fu3(m4_embed, c3_embed)  # [16, 1024, 256]
        c, l, d = c3_fu.size()
        c3_fu = c3_fu.reshape(c, d, int(l**(1/2)), int(l**(1/2)))   # [16, 256, 32, 32]
        c3_fu = self.de_embed3(c3_fu)   
        c3_fu = F.relu(c3_fu + c3)      # [16, 256, 128, 128]


        m3 = self.up3(c3_fu)            # [16, 128, 256, 256]
        m3_embed = self.embed3_v(m3)    # [16, 1024, 128]
        c2_embed = self.embed2(c2)      # [16, 1024, 128]
        c2_fu = self.fu2(m3_embed, c2_embed)  
        c, l, d = c2_fu.size()
        c2_fu = c2_fu.reshape(c, d, int(l**(1/2)), int(l**(1/2)))  
        c2_fu = self.de_embed2(c2_fu)
        c2_fu = F.relu(c2_fu + c2)      # [16, 128, 256, 256]

        m2 = self.up2(c2_fu)            # [16, 64, 512, 512]
        m2_embed = self.embed2_v(m2)    # [16, 1024, 64]
        c1_embed = self.embed1(c1)      # [16, 1024, 64]
        c1_fu = self.fu1(m2_embed, c1_embed)  
        c, l, d = c1_fu.size()
        c1_fu = c1_fu.reshape(c, d, int(l**(1/2)), int(l**(1/2)))   #
        c1_fu = self.de_embed1(c1_fu)
        c1_fu = F.relu(c1_fu + c1)      # [16, 64, 512, 512]

 
        # Output 
        output = self.outconv(c1_fu)    # [16, 1, 512, 512]
        return output

    
    
def FPN101():
    # return FPN(Bottleneck, [3,4,23,3])
    return FPN(down_block=ResNet, num_down_blocks=[2,2,2,2], \
                      upsample_mode='bilinear', \
                        up_block=AttentionBlocks, num_up_blocks=[2,2,2,2])


def test():
    net = FPN101()
    fms = net(Variable(torch.randn(16, 1, 512, 512)))      # [N, C, H, W]


if __name__ == '__main__':
    test()