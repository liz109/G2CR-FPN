'''FPN in PyTorch.

See the paper "Feature Pyramid Networks for Object Detection" for more details.

https://jonathan-hui.medium.com/understanding-feature-pyramid-networks-for-object-detection-fpn-45b227b9106c
src: https://github.com/kuangliu/pytorch-fpn/blob/master/fpn.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable 

# from src.model.element import *
from .elements import * 

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):        # in_channels, out_channels
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
   

    
class FPN(nn.Module):
    # num_blocks (List[int]):  num of Bottle
    def __init__(self, block, num_blocks):  # Bottleneck, [2,2,2,2]
        super(FPN, self).__init__()
        self.in_planes = 64             # in_channels

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Bottom-up layers
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)

        # fusion layers
        self.fusionlayer1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256)
        )
        self.fusionlayer2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256)
        )
        self.fusionlayer3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256)
        )

        # output layer; -> (32,1,512,512)
        self.outputlayer1 = nn.Sequential(
            nn.ConvTranspose2d(256, 256 // 2, kernel_size=2, stride=2),
            DoubleConv(128, 128),
            nn.ConvTranspose2d(128, 128 // 2, kernel_size=2, stride=2),
            DoubleConv(64, 64),
            OutConv(64, 1)
        )
        # self.outputlayer = nn.Sequential(
        #     nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=0),
        #     nn.BatchNorm2d(1),
        #     nn.ReLU()
        # )
        # self.mlp = MLP(256*3*3, 128*3*3, 1)     # dense for regression
        # self.mlp = MLP(in_size=256*3*3, mid_size=128*3*3, out_size=7, dropout_r=0.1) # classifier


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.

        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.

        Returns:
          (Variable) added feature map.

        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.

        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]

        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear') + y
        # return F.upsample(x, size=(H,W), mode='bilinear') + y
        # PAFPN / yolov4: concat op

    def _downsample_fusion(self, x, y):
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), align_corners=False, antialias=True, mode='bilinear') + y
    

    def forward(self, x):
        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        
        # Top-down
        m5 = self.toplayer(c5)
        _c4 = self.latlayer1(c4)
        m4 = self._upsample_add(m5, _c4)
        _c3 = self.latlayer2(c3)
        m3 = self._upsample_add(m4, _c3)
        _c2 = self.latlayer3(c2)
        m2 = self._upsample_add(m3, _c2)

        # Smooth
        p5 = m5
        p4 = self.smooth1(m4)
        p3 = self.smooth2(m3)
        p2 = self.smooth3(m2)

        # # Fusion: bn(conv(p2)) + p3 (self-defined)
        # f3 = self._downsample_fusion(self.fusionlayer1(p2), p3)
        # f4 = self._downsample_fusion(self.fusionlayer2(f3), p4)
        # f5 = self._downsample_fusion(self.fusionlayer3(f4), p5)

        # Output 
        output = self.outputlayer1(p2)
        # print(f'output:{output.shape}')
        # f5 = self.outputlayer(f5)   # [32, 256, 3, 3] 
        # dim = f5.shape[0]
        # f5 = f5.view(dim, -1)        # [32, 2304] / multi [8, 2304]
        # # output = self.mlp(f5)

        # return p2, p3, p4, p5
        return output

    
    
# def FPN101():
#     # return FPN(Bottleneck, [3,4,23,3])
#     return FPN(Bottleneck, [2,2,2,2])


# def test():
#     net = FPN101()
#     fms = net(Variable(torch.randn(32, 1, 512, 512)))      # [N, C, H, W]
#     # for fm in fms:
#     #     print(fm.size())        


# if __name__ == '__main__':
#     test()