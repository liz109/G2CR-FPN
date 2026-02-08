import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable 

from src.model.attention import MultiHeadAttention
from src.model.retention import MultiScaleRetention


class OutConv(nn.Module):
    def __init__(self, in_planes, out_planes, H, W):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
                    nn.LayerNorm([in_planes, H, W]),
                    nn.Conv2d(in_planes, out_planes, kernel_size=1),
                    nn.BatchNorm2d(out_planes),
                    nn.Conv2d(out_planes, out_planes, kernel_size=1)
                )
    def forward(self, x):
        return self.conv(x)
    

class ResNet(nn.Module):
    scale = 2
    def __init__(self, in_planes, out_planes, stride=1):       
        super(ResNet, self).__init__()

        self.triple_conv = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),     # kernel=1
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_planes, self.scale*out_planes, kernel_size=1, stride=1, padding=0, bias=False),   # kernel=1
            nn.BatchNorm2d(self.scale*out_planes),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.scale*out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.scale*out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.scale*out_planes)
            )

    def forward(self, x):
        out = self.triple_conv(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PatchEmbeddings(nn.Module):
    """
    Convert the image into patches and then project them into a vector space.
    Shape:
        - Input: [batch_size, num_channels, image_size, image_size]
        - Ouput: [batch_size, num_patches, embed_size]
        num_patches = (image_size/patch_size)**2
    """

    def __init__(self, image_size, num_channels, patch_size, embed_size):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.projection = nn.Conv2d(num_channels, embed_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.projection(x)      # [batch_size, embed_size, image_size/patch_size, image_size/patch_size]
        x = x.flatten(start_dim=2).transpose(1, 2)  
        return x


class Embeddings(nn.Module):
    """
    Combine the patch embeddings with the position embeddings.
    Args:
        num_patches: number of patches of an image
    Shape:
        - Input: [batch_size, num_channels, image_size, image_size]
        - Ouput: [batch_size, num_patches (L), embed_size (E)] (same shape as input)
    """
        
    def __init__(self, image_size, num_channels, patch_size, embed_size, emb_dropout=0.):
        super().__init__()
        self.patch_embeddings = PatchEmbeddings(image_size, num_channels, patch_size, embed_size)
        self.position_embeddings = \
            nn.Parameter(torch.randn(1, self.patch_embeddings.num_patches, embed_size))
        self.dropout = nn.Dropout(emb_dropout)

    def forward(self, x):
        x = self.patch_embeddings(x)    
        x += self.position_embeddings
        x = self.dropout(x)         
        return x


class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415

    Taken from https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py
    """

    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


class MLP(nn.Module):
    """
    A multi-layer perceptron module.
    """

    def __init__(self, embed_size, intermediate_size, hidden_dropout_prob=0.):
        super().__init__()
        self.dense_1 = nn.Linear(embed_size, intermediate_size)
        self.activation = NewGELUActivation()
        self.dense_2 = nn.Linear(intermediate_size, embed_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return x


class FusionBlock(nn.Module):
    """
    An attention/retention block for decoder (self_attention+guided_attention+MLP)
    """

    def __init__(self, embed_size, fu_mode):
        super().__init__()
        if fu_mode == 'attention':
            self.fusion1 = MultiHeadAttention(embed_size, num_heads=8)
            self.fusion2 = MultiHeadAttention(embed_size, num_heads=8)
        elif fu_mode == 'retention':
            self.fusion1 = MultiScaleRetention(embed_size, num_heads=8)
            self.fusion2 = MultiScaleRetention(embed_size, num_heads=8)
        else:
            raise Exception("Unrecoginized fusion mode.")

        self.layernorm1 = nn.LayerNorm(embed_size)
        self.layernorm_v = nn.LayerNorm(embed_size)
        self.layernorm_q = nn.LayerNorm(embed_size)
        self.layernorm_k = nn.LayerNorm(embed_size)
        self.layernorm3 = nn.LayerNorm(embed_size)

        self.mlp = MLP(embed_size, embed_size)
        
        # add a gated residual
        self.gate = nn.Sequential(nn.Linear(embed_size, embed_size), nn.Sigmoid())
    
    def forward(self, q, k, v):     # new connections in the paper
        # 1) self fusion
        v_ = self.layernorm1(v)
        fusion_output = self.fusion1(v_,v_,v_)
        v = v + fusion_output

        # 2) guided fusion 
        v_ = self.layernorm_v(v)
        q_ = self.layernorm_q(q)
        k_ = self.layernorm_k(k)
        fusion_output = self.fusion2(q_, k_, v_)
        
        if isinstance(self.fusion2, MultiScaleRetention):
            # gate (token-wise, channel-wise)
            g = self.gate(v_)       # shape [B, L, C], sigmoid in [0,1]
            v = v + g * fusion_output
        else:
            v = v + fusion_output   # attention
        # v = v + fusion_output

        # 3) MLP
        v_ = self.layernorm3(v)
        mlp_output = self.mlp(v_)
        v = v + mlp_output
        return v
    

class FusionBlocks(nn.Module):        # NEW VERSION
    def __init__(self, embed_size, fu_mode, num_blocks=1):
        super().__init__()
        self.blocks = nn.ModuleList([FusionBlock(embed_size, fu_mode) for _ in range(num_blocks)])

    def forward(self, q, k, v):
        for block in self.blocks:
            v = block(q, k, v)
        return v    

# class FusionBlocks(nn.Module):
#     """
#     The attention / retention blocks module.
#     """

#     def __init__(self, embed_size, fu_mode, num_blocks=1):
#         super().__init__()
#         # Create a list of transformer blocks
#         self.blocks = nn.ModuleList([])
#         for _ in range(num_blocks):
#             block = FusionBlock(embed_size, fu_mode)
#             self.blocks.append(block)

#     def forward(self, v, q, k):
#         for block in self.blocks:
#             v = block(v, q, k)
#         return v



class Tokenize(nn.Module):
    def __init__(self, in_planes, num_patches=32) -> None:
        super().__init__()
        # non-overlapping: kernel_size = stride and image_size % kenel_size == 0
        self.num_patches = num_patches
        size = in_planes//self.num_patches, in_planes//self.num_patches
        self.unfold = nn.Unfold(kernel_size=size, stride=size)

    def forward(self, x):
        return self.unfold(x)
    

class DeTokenize(nn.Module):
    def __init__(self, output_size, kernel_size, stride) -> None:
        super().__init__()
        self.fold = nn.Fold(output_size=output_size, kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        return self.fold(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2
    original input size: [N,_,H,W] -> output size: [N,_,H,W]
    """

    def __init__(self, in_planes, out_planes, mid_planes=None):
        super().__init__()
        if not mid_planes:
            mid_planes = out_planes
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_planes, mid_planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(mid_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_planes, out_planes)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_planes, out_planes, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_planes, out_planes, in_planes // 2)
        else:
            self.up = nn.ConvTranspose2d(in_planes, in_planes // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_planes, out_planes)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)



# def test():
#     model = MLP(256*3*3, 128*3*3, 1)
#     vars = Variable(torch.randn(32, 256, 3, 3))
#     vars = vars.view(32, -1)

#     output = model(vars)
#     print(output.size())

# if __name__ == '__main__':
#     test()