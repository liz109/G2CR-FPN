import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from torch.autograd import Variable 


# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)
    


class ViT(nn.Module):
    def __init__(self, *, image_size, channels, num_patch=32, dim, depth, heads, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        """
            num_patch: number of patches for each dim -> num_patches = num_patch**2
            dim: hidden size for each pos embedding size
        """
        image_height, image_width = pair(image_size)    # 512, 512
        patch_height, patch_width = image_height//num_patch, image_width//num_patch    # 16, 16
        assert image_height%num_patch == 0 and image_width%num_patch == 0, 'Image dimensions must be divisible by the num_patches.'

        patch_dim = channels * patch_height * patch_width   # 64*16*16

        self.to_patch_embedding = nn.Sequential(
            # [16, 64, 512, 512] -> [16, 32*32, 16*16*64]
            Rearrange('b c (p1 h) (p2 w) -> b (p1 p2) (h w c)', h = patch_height, w = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        num_patches = num_patch*num_patch
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, num_patches, dropout)

        self.to_latent = nn.Identity()  


    def forward(self, img):
        
        x = self.to_patch_embedding(img)    # [16, 64, 512, 512] -> [16, 1024, 128]
        x += self.pos_embedding             # pos_embedding: [1, 1024, 128]
        x = self.dropout(x)                 # [16, 1024, 128]
        
        x = self.transformer(x)             # [16, 1024, 128]
        x = self.to_latent(x)               # [16, 1024, 128]   
        return x    


def test():
    data = Variable(torch.randn(16, 64, 512, 512))
    b, c, h, w = data.shape

    model = ViT(
        image_size = h,
        channels = c, 
        num_patch = 32,
        
        dim=128,            # pos embedding dim
        depth=12,
        heads=8,
        dim_head = 64, 
        dropout = 0., 
        emb_dropout = 0.
    )

    pred_label = model(data)
    return pred_label

if __name__ == '__main__':
    test()