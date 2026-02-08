import math
import torch
from torch import nn

from src.model.xpos import XPOS

class Retention(nn.Module):
    def __init__(self, gamma, embed_size, head_size, double_v_dim=False, bias=True):
        """
        Retention mechanism based on the paper
        "Retentive Network: A Successor to Transformer for Large Language Models"[https://arxiv.org/pdf/2307.08621.pdf]
        """
        super(Retention, self).__init__()
        self.gamma = gamma
        self.embed_size = embed_size
        self.head_size = head_size

        self.v_dim = head_size * 2 if double_v_dim else head_size

        self.query = nn.Linear(embed_size, head_size, bias=bias)
        self.key = nn.Linear(embed_size, head_size, bias=bias)
        self.value = nn.Linear(embed_size, head_size, bias=bias)    # guided

        self.xpos = XPOS(head_size)

    def _get_D(self, sequence_length):
        n = torch.arange(sequence_length).unsqueeze(1)
        m = torch.arange(sequence_length).unsqueeze(0)

        # Broadcast self.gamma ** (n - m) with appropriate masking to set values where n < m to 0
        D = (self.gamma ** (n - m)) * (n >= m).float()  #this results in some NaN when n is much larger than m
        # fill the NaN with 0
        D[D != D] = 0

        return D
    
    def forward(self, q, k, v):
        """
        Parallel (default) representation of the retention mechanism.
        X: (batch_size, sequence_length, embed_size)
        """
        assert q.shape[1] == k.shape[1] and k.shape[1] == v.shape[1], 'Unmatched sequence length.'
        sequence_length = q.shape[1]

        Q = self.query(q)
        K = self.key(k)
        Q = self.xpos(Q)
        K = self.xpos(K, downscale=True)

        D = self._get_D(sequence_length).to(Q.device)
        V = self.value(v)
        ret = (Q @ K.permute(0, 2, 1)) * D.unsqueeze(0)
        
        return ret @ V
        
        
class MultiScaleRetention(nn.Module):
    def __init__(self, embed_size, num_heads, double_v_dim=False):
        """
        Multi-scale retention mechanism based on the paper
        "Retentive Network: A Successor to Transformer for Large Language Models"[https://arxiv.org/pdf/2307.08621.pdf]
        """
        super(MultiScaleRetention, self).__init__()
        self.embed_size = embed_size
        self.v_dim = embed_size * 2 if double_v_dim else embed_size
        self.num_heads = num_heads
        assert embed_size % num_heads == 0, "embed_size must be divisible by num_heads"
        self.head_size = embed_size // num_heads    # 512/8=64
        self.head_v_dim = embed_size * 2 if double_v_dim else embed_size
        
        self.gammas = (1 - torch.exp(torch.linspace(math.log2(1/32), math.log2(1/512), num_heads))).detach().cpu().tolist()

        self.swish = lambda x: x * torch.sigmoid(x)
        self.G = nn.Linear(embed_size, self.v_dim)
        self.O = nn.Linear(embed_size, self.v_dim)
        # self.W_G = nn.Parameter(torch.randn(embed_size, self.v_dim) / embed_size)
        # self.W_O = nn.Parameter(torch.randn(self.v_dim, embed_size) / embed_size)
        self.group_norm = nn.GroupNorm(num_heads, self.v_dim)

        self.retentions = nn.ModuleList([
            Retention(gamma, self.embed_size, self.head_size, double_v_dim) for gamma in self.gammas
        ])

    def forward(self, q, k, v):
        """
        parallel representation of the multi-scale retention mechanism
        """

        # apply each individual retention mechanism to X
        Y = []
        for i in range(self.num_heads):
            Y.append(self.retentions[i](q, k, v))
        
        Y = torch.cat(Y, dim=2)
        Y_shape = Y.shape
        Y = self.group_norm(Y.reshape(-1, self.v_dim)).reshape(Y_shape)

        return self.O((self.swish(self.G(v)) * Y))
    
# class RetentionBlock(nn.Module):
#     """
#     An retention block for decoder (self_retention+guided_retention+MLP)
#     """

#     def __init__(self, embed_size):
#         super().__init__()

#         self.retention1 = MultiScaleRetention(embed_size, num_heads=8)
#         self.retention2 = MultiScaleRetention(embed_size, num_heads=8)

#         self.layernorm1 = nn.LayerNorm(embed_size)
#         self.layernorm_v = nn.LayerNorm(embed_size)
#         self.layernorm_q = nn.LayerNorm(embed_size)
#         self.layernorm_k = nn.LayerNorm(embed_size)
#         self.layernorm3 = nn.LayerNorm(embed_size)

#         self.mlp = MLP(embed_size, embed_size)

#     def forward(self, v, q, k):
#         v = self.layernorm1(v)
#         retention_output = self.retention1(v,v,v)
#         v = v + retention_output

#         v = self.layernorm_v(v)
#         q = self.layernorm_q(q)
#         k = self.layernorm_k(k)
#         attention_output = self.retention2(q, k, v)
#         v = v + attention_output

#         mlp_output = self.mlp(self.layernorm3(v))
#         v = v + mlp_output

#         # Return the transformer block's output 
#         return v