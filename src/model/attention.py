import math
import torch
from torch import nn



class Attention(nn.Module):
    """
    A single attention head.
    This module is used in the MultiHeadAttention module.
    Args:
        embed_size
        head_size: output dim of a head
        num_patches: sequence_length in NLP
    Shape:
        - Input: [batch_size, num_patches, embed_size]
        - Ouput: [batch_size, num_patches, head_size] (same shape as input)
        
    """
    def __init__(self, embed_size, head_size, dropout, bias=True):
        super().__init__()
        self.head_size = head_size
        # Create the query, key, and value projection layers
        self.query = nn.Linear(embed_size, head_size, bias=bias)
        self.key = nn.Linear(embed_size, head_size, bias=bias)
        self.value = nn.Linear(embed_size, head_size, bias=bias)    # guided

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q, k, v):
        query = self.query(q)       
        key = self.key(k)
        value = self.value(v)       # shortcut path for attention ouput
        # Calculate the attention scores
        # softmax(Q*K.T/sqrt(head_size))*V
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        # Calculate the attention output
        attention_output = torch.matmul(attention_probs, value)
        return attention_output


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module.
    This module is used in the TransformerEncoder module.
    """

    def __init__(self, embed_size, num_heads=8, attention_probs_dropout_prob=0., hidden_dropout_prob=0., qkv_bias=True):
        super().__init__()
        assert embed_size % num_heads == 0, 'num_heads division error!'
        head_size = embed_size // num_heads     # head output
        all_head_size = num_heads * head_size
        # Create a list of attention heads
        self.heads = nn.ModuleList([])
        for _ in range(num_heads):
            head = Attention(
                embed_size,
                head_size,
                attention_probs_dropout_prob,
                qkv_bias
            )
            self.heads.append(head)
        # Create a linear layer to project the attention output back to the hidden size
        # In most cases, all_head_size and embed_size are the same
        self.output_projection = nn.Linear(all_head_size, embed_size)
        self.output_dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, q, k, v):   
        # Calculate the attention output for each attention head
        attention_outputs = [head(q, k, v) for head in self.heads]
        # Concatenate the attention outputs from each attention head
        attention_output = torch.cat([attention_output for attention_output in attention_outputs], dim=-1)
        # Project the concatenated attention output back to the hidden size
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)
        # Return the attention output
        return attention_output


class FasterMultiHeadAttention(nn.Module):
    """
    Multi-head attention module with some optimizations.
    All the heads are processed simultaneously with merged query, key, and value projections.
    """

    def __init__(self, config):
        super().__init__()
        self.embed_size = config["embed_size"]
        self.num_heads = config["num_heads"]
        # The attention head size is the hidden size divided by the number of attention heads
        self.head_size = self.embed_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_size
        # Whether or not to use bias in the query, key, and value projection layers
        self.qkv_bias = config["qkv_bias"]
        # Create a linear layer to project the query, key, and value
        self.qkv_projection = nn.Linear(self.embed_size, self.all_head_size * 3, bias=self.qkv_bias)
        self.attn_dropout = nn.Dropout(config["attention_probs_dropout_prob"])
        # Create a linear layer to project the attention output back to the hidden size
        # In most cases, all_head_size and embed_size are the same
        self.output_projection = nn.Linear(self.all_head_size, self.embed_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x, output_attentions=False):
        # Project the query, key, and value
        # (batch_size, sequence_length, embed_size) -> (batch_size, sequence_length, all_head_size * 3)
        qkv = self.qkv_projection(x)
        # Split the projected query, key, and value into query, key, and value
        # (batch_size, sequence_length, all_head_size * 3) -> (batch_size, sequence_length, all_head_size)
        query, key, value = torch.chunk(qkv, 3, dim=-1)
        # Resize the query, key, and value to (batch_size, num_heads, sequence_length, head_size)
        batch_size, sequence_length, _ = query.size()
        query = query.view(batch_size, sequence_length, self.num_heads, self.head_size).transpose(1, 2)
        key = key.view(batch_size, sequence_length, self.num_heads, self.head_size).transpose(1, 2)
        value = value.view(batch_size, sequence_length, self.num_heads, self.head_size).transpose(1, 2)
        # Calculate the attention scores
        # softmax(Q*K.T/sqrt(head_size))*V
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_dropout(attention_probs)
        # Calculate the attention output
        attention_output = torch.matmul(attention_probs, value)
        # Resize the attention output
        # from (batch_size, num_heads, sequence_length, head_size)
        # To (batch_size, sequence_length, all_head_size)
        attention_output = attention_output.transpose(1, 2) \
                                           .contiguous() \
                                           .view(batch_size, sequence_length, self.all_head_size)
        # Project the attention output back to the hidden size
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)
        # Return the attention output and the attention probabilities (optional)
        if not output_attentions:
            return (attention_output, None)
        else:
            return (attention_output, attention_probs)


# class AttentionBlock(nn.Module):
#     """
#     An attention block for decoder (self_attention+guided_attention+MLP)
#     """

#     def __init__(self, embed_size):
#         super().__init__()
#         self.use_faster_attention = False
#         if self.use_faster_attention:
#             self.attention1 = FasterMultiHeadAttention(embed_size)
#             self.attention2 = FasterMultiHeadAttention(embed_size)
#         else:
#             self.attention1 = MultiHeadAttention(embed_size, num_heads=8)
#             self.attention2 = MultiHeadAttention(embed_size, num_heads=8)

#         self.layernorm1 = nn.LayerNorm(embed_size)
#         self.layernorm_v = nn.LayerNorm(embed_size)
#         self.layernorm_q = nn.LayerNorm(embed_size)
#         self.layernorm_k = nn.LayerNorm(embed_size)
#         self.layernorm3 = nn.LayerNorm(embed_size)

#         self.mlp = MLP(embed_size, embed_size)

#     def forward(self, v, q, k):
#         v = self.layernorm1(v)
#         attention_output = self.attention1(v,v,v)
#         v = v + attention_output

#         v = self.layernorm_v(v)
#         q = self.layernorm_q(q)
#         k = self.layernorm_k(k)
#         attention_output = self.attention2(q, k, v)
#         v = v + attention_output

#         mlp_output = self.mlp(self.layernorm3(v))
#         v = v + mlp_output

#         # Return the transformer block's output 
#         return v


# class AttentionBlocks(nn.Module):
#     """
#     The transformer encoder module.
#     """

#     def __init__(self, embed_size, num_blocks=1):
#         super().__init__()
#         # Create a list of transformer blocks
#         self.blocks = nn.ModuleList([])
#         for _ in range(num_blocks):
#             block = AttentionBlock(embed_size)
#             self.blocks.append(block)

#     def forward(self, v, q, k):
#         # Calculate the transformer block's output for each block
#         for block in self.blocks:
#             v = block(v, q, k)
#         return v


# class ViTForClassfication(nn.Module):
#     """
#     The ViT model for classification.
#     """

#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.image_size = config["image_size"]      # 256
#         self.embed_size = config["embed_size"]    # 48
#         self.num_classes = config["num_classes"]    # 10
#         # Create the embedding module
#         self.embedding = Embeddings(config)
#         # Create the transformer encoder module
#         self.encoder = Encoder(config)
#         # Create a linear layer to project the encoder's output to the number of classes
#         self.classifier = nn.Linear(self.embed_size, self.num_classes)
#         # Initialize the weights
#         self.apply(self._init_weights)

#     def forward(self, x, output_attentions=False):
#         # Calculate the embedding output
#         embedding_output = self.embedding(x)
#         # Calculate the encoder's output
#         encoder_output, all_attentions = self.encoder(embedding_output, output_attentions=output_attentions)
#         # Calculate the logits, take the [CLS] token's output as features for classification
#         logits = self.classifier(encoder_output[:, 0, :])
#         # Return the logits and the attention probabilities (optional)
#         if not output_attentions:
#             return (logits, None)
#         else:
#             return (logits, all_attentions)
    
#     def _init_weights(self, module):
#         if isinstance(module, (nn.Linear, nn.Conv2d)):
#             torch.nn.init.normal_(module.weight, mean=0.0, std=self.config["initializer_range"])
#             if module.bias is not None:
#                 torch.nn.init.zeros_(module.bias)
#         elif isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)
#         elif isinstance(module, Embeddings):
#             module.position_embeddings.data = nn.init.trunc_normal_(
#                 module.position_embeddings.data.to(torch.float32),
#                 mean=0.0,
#                 std=self.config["initializer_range"],
#             ).to(module.position_embeddings.dtype)

#             module.cls_token.data = nn.init.trunc_normal_(
#                 module.cls_token.data.to(torch.float32),
#                 mean=0.0,
#                 std=self.config["initializer_range"],
#             ).to(module.cls_token.dtype)