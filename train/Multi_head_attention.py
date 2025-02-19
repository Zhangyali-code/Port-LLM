import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, dim=768, embed_dim=512, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # make sure that "embed_dim" can be divisible by "num_heads"
        assert self.head_dim * num_heads == embed_dim, "Embedding dimension must be divisible by number of heads"

        # Initializing the weight matrix
        self.q_linear = nn.Linear(dim, embed_dim)
        self.k_linear = nn.Linear(dim, embed_dim)
        self.v_linear = nn.Linear(dim, embed_dim)

        self.out_linear = nn.Linear(embed_dim, dim)

    def attention(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        n_batches = query.size(0)

        # 1) Do the Linear transformation of Q、K、V
        query = self.q_linear(query)
        key = self.k_linear(key)
        value = self.v_linear(value)

        # 2) make reshape in the last dimension
        query = query.view(n_batches, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(n_batches, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(n_batches, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 3) applying the attention mechanism
        x, attn = self.attention(query, key, value, mask=mask)

        # 4) concating all the heads and do the last linear transformation
        x = x.transpose(1, 2).contiguous().view(n_batches, -1, self.embed_dim)

        return self.out_linear(x)


# # Example：create a MultiHeadAttention module
# embed_dim = 512
# num_heads = 8
# dim = 768
# multi_head_attention = MultiHeadAttention(dim)
#
# # the shape of the tensor: (batch_size, sequence_length, embed_dim)
# query_tensor = torch.rand(32, 10, dim)
# key_tensor = torch.rand(32, 10, dim)
# value_tensor = torch.rand(32, 10, dim)
#
# output = multi_head_attention(query_tensor, key_tensor, value_tensor)
# print(output.shape)  # the shape of the output data: (batch_size, sequence_length, embed_dim)