import torch
from torch import nn
from masked_head_attention import HeadAttention 
class MultiHeadAttention(nn.Module):
        def __init__(self, num_heads, emb_size, head_size, max_seq_len, dropout = 0.1):
            super().__init__()
            self.num_heads = num_heads
            self.emb_size = emb_size
            self.head_size = head_size
            self.max_seq_len = max_seq_len
            self.dropout = dropout
            
            self.headatt = nn.ModuleList([HeadAttention(emb_size, head_size, max_seq_len) for i in range(num_heads)])
            self.linear  = nn.Linear(head_size*num_heads, emb_size)
            self.drop = nn.Dropout(dropout)


        def forward(self, x):
            batch_size, seq_len, emb_size = x.size()
            output = torch.cat([head(x) for head in self.headatt], dim=-1) 
            lin_rs = self.linear(output)
            drop_rs = self.drop(lin_rs)
            return drop_rs