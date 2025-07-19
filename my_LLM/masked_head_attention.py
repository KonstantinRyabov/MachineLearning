import torch
from torch import nn
import torch.nn.functional as F
import math
class HeadAttention(nn.Module):
        def __init__(self, emb_size, head_size, max_seq_len):
            super().__init__()
            self.emb_size = emb_size
            self.head_size = head_size
            self.max_seq_len = max_seq_len
            
            self.wk = nn.Linear(emb_size, head_size)
            self.wq = nn.Linear(emb_size, head_size)
            self.wv = nn.Linear(emb_size, head_size)
            
            a = torch.ones(max_seq_len,max_seq_len)
            self.mask = torch.tril(a)


        def forward(self, x):
            batch_size, seq_len, emb_size = x.size()
            k = self.wk(x)
            q = self.wq(x)
            v = self.wv(x)
            
            att = q @ k.transpose(-2, -1) / math.sqrt(self.head_size)
            
            att = att.masked_fill(self.mask[:seq_len,:seq_len] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)

            y = att @ v 
            return y