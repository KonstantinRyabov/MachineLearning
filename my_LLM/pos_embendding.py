import torch
from torch import nn
class PositionalEmbeddings(nn.Module):
        def __init__(self, max_seq_len, emb_size):
            super().__init__()
            self.max_seq_len = max_seq_len
            self.emb_size = emb_size
            self.embeddings = nn.Embedding(max_seq_len, emb_size)


        def forward(self, seq_len):
            input_indices = torch.tensor(range(seq_len)) 
            embeds = self.embeddings(input_indices)
            return embeds