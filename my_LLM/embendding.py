import torch
from torch import nn
class TokenEmbeddings(nn.Module):
        def __init__(self, vocab_size, emb_size):
            super().__init__()
            self.vocab_size = vocab_size
            self.emb_size = emb_size
            self.embeddings = nn.Embedding(vocab_size, emb_size)


        def forward(self, inputs):
            embeds = self.embeddings(inputs)
            return embeds