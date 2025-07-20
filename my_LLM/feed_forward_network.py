from torch import nn
class FeedForward(nn.Module):
        def __init__(self, emb_size, dropout = 0.1):
            super().__init__()
            self.dropout = dropout
            self.linear1  = nn.Linear(emb_size, 4 * emb_size)
            self.re = nn.ReLU()
            self.linear2  = nn.Linear(4 * emb_size, emb_size)
            self.drop = nn.Dropout(dropout)


        def forward(self, x):
            rs_ln1 = self.linear1(x)
            rs_ln1r = self.re(rs_ln1)
            rs_ln2 = self.linear2(rs_ln1r)
            rs = self.drop(rs_ln2)
            return rs