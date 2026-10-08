import torch
from torch import nn
import torch.nn as nn
import math
from torchvision import transforms
class selfAttention(nn.Module):
    def __init__(self,dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,Q,K,V,mask=None):
        # Q矩阵维度[batch_size,seq_len,dim_vec]
        d_k = Q.size(-1)#这里取的Q矩阵的最后一维
        scores = torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask==0,float('-inf'))

        attn = self.softmax(scores)
        attn = self.dropout(attn)
        out =torch.matmul(attn,V)
        return out,attn

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,n_heads,dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model({d_model}) 必须能被 n_heads({n_heads}) 整除"
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.W_q = nn.Linear(d_model,d_model)
        self.W_k = nn.Linear(d_model,d_model)
        self.W_v = nn.Linear(d_model,d_model)
        self.fc = nn.Linear(d_model,d_model)

        self.attention = selfAttention(dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self,q,k,v,mask= None):
        batch_size=q.size(0)
        Q = self.W_q(q).view(batch_size,-1,self.n_heads,self.d_k).transpose(1,2)
        #先把q向量最后一维维度（代表的是词向量的维度d_model）分解为num_heads(多头头数)*d_k,调换第2维和第3维，最终Q[batch_size,num_heads,seq_len,d_k]
        K = self.W_k(k).view(batch_size,-1,self.n_heads,self.d_k).transpose(1,2)
        V = self.W_v(v).view(batch_size,-1,self.n_heads,self.d_k).transpose(1,2)

        out,attn = self.attention(Q,K,V,mask)
        out = out.transpose(1,2).contiguous().view(batch_size,-1,self.n_heads*self.d_k)
        out = self.fc(out)
        out = self.dropout(out)
        return self.norm(out+q),attn

class FeedForward(nn.model):
    def __init__(self,d_model,dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model,d_model)
        self.norm = nn.LayerNorm(d_model)



