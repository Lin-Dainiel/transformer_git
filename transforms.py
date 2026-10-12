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
        return out,attn #返回V向量

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
        out,attn = self.attention(Q,K,V,mask)#经过自注意力机制，加权求和后，得到新的V向量
        out = out.transpose(1,2).contiguous().view(batch_size,-1,self.n_heads*self.d_k)#调换第2维和第3维，out形状为[batch_size,seq_len,d_model]#拼接V向量
        out = self.fc(out)#给每个头的V向量引入一个权重矩阵
        out = self.dropout(out)
        return self.norm(out+q),attn#残差连接,Post-LN
class FeedForward(nn.Module):
    def __init__(self,d_model,d_ff,dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model), #层归一化
        self.feed_forward= nn.Sequential(
            nn.Linear(d_model,d_ff),#全连接层
            nn.ReLU(),#非线性
            nn.Dropout(dropout),#
            nn.Linear(d_ff,d_model) ,
        )
    def forward(self,x):
        out = self.model(x)
        return self.norm(out + x)
 class EncoderLayer(nn.Moduel):
    def __init__(self,d_model,n_heads,d_ff,dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model,n_heads,dropout=0.1),
        self.ffn =  FeedForward(d_model,d_ff,dropout=0.1)
    def forward(self,src,src_mask=None):
        src 输入序列张量，形状（batch_size,seq_len,d_model）
        out,_= self.self_attn(src,src,src,src_mask)
        out = self.ffn(out)
        return out

class DecoderLayer(nn.Module):
    def __init__(self,d_model,n_heads,d_ff,dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model,n_heads,dropout=0.1)
        self.cross_attn = MultiHeadAttention(d_model,n_heads,dropout=0.1)
        self.ffn =  FeedForward(d_model,d_ff,dropout=0.1)
        
    def foward(self,tgt,memory,tgt_mask=None,memory_mask=None):
        out,_self.self_attn(tgt,tgt,tgt,tgt_mask)
        out,_self.cross_attn(out,memory,memory,memory_mask)
        out = self.ffn(out)
        return out
        
        
        
        



