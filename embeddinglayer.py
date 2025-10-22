import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * -(math.log(10000.0) / embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 添加batch维度
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].detach()  # 添加位置编码


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, max_len=5000):
        super(TransformerEmbedding, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_size)  # 词向量映射
        self.positional_encoding = PositionalEncoding(embed_size, max_len)  # 位置编码

    def forward(self, x):
        embedded = self.token_embedding(x)  # 将词ID转为词嵌入
        return self.positional_encoding(embedded)  # 添加位置编码
