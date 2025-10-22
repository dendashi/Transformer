import torch
import torch.nn as nn
from embeddinglayer import TransformerEmbedding
from encoder import TransformerEncoder
from decoder import TransformerDecoder
from paddingmask import make_causal_mask

# class Transformer(nn.Module):
#     def __init__(self, vocab_size, embed_size, num_heads, num_layers, hidden_dim):
#         super(Transformer, self).__init__()
#         self.embedding = TransformerEmbedding(vocab_size, embed_size)  # 嵌入层
#         self.encoder = TransformerEncoder(embed_size, num_heads, hidden_dim, num_layers)  # 编码器
#         self.decoder = TransformerDecoder(embed_size, num_heads, hidden_dim, num_layers)  # 解码器
#         self.fc_out = nn.Linear(embed_size, vocab_size)  # 输出层

#     def forward(self, src, tgt, src_mask, tgt_mask):
#         src_emb = self.embedding(src)  # 输入嵌入
#         tgt_emb = self.embedding(tgt)  # 目标嵌入
        
#         memory = self.encoder(src_emb, src_mask)  # 编码器输出
#         output = self.decoder(tgt_emb, memory, tgt_mask, src_mask)  # 解码器输出
        
#         return self.fc_out(output)  # 通过全连接层输出最终的logits




class Transformer(nn.Module):
    """
    约定：
      - 所有注意力层均 batch_first=True
      - 输入张量 src, tgt_inp 形状 (B, L)
      - key_padding_mask 形状 (B, L)，dtype=bool，True=pad（被屏蔽）
      - causal_mask 形状 (L, L)，dtype=bool，True=mask（屏蔽未来）
    """
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, hidden_dim, dropout=0.1):
        super().__init__()
        self.embedding = TransformerEmbedding(vocab_size, embed_size)  # 必须输出 (B, L, E)
        self.encoder = TransformerEncoder(embed_size, num_heads, hidden_dim, num_layers, dropout=dropout)
        self.decoder = TransformerDecoder(embed_size, num_heads, hidden_dim, num_layers, dropout=dropout)
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(
        self,
        src_ids,                  # (B, L_src)
        tgt_inp_ids,              # (B, L_tgt_in)  右移后的 decoder 输入
        src_key_padding_mask=None,# (B, L_src)  bool, True=pad
        tgt_key_padding_mask=None,# (B, L_tgt_in) bool, True=pad
        causal_mask=None          # (L_tgt_in, L_tgt_in) bool, True=mask
    ):
        # 嵌入 (B, L, E)
        src_emb = self.embedding(src_ids)
        tgt_emb = self.embedding(tgt_inp_ids)

        # 若未提供因果 mask，则按当前 batch 长度动态创建
        if causal_mask is None:
            L = tgt_inp_ids.size(1)
            causal_mask = make_causal_mask(L, device=tgt_inp_ids.device)

        # 编码器（使用 src_key_padding_mask）
        memory = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)

        # 解码器（使用 tgt_key_padding_mask + causal_mask + src_key_padding_mask）
        dec_out = self.decoder(
            tgt_emb, memory,
            tgt_key_padding_mask=tgt_key_padding_mask,
            src_key_padding_mask=src_key_padding_mask,
            causal_mask=causal_mask
        )

        # 输出 logits (B, L_tgt_in, V)
        return self.fc_out(dec_out)

