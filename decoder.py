import torch.nn as nn

# ===== Decoder =====
class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_size, num_heads, hidden_dim, dropout=0.1):
        super().__init__()
        # 统一用 batch_first=True，输入输出均为 (B, L, E)
        self.self_attention  = nn.MultiheadAttention(embed_size, num_heads, batch_first=True, dropout=dropout)
        self.cross_attention = nn.MultiheadAttention(embed_size, num_heads, batch_first=True, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_size)
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(embed_size)
        self.layer_norm2 = nn.LayerNorm(embed_size)
        self.layer_norm3 = nn.LayerNorm(embed_size)

    # 注意：参数命名/含义已明确
    # tgt_key_padding_mask: (B, L_tgt)  bool，True=pad（被屏蔽）
    # src_key_padding_mask: (B, L_src)  bool，True=pad（被屏蔽）
    # causal_mask         : (L_tgt, L_tgt) bool，上三角 True=不允许关注（屏蔽未来）
    def forward(self, tgt, memory, tgt_key_padding_mask=None, src_key_padding_mask=None, causal_mask=None):
        # Masked self-attention（因果mask + tgt padding）
        self_attn_output, _ = self.self_attention(
            tgt, tgt, tgt,
            attn_mask=causal_mask,                       # (L_tgt, L_tgt) bool, True=mask
            key_padding_mask=tgt_key_padding_mask        # (B, L_tgt)     bool, True=mask
        )
        tgt = self.layer_norm1(tgt + self.dropout(self_attn_output))

        # Cross-attention（使用 src padding mask）
        cross_attn_output, _ = self.cross_attention(
            tgt, memory, memory,
            key_padding_mask=src_key_padding_mask        # (B, L_src) bool, True=mask
        )
        tgt = self.layer_norm2(tgt + self.dropout(cross_attn_output))

        # Feed-forward
        ffn_output = self.ffn(tgt)
        return self.layer_norm3(tgt + self.dropout(ffn_output))


class TransformerDecoder(nn.Module):
    def __init__(self, embed_size, num_heads, hidden_dim, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(embed_size, num_heads, hidden_dim, dropout=dropout)
        for _ in range(num_layers)])

    # 与 layer 的 forward 保持一致
    def forward(self, tgt, memory, tgt_key_padding_mask=None, src_key_padding_mask=None, causal_mask=None):
        for layer in self.layers:
            tgt = layer(
                tgt, memory,
                tgt_key_padding_mask=tgt_key_padding_mask,
                src_key_padding_mask=src_key_padding_mask,
                causal_mask=causal_mask
            )
        return tgt
