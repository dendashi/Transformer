import torch.nn as nn

# ===== Encoder =====
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_size, num_heads, hidden_dim, dropout=0.1):
        super().__init__()
        # 统一 batch_first=True
        self.self_attention = nn.MultiheadAttention(embed_size, num_heads, batch_first=True, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_size)
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(embed_size)
        self.layer_norm2 = nn.LayerNorm(embed_size)

    # src_key_padding_mask: (B, L_src) bool，True=pad（被屏蔽）
    def forward(self, x, src_key_padding_mask=None):
        attn_output, _ = self.self_attention(
            x, x, x,
            key_padding_mask=src_key_padding_mask       # (B, L_src) bool, True=mask
        )
        x = self.layer_norm1(x + self.dropout(attn_output))

        ffn_output = self.ffn(x)
        return self.layer_norm2(x + self.dropout(ffn_output))


class TransformerEncoder(nn.Module):
    def __init__(self, embed_size, num_heads, hidden_dim, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_size, num_heads, hidden_dim, dropout=dropout)
        for _ in range(num_layers)])

    def forward(self, x, src_key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return x
