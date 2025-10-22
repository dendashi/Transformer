import torch



def create_padding_mask(seq, pad_token):
    """创建padding mask（布尔类型，2D）"""
    return (seq != pad_token)  # 返回布尔型mask，形状为 (batch_size, seq_len)

def make_pad_kpm(seq, pad_id):     # seq: (B, L) 的 token ids
    return seq.eq(pad_id)          # True=pad（被屏蔽）


def make_causal_mask(L, device):
    return torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)
