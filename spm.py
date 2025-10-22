import sentencepiece as spm

# 训练分词器
spm.SentencePieceTrainer.train(
    input='train_data.txt',        # 输入文件路径
    model_prefix='spm_model',      # 模型前缀
    vocab_size=32000,              # 词汇表大小
    character_coverage=0.995,      # 覆盖率
    model_type='bpe'               # 使用BPE模型
)

