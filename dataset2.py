import sentencepiece as spm
from dataset import clean_text

# 加载训练好的分词器
sp = spm.SentencePieceProcessor()
sp.load('spm_model.model')  # 使用你训练好的模型路径

# 假设文件路径为 "train_data.txt"
file_path = "train_data.txt"

# 转换为ID的函数
def encode(text):
    return sp.encode_as_ids(text)

# 添加 <bos>, <eos>, <pad> 和右移标签
start_token = sp.encode_as_ids('<s>')[0]
end_token = sp.encode_as_ids('</s>')[0]
pad_token = sp.encode_as_ids('<pad>')[0]

def right_shift(start_token, end_token, chinese_text, english_text):
    # 目标标签右移，输入的目标句子去掉最后一个token
    chinese_ids = [[start_token] + encode(text) + [end_token] for text in chinese_text]
    english_ids = [[start_token] + encode(text) + [end_token] for text in english_text]
    return chinese_ids, english_ids

# 读取并处理数据
def load_and_process_data(file_path):
    chinese_text = []
    english_text = []

    # 读取train_data.txt文件
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            src, tgt = line.strip().split("\t")  # 假设句子是通过制表符分隔的
            chinese_text.append(src)
            english_text.append(tgt)

    # 清理数据并转换为ID
    chinese_text_cleaned = [clean_text(text) for text in chinese_text]
    english_text_cleaned = [clean_text(text) for text in english_text]
    
    # 右移标签并添加特殊标记
    chinese_ids, english_ids = right_shift(start_token, end_token, chinese_text_cleaned, english_text_cleaned)

    # 找出最大句子长度
    max_len = max(max(len(ids) for ids in chinese_ids), max(len(ids) for ids in english_ids))

    # 填充序列
    chinese_ids_padded = [ids + [pad_token] * (max_len - len(ids)) for ids in chinese_ids]
    english_ids_padded = [ids + [pad_token] * (max_len - len(ids)) for ids in english_ids]

    return chinese_ids_padded, english_ids_padded



# # 输出处理后的数据
# print(f"Chinese IDs: {chinese_ids[:1]}")  # 仅打印第一条数据
# print(f"English IDs: {english_ids[:1]}")  # 仅打印第一条数据
