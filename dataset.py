import os
import re

# 设置网络代理
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"

from datasets import load_dataset
import sentencepiece as spm

# 加载中英文平行语料数据集
dataset = load_dataset('Mxode/Chinese-English-Parallel-Synonym-Corpus-75k')

# 查看数据集的基本信息
print(dataset)

# 清理函数：去除<|prompt|>及其之间的内容
def clean_text(text):
    # 去除 <|prompt|> 标签及其中的内容
    text = re.sub(r'<\|prompt\|>.*?<\|prompt\|>', '', text)
    # 去除多余的空格
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# 提取中文和英文句子
def preprocess_data(dataset):
    # 提取中文和英文句子
    chinese_text = dataset['train']['input']
    english_text = dataset['train']['output']
    # 清理数据
    chinese_text_cleaned = [clean_text(text) for text in chinese_text]
    english_text_cleaned = [clean_text(text) for text in english_text]
    # 合并为单一文本，两个句子通过制表符连接
    data = [f"{zh}\t{en}" for zh, en in zip(chinese_text_cleaned, english_text_cleaned)]
    return data

# 处理数据集
data = preprocess_data(dataset)

# 保存处理后的数据集到文件
with open("train_data.txt", "w", encoding="utf-8") as f:
    for line in data:
        f.write(line + "\n")



