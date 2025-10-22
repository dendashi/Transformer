# Transformer
A translator based on Transformer
## 数据下载
  运行 **dataset.py** ，从 huggingface 上下载中英文语料库，并且做一个数据的预处理操作。将数据集存储为 **train_data.txt**。

## 分词器训练
  运行 **spm.py** ，训练分词器并保存为 **spm_model.model** 

## 词处理
  **dataset2.py** 可以将文本数据转换成对应 ID，并且用 pad 填充短句子。

## 模型构建
**transformer.py** 为模型总框架。

**encoder.py** 为编码器。

**decoder.py** 为解码器。

**embeddinglayer.py** 添加位置编码。

**paddingmask.py** 实现掩码操作。

## 训练
  运行 **train.ipynb** ，训练模型。这里我数据集质量不是很好，可以使用更好的中英文对照数据集进行训练。

  训练结果如下图：
  ![output.pdf](output.pdf)
  
  

