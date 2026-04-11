import torch
from torch import nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        :param tokens: [seq_len, batch_size]
        :return: [seq_len, batch_size, d_model]
        """
        return self.embedding(tokens)
    

# 实例训练
if __name__ == "__main__":
    # 1.构建一个极简英文词典
    word_to_idx = {
        '<PAD>': 0,
        'i': 1,
        'am': 2,
        'learning': 3,
        'transformer': 4,
        'hello': 5,
        'world': 6
    }
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    vocab_size = len(word_to_idx)
    d_model = 4

    # 2.输入一句英文
    setence = "i am learning transformer"
    print("原始英文句子：", setence)

    # 3.句子 -> 词索引序列
    tokens = [word_to_idx[word] for word in setence.split()]
    print("对应词索引：", tokens)

    # 4.构造模型输入 [seq_len, batch_size]
    seq_len = len(tokens)
    batch_size = 1
    token_tensor = torch.tensor(tokens).unsqueeze(1)  # shape: (4) -> (4, 1)
   
    # 5.初始化embedding并前向传播
    embed = TokenEmbedding(vocab_size, d_model)
    emb_vecs = embed(token_tensor)