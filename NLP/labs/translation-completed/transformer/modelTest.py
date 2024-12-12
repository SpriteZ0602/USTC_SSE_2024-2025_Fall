import os
import re
import math
import pandas as pd
import torch
from torch import nn
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm


# 数据处理组件
class SimpleTokenizer:
    def __init__(self, vocab_file):
        self.vocab = {}
        self.ids_to_tokens = {}

        # 读取词表文件
        with open(vocab_file, 'r', encoding='utf-8') as f:
            for idx, token in enumerate(f.readlines()):
                token = token.strip()
                self.vocab[token] = idx
                self.ids_to_tokens[idx] = token

        self.pad_token = '<pad>'
        self.eos_token = '<eos>'
        self.bos_token = '<bos>'
        self.unk_token = '<unk>'

        self.pad_token_id = self.vocab.get(self.pad_token, len(self.vocab))
        self.eos_token_id = self.vocab.get(self.eos_token, len(self.vocab) + 1)
        self.bos_token_id = self.vocab.get(self.bos_token, len(self.vocab) + 2)
        self.unk_token_id = self.vocab.get(self.unk_token, len(self.vocab) + 3)

        self.vocab[self.pad_token] = self.pad_token_id
        self.vocab[self.eos_token] = self.eos_token_id
        self.vocab[self.bos_token] = self.bos_token_id
        self.vocab[self.unk_token] = self.unk_token_id

        self.ids_to_tokens[self.pad_token_id] = self.pad_token
        self.ids_to_tokens[self.eos_token_id] = self.eos_token
        self.ids_to_tokens[self.bos_token_id] = self.bos_token
        self.ids_to_tokens[self.unk_token_id] = self.unk_token

        self.vocab_size = len(self.vocab)

    def tokenize(self, text):
        return text.split()

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab.get(token, self.vocab.get(self.unk_token)) for token in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self.ids_to_tokens.get(index, self.unk_token) for index in ids]

    def prepare_batch(self, texts, max_length=None):
        batch_token_ids = [self.convert_tokens_to_ids([self.bos_token] + self.tokenize(text) + [self.eos_token]) for
                           text in texts]
        return self.pad_ids(batch_token_ids, max_length)

    def pad_ids(self, batch_ids, max_length=None):
        if max_length is None:
            max_length = max(len(ids) for ids in batch_ids)
        padded_ids = []
        attention_masks = []
        for ids in batch_ids:
            padding_length = max_length - len(ids)
            padded_ids.append(ids + [self.pad_token_id] * padding_length)
            attention_masks.append([1] * len(ids) + [0] * padding_length)
        return padded_ids, attention_masks

    def get_vocab(self):
        return self.vocab


def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    pairs = [line.strip().split('\t') for line in lines]
    return pairs


class TranslationDataset(Dataset):
    def __init__(self, pairs, tokenizer):
        self.pairs = pairs
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def collate_fn(batch, tokenizer):
    src_texts, tgt_texts = zip(*batch)
    src_ids, src_masks = tokenizer.prepare_batch(src_texts)
    tgt_ids, tgt_masks = tokenizer.prepare_batch(tgt_texts)
    return (torch.tensor(src_ids, dtype=torch.long),
            torch.tensor(src_masks, dtype=torch.long),
            torch.tensor(tgt_ids, dtype=torch.long),
            torch.tensor(tgt_masks, dtype=torch.long))


# transformer组件
class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = nn.Linear(ffn_num_outputs, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


class AddNorm(nn.Module):
    """残差连接后进行层规范化"""

    def __init__(self, norm_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(norm_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(10000, torch.arange(0, num_hiddens, 2,
                                                                                                      dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)


# 多头注意力
def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状"""
    # 输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
    # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，
    # num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)

    # 最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    """逆转transpose_qkv函数的操作"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


def sequence_mask(X, attntion_mask, value=0):
    """在序列中屏蔽不相关的项"""
    attntion_mask = attntion_mask.bool()
    X[~attntion_mask] = value
    return X


def masked_softmax(X, attntion_mask):
    """通过在最后一个轴上掩蔽元素来执行softmax操作"""
    if attntion_mask is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if attntion_mask.dim() == 2:
            attntion_mask = torch.repeat_interleave(attntion_mask, shape[1], dim=0)
        else:
            attntion_mask = attntion_mask.view(-1, shape[-1])
        # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
        X = sequence_mask(X.reshape(-1, shape[-1]), attntion_mask, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


class DotProductAttention(nn.Module):
    """缩放点积注意力"""

    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # queries的形状：(batch_size，查询的个数，d)
    # keys的形状：(batch_size，“键－值”对的个数，d)
    # values的形状：(batch_size，“键－值”对的个数，值的维度)
    # attention_weights:(batch_size，“键－值”对的个数)或者(batch_size，查询的个数, “键－值”对的个数)
    def forward(self, queries, keys, values, attntion_mask=None):
        d = queries.shape[-1]
        # 设置transpose_b=True为了交换keys的最后两个维度
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        attention_weights = masked_softmax(scores, attntion_mask)
        return torch.bmm(self.dropout(attention_weights), values)


class MultiHeadAttention(nn.Module):
    """多头注意力"""

    def __init__(self, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        self.W_k = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        self.W_v = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, attntion_mask=None):
        # queries，keys，values的形状:
        # (batch_size，查询或者“键－值”对的个数，num_hiddens)
        # attntion_mask:
        # (batch_size，“键－值”对的个数)或者(batch_size，查询的个数, “键－值”对的个数)
        # 经过变换后，输出的queries，keys，values　的形状:
        # (batch_size*num_heads，查询或者“键－值”对的个数，
        # num_hiddens/num_heads)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if attntion_mask is not None:
            # 在轴0，将第一项复制num_heads次，
            attntion_mask = torch.repeat_interleave(attntion_mask, repeats=self.num_heads, dim=0)

        # output的形状:(batch_size*num_heads，查询的个数，
        # num_hiddens/num_heads)
        output = self.attention(queries, keys, values, attntion_mask)

        # output_concat的形状:(batch_size，查询的个数，num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)



# transformer encoder
class TransformerEncoderBlock(nn.Module):
    """Transformer编码器块"""

    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout,
                 use_bias=False):
        super().__init__()
        self.attention = MultiHeadAttention(num_hiddens, num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(num_hiddens, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(num_hiddens, dropout)

    def forward(self, X, attntion_mask=None):
        Y = self.addnorm1(X, self.attention(X, X, X, attntion_mask))
        return self.addnorm2(Y, self.ffn(Y))


class TransformerEncoder(nn.Module):
    """Transformer编码器块"""

    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens,
                 num_heads, num_blks, dropout, use_bias=False):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module("block" + str(i), TransformerEncoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, use_bias))

    def forward(self, X, attntion_mask=None):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        for i, blk in enumerate(self.blks):
            X = blk(X, attntion_mask)
        return X



# transformer decoder
class TransformerDecoderBlock(nn.Module):
    # 解码器中第i个块
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout, i):
        super().__init__()
        self.i = i
        self.attention1 = MultiHeadAttention(num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(num_hiddens, dropout)
        self.attention2 = MultiHeadAttention(num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(num_hiddens, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(num_hiddens, dropout)

    def forward(self, X, state):
        enc_outputs, enc_attention_mask = state[0], state[1]
        # 训练阶段，输出序列的所有词元都在同一时间处理，
        # 因此state[2][self.i]初始化为None。
        # 预测阶段，输出序列是通过词元一个接着一个解码的，
        # 因此state[2][self.i]包含着直到当前时间步第i个块解码的输出表示
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), dim=1)
        state[2][self.i] = key_values
        if self.training:
            batch_size, num_steps, _ = X.shape
            dec_attention_mask = torch.tril(torch.ones(batch_size, num_steps, num_steps)).to(X.device)
        else:
            dec_attention_mask = None
        # 自注意力
        X2 = self.attention1(X, key_values, key_values, dec_attention_mask)
        Y = self.addnorm1(X, X2)
        # 编码器－解码器注意力。
        # enc_outputs的开头:(batch_size,num_steps,num_hiddens)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_attention_mask)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                 num_blks, dropout):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.num_blks = num_blks
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module("block" + str(i),
                                 TransformerDecoderBlock(num_hiddens, ffn_num_hiddens, num_heads, dropout, i))
        self.dense = nn.Linear(num_hiddens, vocab_size)
        # with torch.no_grad():
        #     self.dense.weight = self.embedding.weight

    def init_state(self, enc_outputs, enc_attention_mask):
        return [enc_outputs, enc_attention_mask, [None] * self.num_blks]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
        return self.dense(X), state


# transformer
class Transformer(nn.Module):
    """编码器-解码器架构的基类"""

    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads, num_blks, dropout=0.2):
        super().__init__()
        self.encoder = TransformerEncoder(vocab_size, num_hiddens, ffn_num_hiddens, num_heads, num_blks, dropout)
        self.decoder = TransformerDecoder(vocab_size, num_hiddens, ffn_num_hiddens, num_heads, num_blks, dropout)

    def forward(self, enc_X, dec_X, enc_attention_mask=None):
        enc_outputs = self.encoder(enc_X, enc_attention_mask)
        dec_state = self.decoder.init_state(enc_outputs, enc_attention_mask)
        return self.decoder(dec_X, dec_state)


# 创建数据集和数据加载器
tokenizer = SimpleTokenizer('../data/ch_en_vocab.txt')


# 序列生成函数
def generate_sequence(model, src_batch, src_mask, tokenizer, max_length=50):
    model.eval()
    enc_outputs = model.encoder(src_batch, src_mask)
    dec_input = torch.tensor([[tokenizer.vocab['<bos>']]]).to(src_batch)
    generated_tokens = []
    dec_state = model.decoder.init_state(enc_outputs, src_mask)
    for _ in range(max_length):
        dec_logits, dec_state = model.decoder(dec_input, dec_state)
        next_token = dec_logits.argmax(dim=-1)[0, -1]
        if next_token.item() == tokenizer.eos_token_id:
            break
        generated_tokens.append(next_token.item())
        dec_input = next_token.view(1, 1)
    return generated_tokens


# 测试模型并计算BLEU分数
def evaluate_bleu(model, data_loader, tokenizer, max_length=50):
    bleu_scores = []
    model.eval()
    with torch.no_grad():
        for src_batch, src_mask, tgt_batch, _ in tqdm(data_loader):
            src_batch, src_mask = src_batch.to(device), src_mask.to(device)
            for ids in range(src_batch.shape[0]):
                src_batch_ = src_batch[ids, None]
                src_mask_ = src_mask[ids, None]
                generated_tokens = generate_sequence(model, src_batch_, src_mask_, tokenizer, max_length)

                pred_tokens = tokenizer.convert_ids_to_tokens(generated_tokens)
                target_tokens = tokenizer.convert_ids_to_tokens(tgt_batch[ids].numpy())

                pred_tokens = [t for t in pred_tokens if t not in [tokenizer.pad_token, tokenizer.unk_token]]
                target_tokens = [t for t in target_tokens if
                                 t not in [tokenizer.pad_token, tokenizer.unk_token, tokenizer.eos_token,
                                           tokenizer.bos_token]]

                bleu_score = sentence_bleu([target_tokens], pred_tokens, smoothing_function=SmoothingFunction().method1)
                bleu_scores.append(bleu_score)

    average_bleu = sum(bleu_scores) / len(bleu_scores)
    print(f'Average BLEU score: {average_bleu:.4f}')

def translate_sentence(model, sentence, tokenizer, max_length=50):
    # 对输入句子进行tokenize和转换为ID
    src_ids, src_mask = tokenizer.prepare_batch([sentence])

    # 转换为tensor
    src_ids = torch.tensor(src_ids, dtype=torch.long).to(device)
    src_mask = torch.tensor(src_mask, dtype=torch.long).to(device)

    # 使用生成序列的函数
    translated_ids = generate_sequence(model, src_ids, src_mask, tokenizer, max_length=max_length)

    # 转换为中文句子
    translated_tokens = tokenizer.convert_ids_to_tokens(translated_ids)
    return ''.join(translated_tokens).replace('<eos>', '')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load("translationModel10.pth")
model.eval()

english_sentence = "how are you"
chinese_translation = translate_sentence(model, english_sentence, tokenizer)
print("英文句子:", english_sentence)
print("中文翻译:", chinese_translation)


