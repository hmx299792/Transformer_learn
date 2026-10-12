import torch
import torch.nn as nn
from torch.utils.data import Dataset

from typing import Any

#把原始的双语句对，处理成 Transformer 训练所需的 编码器输入、解码器输入、标签、注意力掩码，直接可用于 DataLoader 批量化训练
class BilingualDataset(Dataset): #bilingual 双语的

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        #将[sos]等特殊token转化为对应的token id
        self.sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype = torch.int64)#src、tgt的tokenizer都可以(都含有这些特殊token)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype = torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype = torch.int64)
        #torch.Tensor(x_dim,y_dim,...)  vs  torch.tensor(数据本身，类型)

    def __len__(self):
        return len(self.ds)

    #取第 index 条双语样本，拆出源文本和目标文本，再用各自的 tokenizer 把两段文本分别转成整数 id 列表。
    def __getitem__(self, index: Any) -> Any: #任意类型的索引
        src_target_pair = self.ds[index] #每个index有一对--两句话，对应有长度
        src_text = src_target_pair['translation'][self.src_lang] #字典套字典
        tgt_text = src_target_pair['translation'][self.tgt_lang] #transolation键对应的值value为双语文本对；s\t语言键key对应的值value为某种语言的语句

        #把源文本 src_text 某句话 交给源语言分词器（self.tokenizer）；切分成 token，查词表转成整数id（.encode()）；取出整数 id 列表（ids()）。
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids #.encode自带，和编码器Encoder无关

        enc_num_pad_tokens = self.seq_len - len(enc_input_tokens) - 2 # -2 是因为还有[sos]、[eos]特殊token
        dec_num_pad_tokens = self.seq_len - len(dec_input_tokens) - 1 # -1 是因为decoder输入只有[sos]，标签y只有[eos]

        if enc_num_pad_tokens < 0 or dec_num_pad_tokens < 0:
            raise ValueError('句子太长了！')

        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_pad_tokens, dtype=torch.int64)
            ]
        )

        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_pad_tokens, dtype=torch.int64)
            ]
        )

        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_pad_tokens, dtype=torch.int64)
            ]
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,#s_l长的id
            "decoder_input": decoder_input,#s_l长的id
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), #1*1*s；返回布尔张量用于掩码
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), #&：且，用于张量逐个元素运算,1*s*s
            "label": label, #s_l
            "src_text": src_text,
            "tgt_text": tgt_text
        }


def causal_mask(size):
    mask = torch.triu(torch.ones(1,size,size), diagonal=1).type(torch.int)
    return mask == 0 #保留下三角阵