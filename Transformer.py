import torch
import torch.nn as nn
import math

#进行输入嵌入
class InputEmbeddings(nn.Module):

    def __init__(self,d_model:int,vocab_size:int) -> None:
        super().__init__()
        self.d_model=d_model
        self.vocab_size=vocab_size
        self.embedding=nn.Embedding(vocab_size,d_model)

    def forward(self,x):
        return self.embedding(x)*math.sqrt(self.d_model) #x: b*s --> b*s*d

class PositionalEncoding(nn.Module):
    def __init__(self,d_model:int,seq_len:int,dropout:float) -> None:
        super().__init__()
        self.d_model=d_model
        self.seq_len=seq_len
        self.dropout=nn.Dropout(dropout)

        #创建形状矩阵
        pe=torch.zeros(seq_len,d_model)
        #形状向量
        position=torch.arange(0,seq_len,dtype=float).unsqueeze(1)  #seq_len * 1 维度
        div_term=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
        #sin
        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)

        pe=pe.unsqueeze(0) # 1 * seq_len * d_model

        self.register_buffer('pe',pe)

    def forward(self,x):
        x = x +self.pe[:,:x.shape[1],:].requires_grad_(False)
        return self.dropout(x)

class LayerNormalization(nn.Module):

    def __init__(self,eps:float=10**-6) -> None:
        super().__init__()
        self.eps=eps
        #nn.Parameter 是 PyTorch 里用来把张量注册为模型 可学习参数 的包装类
        self.alpha=nn.Parameter(torch.ones(1))
        self.bias=nn.Parameter(torch.zeros(1))

    def forward(self,x):
        mean = x.mean(dim=-1,keepdim=True) #mean:b*s*1
        std = x.std(dim=-1,keepdim=True) #std:b*s*1

        return self.alpha*(x-mean)/(std + self.eps) + self.bias #b*s*d

class FeedForwardBlock(nn.Module):

    def __init__(self,d_model:int,d_ff:int,dropout:float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model,d_ff)
        self.linear2 = nn.Linear(d_ff,d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self,x):
        # batch_size * s_l * d_model --> batch_size * s_l * d_ff --> batch_size * s_l * d_model
        return self.linear2(self.drop(torch.relu(self.linear1(x))))

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model%h == 0, "h头数要能被d_model嵌入向量维数整除！！！"
        self.d_k = d_model//h

        # 先切后映射再合并 和 直接映射 一样
        self.w_q = nn.Linear(d_model,d_model)#wq
        self.w_k = nn.Linear(d_model,d_model)#wk
        self.w_v = nn.Linear(d_model,d_model)#wv

        self.w_o = nn.Linear(d_model,d_model)#wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout:nn.Dropout):
        d_k = query.shape[-1] #query: b*h*s*d_k

        # b*h*s*d_k和b*h*s*d_k --> b*h*s*s
        attention_scores = (query @ key.transpose(-1,-2))/math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
            # maked_fill 是PyTorch张量自带方法，作用：按布尔条件，把某些满足条件的位置填成指定值(带下划线_ : 原地修改)
        attention_scores = attention_scores.softmax(dim = -1) #按最后一个维度，也就是对每个qi对应的那一行进行softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # b*h*s*s --> b*h*s*d_k
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q)#b*s*d --> b*s*d
        key = self.w_k(k)#b*s*d --> b*s*d
        value = self.w_v(v)#b*s*d --> b*s*d

        # b*s*d -> b*s*h*d_k --> b*h*s*d_k
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # x：b*h*s*d_k --> b*s*h*d_k --> b*s*d_model
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k) #view中的-1也可以写成x.shape[1] #contiguous就是把transpose之后内存不连续的张量，复制重排成连续，好让view能用。

        return self.w_o(x) #b*s*d_model

class ResidualConnection(nn.Module):

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        #return self.norm(x + self.dropout(sublayer(x))) 为原论文方法顺序
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        #用子模块RC，这里面用dropout，所以不用进行self.dropout=nn.D(dropout)，子模块RC自己用自己的self.dropout
        #为了让MHA和FF都有自己的残差连接，所以不把RC写进__init()__里，而是在内部自己创建两个RC实例
        self.residual_connections = nn.ModuleList(ResidualConnection(dropout) for _ in range(2)) #nn.ModuleList：一个“像列表一样”的容器
        #nn.ModuleList()是给后面生成的实例进行注册，从而保存住参数；如果不加nn.MList仅仅是普通的list那就记不住；现在nn.MList长度是 2 
        #创建两个独立的 RC 实例，因为ATTENTION一次，FEEDFORWARD一次

    def forward(self, x, src_mask):
        #lamda x: （小函数） 因为sublayer只接受一个参数，但是self_attention_block有4个参数；小匿名函数lambda将4个参数转化为1个参数作为sublayer
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None: #这里的nn.MList是layers的类型标注 layers是外部传进来的，不知道啥样儿
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers: # layer = self.layers[i] layers的每一个layer进行一次操作
            x = layer(x, mask) #跑EncoderBlock.forward(x, src_mask)
        return self.norm(x)

class DecoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block:MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(ResidualConnection(dropout) for _ in range(3))

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.proj = nn.Linear(d_model, vocab_size) #输出logits

    def forward(self, x):
        # b*s*d --> b*s*v
        x = torch.log_softmax(self.proj(x), dim=-1)
        #因为词表很大，直接softmax后有的概率会非常小，小概率容易被float精度压成0，为了数值稳定性，log使得log_prob在(-无穷，0]
        #但是注意：要torch.log_softmax(zi)，不能torch.log(torch.softmax(zi))，因为前者是计算zi-log(∑e^zi)，为避免出现log(0)=-inf情况
        return x

#是用前面一堆类的实例组成 Transformer 
class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        #src源序列 tgt目标序列 encoder编码器 decoder解码器 embed嵌入 pos位置嵌入
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        x = self.src_embed(src)
        x = self.src_pos(x)
        x = self.encoder(x, src_mask)
        return x

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        x = self.tgt_embed(tgt)
        x = self.tgt_pos(x)
        x = self.decoder(x, encoder_output, src_mask, tgt_mask)
        return x

    def project(self, x):
        return self.projection_layer(x)

#编写函数：接收超参数 + 用它们搭建模型  创建层(new层)！
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    #调用绿色类InputEmbeddings等等里面的__init__()进行new层 创建实例
    #创建嵌入层
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    #创建位置编码层
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    #创建encoder块(blocks)
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    #创建decoder块(blocks)
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    #创建编码器和解码器
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    #投影层
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    #创建transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    #初始化参数：Xavier
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    return transformer