import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split #DataLoader：把 Dataset 变成一批一批（batch）数据、并送给模型训练的工具

from dataset import BilingualDataset, causal_mask
from Transformer import build_transformer
from config import get_config, get_weights_file_path

from datasets import load_dataset #从 Hugging Face 的 datasets 库中导入 load_dataset 函数，用于加载数据集
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from torch.utils.tensorboard import SummaryWriter #可视化

import warnings
from pathlib import Path
from tqdm import tqdm

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device): #greedy贪心：选概率最大的
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    #预先计算编码器输出，用于每个解码器输出token
    encoder_output = model.encode(source, source_mask)
    #初始化只有sos的解码器输入
    decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source).to(device)

        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask) #out：b*s*d

        prob = model.project(out[:,-1])
        #1.这里的project是线性映射，所以和project(out)[:, -1]等价
        #2.out[:,-1]和out[:,-1,:]等价：所有的b，所有的d，但是s的最后一个位置；因为下一个token的预测只看上一个词
        #3.[:,-1]这一步取s的最后一个位置，自动把s这个维度删除，b*s*v变成了b*v
        _, next_word =torch.max(prob, dim=1) #取v词表中概率最大的值以及其索引
        decoder_input = torch.cat([ decoder_input, torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device) ], dim=1)
        #.item()把只含单个元素的张量，转成 Python 的标量（数字）   序列长度加1
        
        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_state, writer, num_examples=2):
    model.eval()
    count = 0
    console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            assert encoder_input.size(0) == 1, "批量大小必须是1！"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch['src_text'][0] #每批的第一条样本
            target_text = batch['tgt_text'][0]
            #把 token id 序列翻译回字符串文本
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy()) #放cpu算快点儿

            print_msg('-'*console_width)
            print_msg(f'源序列: {source_text}')
            print_msg(f'目标序列: {target_text}')
            print_msg(f'预测结果: {model_out_text}')

            if count == num_examples:
                break

def get_all_sentences(ds, lang):
    #遍历数据集，每次取出一条样本item：一对翻译句
    for item in ds:
    #'translation'里找翻译对，[lang]是提取的语言，yield是逐个输出
        yield item['translation'][lang] #'translation'是hugging face作者起的键的名字；2层字典

def get_or_build_tokenizer(config, ds, lang): #config：字典，用来存放配置参数
    #假设 lang = 'en'：'tokenizer_{0}.json'.format('en')  -→  'tokenizer_en.json'
    tokenizer_path = Path(config['tokenizer_file'].format(lang)) #取模板 → 用 lang 填占位符得到该语言的文件名字符串 → 包装成 Path 对象 → 作为该语言分词器的存取路径
    if  not Path.exists(tokenizer_path):
        #unk_token、special_tokens都是特殊名字，不能改
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]')) #WordLevel：闭词表模型
        tokenizer.pre_tokenizer = Whitespace()
        #min_frequency只管普通词，大于等于这个频率的才进入词表；特殊词保送进词表
        trainer = WordLevelTrainer(special_tokens = ['[UNK]', '[PAD]', '[SOS]', '[EOS]'], min_frequency = 2) #制定词表构建规则 超过特定频率的词才能进入分词器
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer) #train_from_iterator：从一段"数据流"（迭代器）上训练分词器，构建词表
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    #从 Hugging Face Hub 下载并加载“英语-意大利语”的书籍翻译数据集，只取训练集(raw：原始未加工数据)
    #config字典里有lang_src和lang_tgt两个键key，对应两个值value
    ds_raw = load_dataset('Helsinki-NLP/opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split='train') #split='train：只选择训练集的划分

    #创建分词器
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src']) #例如config = {'lang_src': 'en','lang_tgt': 'zh'}
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    #90%训练，10%验证
    train_ds_size = int(0.9*len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    #原始数据经过BilingualDataset类处理后，变成了一个个的字典，字典里有编码器输入、解码器输入、标签、掩码等
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    max_len_src = 0
    max_len_tgt = 0

    #遍历整个原始数据集，统计源语言和目标语言中最长句子的 token 数量（可重复）
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]) # Encoding 对象
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']])
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'源语言最长句子长度 {max_len_src}')
    print(f'目标语言最长句子长度 {max_len_tgt}')

    #数据分成很多个大小为b的批次，再输出；被分完批的训练和验证数据集；数据多一个维度b
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'],shuffle=True) #shuffle=True：每个 epoch（每一轮训练）开始时，把样本的顺序随机打乱一遍
    val_dataloader = DataLoader(val_ds,batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt #返回的是已经利用数据集做好了的

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len,config['seq_len'], config['seq_len'], config['d_model'])
    return model 
    #返回的是一个transformer模型，transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    #Transformer里有encode()、decode()、project()

def train_model(config):
    #哪里训练
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备为：{device}')

    #创建一个保存模型权重的文件夹
    Path(config['model_folder']).mkdir(parents=True, exist_ok= True) #Path()：把字符串路径变成一个"路径对象"

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device) #把模型搬到cuda上
    #Tensorboard
    writer = SummaryWriter(config['experiment_name']) #把“这次实验的名字”当成文件夹名，用来存日志

    #创建优化器，用 Adam 算法来更新模型参数
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global global_step
    global_step = 0
    
    #第N轮出问题了在这一轮处重新来（设置个存档点），不必完全从头来过
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'预加载模型：{model_filename}')
        state = torch.load(model_filename) #torch.load和torch.save相反
        initial_epoch = state['epoch'] + 1

        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict']) #恢复优化器状态

    #计算交叉熵损失：1.忽略 padding token，不计算梯度；2.标签平滑，防止过拟合：从[一个1其它全0]的 独热编码 转变成[一个接近1的数，其余很小数] 软标签 
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id('[PAD]'), label_smoothing=0.1).to(device) #loss_fn计算损失的function

    for epoch in range(initial_epoch, config['epoch']):
        
        batch_iterator = tqdm(train_dataloader, desc=f'训练轮次：{epoch:02d}') #显示训练进度条：第n轮次； {epoch:02d}：至少两位数，不足两位数前补0，d：十进制数
        #batch_iterator：被tqdm包装过的train_dataloader，是一个能一批一批吐出数据、同时还会画进度条的迭代器
        for batch in batch_iterator:
            model.train() #修改成训练模式，把模型（包括所有子模块）的 self.training 标志设成 True； torch.nn.Module 这个基类提供的方法

            encoder_input = batch['encoder_input'].to(device) #从batch字典里取出编码序列id存在device里：b*s
            decoder_input = batch['decoder_input'].to(device) #从batch字典里取出解码序列id存在device里：b*s
            encoder_mask = batch['encoder_mask'].to(device) #b*1*1*s
            decoder_mask = batch['decoder_mask'].to(device) #b*1*s*s

            #将张量通过Transformer进行训练！
            encoder_output = model.encode(encoder_input, encoder_mask) #b*s*d
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) #b*s*d
            proj_output = model.project(decoder_output) #b*s*v_tgt

            label = batch['label'].to(device) #b*s

            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            # (B, Seq_Len, tgt_vocab_size) --> (B * Seq_Len, tgt_vocab_size)
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"}) #loss.item()从只包含一个元素的张量中取出 Python 标量值的方法

            #记录损失
            writer.add_scalar('train loss', loss.item(), global_step) #把 loss 写入 TensorBoard，记录成曲线，标签是 'train loss'，横轴是 global_step
            writer.flush() #立刻把数据刷到磁盘，方便实时查看训练曲线

            #反向传播，计算所有参数的梯度
            loss.backward()

            #更新权重
            optimizer.step() #用梯度更新参数（沿着梯度反方向）
            
            optimizer.zero_grad() #清空梯度，为下一步准备

            global_step += 1

        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)
        #print_msg(...) 括号里的字符串，会成为 lambda 的 msg，然后被 batch_iterator.write(msg) 打印到进度条下方。

        #每轮结束后保存模型数据
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(), #模型所有参数（权重、偏置等）
            'optimizer_state_dict': optimizer.state_dict(), #优化器状态（如 Adam 的动量、二阶矩等）
            'global_step': global_step #全局训练步数，也就是每一个batch后都+1
        }, model_filename) #model_filename 是一个字符串，表示保存的文件路径，torch.save 会把这个字典写到这个路径上


if __name__ == '__main__': #运行这个文件本身时，这个文件的__name__是'__main__'
#该py文件被import为模块时，__name__是文件名去掉.py即“train”（此文件）    
    warnings.filterwarnings('ignore') #忽略所有报错
    config = get_config()
    train_model(config)