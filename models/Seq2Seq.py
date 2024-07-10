#Seq2Seq基础模型定义
import torch
import torch.nn as nn
import torchvision.models as models
import random

import os, inspect, sys
# 将当前脚本文件所在目录添加到路径中，以便导入其他模块
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, currentdir)
from ConvLSTM import ResCRNN

# 序列到序列模型的实现
# 编码器：编码视频的空间和时间动态，例如CNN+LSTM
# 解码器：解码来自编码器的压缩信息
class Encoder(nn.Module):
    def __init__(self, lstm_hidden_size=512, arch="resnet18"):
        super(Encoder, self).__init__()
        self.lstm_hidden_size = lstm_hidden_size

        # 根据参数选择预训练的ResNet模型并删除最后一个全连接层
        if arch == "resnet18":
            resnet = models.resnet18(pretrained=True)
        elif arch == "resnet34":
            resnet = models.resnet34(pretrained=True)
        elif arch == "resnet50":
            resnet = models.resnet50(pretrained=True)
        elif arch == "resnet101":
            resnet = models.resnet101(pretrained=True)
        elif arch == "resnet152":
            resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.lstm = nn.LSTM(
            input_size=resnet.fc.in_features,
            hidden_size=self.lstm_hidden_size,
            batch_first=True,
        )

    def forward(self, x):
        # CNN
        cnn_embed_seq = []
        for t in range(x.size(2)):  # 遍历每一帧
            out = self.resnet(x[:, :, t, :, :])
            out = out.view(out.size(0), -1)
            cnn_embed_seq.append(out)

        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0)
        cnn_embed_seq = cnn_embed_seq.transpose_(0, 1)

        # LSTM
        self.lstm.flatten_parameters()
        out, (h_n, c_n) = self.lstm(cnn_embed_seq, None)
        return out, (h_n.squeeze(0), c_n.squeeze(0))


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim + enc_hid_dim, dec_hid_dim)
        self.fc = nn.Linear(emb_dim + enc_hid_dim + dec_hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, context):
        input = input.unsqueeze(0)  # 增加序列长度维度
        embedded = self.dropout(self.embedding(input))
        rnn_input = torch.cat((embedded, context.unsqueeze(0)), dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden.unsqueeze(0), cell.unsqueeze(0)))
        hidden = hidden.squeeze(0)
        cell = cell.squeeze(0)
        embedded = embedded.squeeze(0)
        prediction = self.fc(torch.cat((embedded, context, hidden), dim=1))
        return prediction, (hidden, cell)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, imgs, target, teacher_forcing_ratio=0.5):
        batch_size = imgs.shape[0]
        trg_len = target.shape[1]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        encoder_outputs, (hidden, cell) = self.encoder(imgs)
        context = encoder_outputs.mean(dim=1)
        input = target[:, 0]

        for t in range(1, trg_len):
            output, (hidden, cell) = self.decoder(input, hidden, cell, context)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = target[:, t] if teacher_force else top1

        return outputs


# 测试代码
if __name__ == '__main__':
    device = torch.device("cpu")
    encoder = Encoder(lstm_hidden_size=512)
    decoder = Decoder(output_dim=500, emb_dim=256, enc_hid_dim=512, dec_hid_dim=512, dropout=0.5)
    seq2seq = Seq2Seq(encoder=encoder, decoder=decoder, device=device)
    imgs = torch.randn(16, 3, 8, 128, 128)
    target = torch.LongTensor(16, 8).random_(0, 500)
    print(seq2seq(imgs, target).argmax(dim=2).permute(1, 0))  # 输出batch first
