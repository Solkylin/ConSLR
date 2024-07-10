# 注意力模块，参考文献: Learn To Pay Attention
import torch
import torch.nn as nn
import torch.nn.functional as F
class ProjectorBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ProjectorBlock, self).__init__()
        # 使用1x1的卷积核来改变通道数，不使用偏置项
        self.op = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
            kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        return self.op(x)  # 执行卷积操作

# 三维版本的投影块，用于处理3D数据
class ProjectorBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ProjectorBlock3D, self).__init__()
        # 使用1x1的3D卷积核来改变通道数，不使用偏置项
        self.op = nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
            kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        return self.op(x)  # 执行3D卷积操作

# 线性注意力块，可选择是否归一化注意力权重
class LinearAttentionBlock(nn.Module):
    def __init__(self, in_channels, normalize_attn=True):
        super(LinearAttentionBlock, self).__init__()
        self.normalize_attn = normalize_attn  # 是否归一化
        self.op = nn.Conv2d(in_channels=in_channels, out_channels=1,
            kernel_size=1, padding=0, bias=False)

    def forward(self, l, g):
        N, C, H, W = l.size()
        c = self.op(l + g)  # 生成注意力图
        if self.normalize_attn:
            a = F.softmax(c.view(N, 1, -1), dim=2).view(N, 1, H, W)  # 归一化处理
        else:
            a = torch.sigmoid(c)  # 使用sigmoid函数处理
        g = torch.mul(a.expand_as(l), l)  # 应用注意力权重
        if self.normalize_attn:
            g = g.view(N, C, -1).sum(dim=2)  # 求和
        else:
            g = F.adaptive_avg_pool2d(g, (1, 1)).view(N, C)  # 平均池化
        return c.view(N, 1, H, W), g  # 返回注意力图和结果

# 三维数据的线性注意力块
class LinearAttentionBlock3D(nn.Module):
    def __init__(self, in_channels, normalize_attn=True):
        super(LinearAttentionBlock3D, self).__init__()
        self.normalize_attn = normalize_attn
        self.op = nn.Conv3d(in_channels=in_channels, out_channels=1,
            kernel_size=1, padding=0, bias=False)

    def forward(self, l, g):
        N, C, T, H, W = l.size()
        c = self.op(l + g)  # 生成注意力图
        if self.normalize_attn:
            a = F.softmax(c.view(N, 1, -1), dim=2).view(N, 1, T, H, W)
        else:
            a = torch.sigmoid(c)
        g = torch.mul(a.expand_as(l), l)
        if self.normalize_attn:
            g = g.view(N, C, -1).sum(dim=2)
        else:
            g = F.adaptive_avg_pool3d(g, (1, 1, 1)).view(N, C)
        return c.view(N, 1, T, H, W), g

# LSTM注意力块
class LSTMAttentionBlock(nn.Module):
    def __init__(self, hidden_size):
        super(LSTMAttentionBlock, self).__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.fc2 = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)

    def forward(self, hidden_states):
        # 对隐藏状态进行变换以计算得分
        score_first_part = self.fc1(hidden_states)
        h_t = hidden_states[:, -1, :]  # 获取最后一个时间步的隐藏状态
        score = torch.bmm(score_first_part, h_t.unsqueeze(2)).squeeze(2)  # 计算注意力得分
        attention_weights = F.softmax(score, dim=1)  # 得到注意力权重
        context_vector = torch.bmm(hidden_states.permute(0, 2, 1), attention_weights.unsqueeze(2)).squeeze(2)
        pre_activation = torch.cat((context_vector, h_t), dim=1)
        attention_vector = self.fc2(pre_activation)  # 计算注意力向量
        attention_vector = torch.tanh(attention_vector)  # 应用tanh激活函数
        return attention_vector

# 测试代码，实例化并运行定义的模块
if __name__ == '__main__':
    # 2D注意力模块
    attention_block = LinearAttentionBlock(in_channels=3)
    l = torch.randn(16, 3, 128, 128)
    g = torch.randn(16, 3, 128, 128)
    print(attention_block(l, g))
    # 3D注意力模块
    attention_block_3d = LinearAttentionBlock3D(in_channels=3)
    l = torch.randn(16, 3, 16, 128, 128)
    g = torch.randn(16, 3, 16, 128, 128)
    print(attention_block_3d(l, g))
    # LSTM注意力模块
    attention_block_lstm = LSTMAttentionBlock(hidden_size=256)
    hidden_states = torch.randn(32, 16, 256)
    print(attention_block_lstm(hidden_states).shape)
