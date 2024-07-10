import torch
import torch.nn as nn
import torch.nn.functional as F

# LSTM模型的实现，参考文献：SIGN LANGUAGE RECOGNITION WITH LONG SHORT-TERM MEMORY
class LSTM(nn.Module):
    def __init__(self, lstm_input_size=512, lstm_hidden_size=512, lstm_num_layers=3,
                num_classes=100, hidden1=256, drop_p=0.0):
        super(LSTM, self).__init__()
        # 网络参数
        self.lstm_input_size = lstm_input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.num_classes = num_classes
        self.hidden1 = hidden1
        self.drop_p = drop_p

        # 网络结构
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
            batch_first=True,
        )
        self.drop = nn.Dropout2d(p=self.drop_p)
        self.fc1 = nn.Linear(self.lstm_hidden_size, self.hidden1)
        self.fc2 = nn.Linear(self.hidden1, self.num_classes)

    def forward(self, x):
        # LSTM操作
        self.lstm.flatten_parameters()
        out, (h_n, c_n) = self.lstm(x, None)
        # 选择序列中最后一个时间步的输出进行分类
        out = F.relu(self.fc1(out[:, -1, :]))
        out = F.dropout(out, p=self.drop_p, training=self.training)
        out = self.fc2(out)

        return out

# GRU模型的实现
class GRU(nn.Module):
    def __init__(self, gru_input_size=512, gru_hidden_size=512, gru_num_layers=3,
                num_classes=100, hidden1=256, drop_p=0.0):
        super(GRU, self).__init__()
        # 网络参数
        self.gru_input_size = gru_input_size
        self.gru_hidden_size = gru_hidden_size
        self.gru_num_layers = gru_num_layers
        self.num_classes = num_classes
        self.hidden1 = hidden1
        self.drop_p = drop_p

        # 网络结构
        self.gru = nn.GRU(
            input_size=self.gru_input_size,
            hidden_size=self.gru_hidden_size,
            num_layers=self.gru_num_layers,
            batch_first=True,
        )
        self.drop = nn.Dropout2d(p=self.drop_p)
        self.fc1 = nn.Linear(self.gru_hidden_size, self.hidden1)
        self.fc2 = nn.Linear(self.hidden1, self.num_classes)

    def forward(self, x):
        # GRU操作
        self.gru.flatten_parameters()
        out, hidden = self.gru(x, None)
        # 选择序列中最后一个时间步的输出进行分类
        out = F.relu(self.fc1(out[:, -1, :]))
        out = F.dropout(out, p=self.drop_p, training=self.training)
        out = self.fc2(out)

        return out

# 测试部分
if __name__ == '__main__':
    import sys
    sys.path.append("..")
    from dataset import CSL_Skeleton
    selected_joints = ['HANDLEFT', 'HANDRIGHT', 'ELBOWLEFT', 'ELBOWRIGHT']
    dataset = CSL_Skeleton(data_path="/home/haodong/Data/CSL_Isolated/xf500_body_depth_txt",
        label_path="/home/haodong/Data/CSL_Isolated/dictionary.txt", selected_joints=selected_joints)
    input_size = len(selected_joints) * 2  # 输入大小为选定关节数量的两倍

    # 测试 LSTM
    lstm = LSTM(lstm_input_size=input_size)
    print(lstm(dataset[0]['data'].unsqueeze(0)))

    # 测试 GRU
    gru = GRU(gru_input_size=input_size)
    print(gru(dataset[0]['data'].unsqueeze(0)))
