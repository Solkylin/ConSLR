#seq2seq模型训练代码
import os
import logging
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from dataset import CSL_Continuous, CSL_Continuous_Char
from models.Seq2Seq import Encoder, Decoder, Seq2Seq
from train import train_seq2seq
from validation import val_seq2seq

# Path setting
data_path = "SLR_Dataset/LIANXU_SLR_dataset/color"
dict_path = "SLR_Dataset/GULI_SLR_dataset/dictionary.txt"
corpus_path = "SLR_Dataset/LIANXU_SLR_dataset/corpus.txt"
model_path = "checkpoint/seq2seq_models"
log_path = "log/seq2seq_{:%Y-%m-%d_%H-%M-%S}.log".format(datetime.now())
sum_path = "runs/slr_seq2seq_{:%Y-%m-%d_%H-%M-%S}".format(datetime.now())

# Log to file & tensorboard writer
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.FileHandler(log_path), logging.StreamHandler()])
logger = logging.getLogger('SLR')
logger.info('Logging to file...')
writer = SummaryWriter(sum_path)

# gpus setting
os.environ["CUDA_VISIBLE_DEVICES"]="0"#使用的GPU编号
# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Print the device being used
print("Using device:", device)
# 检查CUDA是否可用，然后返回True或False
cuda_available = torch.cuda.is_available()
# 打印CUDA设备数量和当前选定的CUDA设备
if cuda_available:
    print(f"CUDA is available. Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current CUDA Device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("CUDA is not available. Running on CPU.")


# Hyperparams
epochs = 20#大概一个半小时跑一个循环
batch_size = 8
learning_rate = 1e-4# 01 使用4 02使用3效果不佳
weight_decay = 1e-5# 01 使用5  02使用4效果不佳
sample_size = 128
sample_duration = 48
enc_hid_dim = 512
emb_dim = 256
dec_hid_dim = 512
dropout = 0.5
clip = 1
log_interval = 100

if __name__ == '__main__':
    # Load data
    transform = transforms.Compose([transforms.Resize([sample_size, sample_size]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])
    train_set = CSL_Continuous(data_path=data_path, dict_path=dict_path,
        corpus_path=corpus_path, frames=sample_duration, train=True, transform=transform)
    val_set = CSL_Continuous(data_path=data_path, dict_path=dict_path,
        corpus_path=corpus_path, frames=sample_duration, train=False, transform=transform)
    #train_set = CSL_Continuous_Char(data_path=data_path, corpus_path=corpus_path,
    #    frames=sample_duration, train=True, transform=transform)
    #val_set = CSL_Continuous_Char(data_path=data_path, corpus_path=corpus_path,
    #    frames=sample_duration, train=False, transform=transform)
    logger.info("Dataset samples: {}".format(len(train_set)+len(val_set)))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    # Create Model
    encoder = Encoder(lstm_hidden_size=enc_hid_dim, arch="resnet18").to(device)
    decoder = Decoder(output_dim=train_set.output_dim, emb_dim=emb_dim, enc_hid_dim=enc_hid_dim, dec_hid_dim=dec_hid_dim, dropout=dropout).to(device)
    model = Seq2Seq(encoder=encoder, decoder=decoder, device=device).to(device)
    # Run the model parallelly
    if torch.cuda.device_count() > 1:
        logger.info("Using {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    # Create loss criterion & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # 训练 启动！
    logger.info("Training Started".center(60, '#'))
    for epoch in range(epochs):
        # Train the model
        train_seq2seq(model, criterion, optimizer, clip, train_loader, device, epoch, logger, log_interval, writer)

        # Validate the model
        val_seq2seq(model, criterion, val_loader, device, epoch, logger, writer)

        # Save model
        torch.save(model.state_dict(), os.path.join(model_path, "04slr_seq2seq_epoch{:03d}.pth".format(epoch+1)))#第几次启动记得改模型保存名
        logger.info("Epoch {} Model Saved".format(epoch+1).center(60, '#'))

    logger.info("Training Finished".center(60, '#'))
