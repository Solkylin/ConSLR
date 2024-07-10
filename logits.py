#对单个视频的预测
import cv2
from PIL import Image
from models.Seq2Seq import Encoder, Decoder, Seq2Seq
import torchvision.transforms as transforms
import torch

# 加载模型和字典
output_dim = 3
dictionary = {'<pad>': 0, '<sos>': 1, '<eos>': 2}
with open('SLR_Dataset/GULI_SLR_dataset/dictionary.txt', 'r', encoding='utf-8') as f:
#with open('SLR_Dataset/LIANXU_SLR_dataset/corpus.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        line = line.strip().split('\t')
        # word with multiple expressions
        if '(' in line[1] and ')' in line[1]:
            for delimeter in ['(', ')', '、']:
                line[1] = line[1].replace(delimeter, " ")
            words = line[1].split()
        else:
            words = [line[1]]
        for word in words:
            dictionary[word] = output_dim
        output_dim += 1
label2word = {label: word for word, label in dictionary.items()}

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
sample_size = 128
enc_hid_dim = 512
emb_dim = 256
dec_hid_dim = 512
dropout = 0.5
encoder = Encoder(
    lstm_hidden_size=enc_hid_dim, 
    arch="resnet18")
decoder = Decoder(
    output_dim=output_dim, 
    emb_dim=emb_dim, enc_hid_dim=enc_hid_dim, 
    dec_hid_dim=dec_hid_dim, 
    dropout=dropout)
model = Seq2Seq(
    encoder=encoder, 
    decoder=decoder, 
    device=device).to(device)

#使用这个的话，模型是503，dictionary.txt里500-503记得删
# 000500	国家
# 000501	经济
# 000502	孩子
# 000503	的
#model.load_state_dict(torch.load('./checkpoint/seq2seq_models/03slr_seq2seq_epoch018.pth'))

#如果使用097那一个，是507，不是503，要在原始数据表 dictionary.txt 里添加4个词库。就是 500-503 四个。
#097的效果好很多，就先用这个吧。怪不得之前谓语一直识别不出来，狗屎一样的东西，浪费我时间。
model.load_state_dict(torch.load('./checkpoint/seq2seq_models/slr_seq2seq_epoch097.pth'))

model.eval()

transform = transforms.Compose([transforms.Resize([sample_size, sample_size]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])
sample_duration = 48
max_len = 16#预测手语的最大长度，初始为16


def get_inputs(video_path):
    cap = cv2.VideoCapture(video_path)
    fps_all = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # 取整数部分
    timeF = int(fps_all / sample_duration) if fps_all > sample_duration else 1
    for _ in range(10):
        ret, frame = cap.read()
    images = []
    n = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            break
        else:
            if (n % timeF == 0):
                image = Image.fromarray(frame)
                images.append(transform(image))
        n += 1
    images = torch.stack(images, dim=0)
    # 原本是帧，通道，h，w，需要换成可供3D CNN使用的形状
    images = images.permute(1, 0, 2, 3)
    return images

# 给视频加字幕
def translate(video_path):
    images = get_inputs(video_path)
    with torch.no_grad():
        target = torch.zeros(1, max_len).long().to(device)
        outputs = model(images.unsqueeze(0).to(device), target, 0)
    preds = torch.argmax(outputs, -1).cpu().numpy()
    pred_result = []
    for i in range(preds.shape[0]):
        pred = preds[i]
        if pred == dictionary['<sos>'] or pred == dictionary['<pad>']:
            continue
        if pred == dictionary['<eos>']:
            break
        res = label2word[pred[0]]
        pred_result.append(res)
    return pred_result