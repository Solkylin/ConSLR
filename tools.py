#一些工具函数实现
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as utils
import cv2
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def get_label_and_pred(model, test_loader, device):
    all_label = []
    all_pred = []
    with torch.no_grad():  # 禁止梯度计算
        for batch_idx, data in enumerate(test_loader):
            inputs, labels = data['data'].to(device), data['label'].to(device)  # 加载数据到指定设备
            outputs = model(inputs)  # 模型预测
            if isinstance(outputs, list):  # 如果输出为列表，取第一个元素
                outputs = outputs[0]
            prediction = torch.max(outputs, 1)[1]  # 取最大概率标签作为预测结果
            all_label.extend(labels.squeeze())  # 收集所有标签
            all_pred.extend(prediction)  # 收集所有预测结果
    all_label = torch.stack(all_label, dim=0)  # 转换为张量
    all_pred = torch.stack(all_pred, dim=0)
    all_label = all_label.squeeze().cpu().data.squeeze().numpy()  # 转换为numpy数组
    all_pred = all_pred.cpu().data.squeeze().numpy()
    return all_label, all_pred

def plot_confusion_matrix(model, dataloader, dataset, device, save_path='confmat.png', normalize=True):
    all_label, all_pred = get_label_and_pred(model, dataloader, device)  # 获取标签和预测
    confmat = confusion_matrix(all_label, all_pred)  # 生成混淆矩阵

    if normalize:
        confmat = confmat.astype('float') / confmat.sum(axis=1)[:, np.newaxis]  # 归一化
    plt.figure(figsize=(20,20))
    plt.imshow(confmat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    ticks = np.arange(100)
    plt.xticks(ticks, fontsize=8)
    plt.yticks(ticks, fontsize=8)
    plt.grid(True)
    plt.title('Confusion matrix', fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)
    plt.ylabel('True label', fontsize=20)
    plt.savefig(save_path)  # 保存图像

    sorted_index = np.diag(confmat).argsort()  # 排序
    for i in range(10):
        print(dataset.label_to_word(int(sorted_index[i])), confmat[sorted_index[i]][sorted_index[i]])
    np.savetxt('matrix.csv', confmat, delimiter=',')  # 保存为CSV文件

def visualize_attn(I, c):
    img = I.permute((1,2,0)).cpu().numpy()
    N, C, H, W = c.size()
    a = F.softmax(c.view(N,C,-1), dim=2).view(N,C,H,W)  # 计算注意力权重
    up_factor = 128/H
    if up_factor > 1:
        a = F.interpolate(a, scale_factor=up_factor, mode='bilinear', align_corners=False)
    attn = utils.make_grid(a, nrow=4, normalize=True, scale_each=True)
    attn = attn.permute((1,2,0)).mul(255).byte().cpu().numpy()
    attn = cv2.applyColorMap(attn, cv2.COLORMAP_JET)
    attn = cv2.cvtColor(attn, cv2.COLOR_BGR2RGB)
    vis = 0.6 * img + 0.4 * attn  # 融合原始图像和注意力图
    return torch.from_numpy(vis).permute(2,0,1)

def plot_attention_map(model, dataloader, device):
    writer = SummaryWriter("runs/attention_{:%Y-%m-%d_%H-%M-%S}".format(datetime.now()))
    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            inputs = data['data'].to(device)
            if batch_idx == 0:
                images = inputs[0:16,:,:,:,:]
                I = utils.make_grid(images[:,:,0,:,:], nrow=4, normalize=True, scale_each=True)
                writer.add_image('origin', I)
                _, c1, c2, c3, c4 = model(images)
                attn1 = visualize_attn(I, c1[:,:,0,:,:])
                writer.add_image('attn1', attn1)
                attn2 = visualize_attn(I, c2[:,:,0,:,:])
                writer.add_image('attn2', attn2)
                attn3 = visualize_attn(I, c3[:,:,0,:,:])
                writer.add_image('attn3', attn3)
                attn4 = visualize_attn(I, c4[:,:,0,:,:])
                writer.add_image('attn4', attn4)
                break

def wer(r, h):
    d = np.zeros((len(r)+1)*(len(h)+1), dtype=np.uint8)
    d = d.reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion = d[i][j-1] + 1
                deletion = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)
    return float(d[len(r)][len(h)]) / len(r) * 100

if __name__ == '__main__':
    r = [1,2,3,4]
    h = [1,1,3,5,6]
    print(wer(r, h))
