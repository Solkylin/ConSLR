#连续手语模型测试代码
import torch
from sklearn.metrics import accuracy_score
from tools import wer

def val_epoch(model, criterion, dataloader, device, epoch, logger, writer):
    model.eval()  # 设置模型为评估模式
    losses = []  # 存储损失值
    all_label = []  # 存储所有真实标签
    all_pred = []  # 存储所有预测结果

    with torch.no_grad():  # 关闭梯度计算
        for batch_idx, data in enumerate(dataloader):
            inputs, labels = data['data'].to(device), data['label'].to(device)  # 加载数据和标签到设备
            outputs = model(inputs)  # 模型前向传播
            if isinstance(outputs, list):
                outputs = outputs[0]  # 如果输出是列表形式，取第一个元素
            loss = criterion(outputs, labels.squeeze())  # 计算损失
            losses.append(loss.item())
            prediction = torch.max(outputs, 1)[1]  # 获取预测结果
            all_label.extend(labels.squeeze())  # 收集真实标签
            all_pred.extend(prediction)  # 收集预测结果

    validation_loss = sum(losses) / len(losses)  # 计算平均损失
    all_label = torch.stack(all_label, dim=0)
    all_pred = torch.stack(all_pred, dim=0)
    validation_acc = accuracy_score(all_label.cpu().numpy(), all_pred.cpu().numpy())  # 计算准确率

    writer.add_scalars('Loss', {'validation': validation_loss}, epoch+1)  # 记录损失
    writer.add_scalars('Accuracy', {'validation': validation_acc}, epoch+1)  # 记录准确率
    logger.info(f"Average Validation Loss of Epoch {epoch+1}: {validation_loss:.6f} | Acc: {validation_acc*100:.2f}%")  # 打印信息

def val_seq2seq(model, criterion, dataloader, device, epoch, logger, writer):
    model.eval()  # 设置模型为评估模式
    losses = []
    all_trg = []
    all_pred = []
    all_wer = []

    with torch.no_grad():
        for batch_idx, (imgs, target) in enumerate(dataloader):
            imgs = imgs.to(device)
            target = target.to(device)
            outputs = model(imgs, target, 0)  # 前向传播，无teacher forcing

            output_dim = outputs.shape[-1]
            outputs = outputs[1:].view(-1, output_dim)  # 调整输出形状以计算损失
            target = target.permute(1, 0)[1:].reshape(-1)

            loss = criterion(outputs, target)  # 计算损失
            losses.append(loss.item())
            prediction = torch.max(outputs, 1)[1]  # 获取预测结果
            score = accuracy_score(target.cpu().numpy(), prediction.cpu().numpy())
            all_trg.extend(target)
            all_pred.extend(prediction)

            batch_size = imgs.shape[0]
            prediction = prediction.view(-1, batch_size).permute(1, 0).tolist()
            target = target.view(-1, batch_size).permute(1, 0).tolist()
            wers = [wer(target[i], prediction[i]) for i in range(batch_size)]  # 计算WER
            all_wer.extend(wers)

    validation_loss = sum(losses) / len(losses)
    all_trg = torch.stack(all_trg, dim=0)
    all_pred = torch.stack(all_pred, dim=0)
    validation_acc = accuracy_score(all_trg.cpu().numpy(), all_pred.cpu().numpy())
    validation_wer = sum(all_wer) / len(all_wer)

    writer.add_scalars('Loss', {'validation': validation_loss}, epoch+1)  # 记录损失
    writer.add_scalars('Accuracy', {'validation': validation_acc}, epoch+1)  # 记录准确率
    writer.add_scalars('WER', {'validation': validation_wer}, epoch+1)  # 记录WER
    logger.info(f"Average Validation Loss of Epoch {epoch+1}: {validation_loss:.6f} | Acc: {validation_acc*100:.2f}% | WER: {validation_wer:.2f}%")  # 打印信息
