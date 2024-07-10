#模型训练代码
import torch
from sklearn.metrics import accuracy_score
from tools import wer

def train_epoch(model, criterion, optimizer, dataloader, device, epoch, logger, log_interval, writer):
    model.train()  # 设置模型为训练模式
    losses = []  # 存储每个批次的损失
    all_label = []  # 存储所有的真实标签
    all_pred = []  # 存储所有的预测结果

    for batch_idx, data in enumerate(dataloader):
        inputs, labels = data['data'].to(device), data['label'].to(device)  # 加载数据和标签至指定设备
        optimizer.zero_grad()  # 梯度清零
        outputs = model(inputs)  # 前向传播
        if isinstance(outputs, list):
            outputs = outputs[0]  # 如果输出是列表，取第一个元素

        loss = criterion(outputs, labels.squeeze())  # 计算损失
        losses.append(loss.item())

        prediction = torch.max(outputs, 1)[1]  # 获取预测结果
        all_label.extend(labels.squeeze())  # 收集标签
        all_pred.extend(prediction)  # 收集预测结果
        score = accuracy_score(labels.squeeze().cpu().data.squeeze().numpy(), prediction.cpu().data.squeeze().numpy())  # 计算准确率

        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        if (batch_idx + 1) % log_interval == 0:  # 每隔一定间隔记录日志
            logger.info(f"epoch {epoch+1} | iteration {batch_idx+1} | Loss {loss.item():.6f} | Acc {score*100:.2f}%")

    # 计算平均损失和准确率
    training_loss = sum(losses) / len(losses)
    all_label = torch.stack(all_label, dim=0)
    all_pred = torch.stack(all_pred, dim=0)
    training_acc = accuracy_score(all_label.cpu().data.squeeze().numpy(), all_pred.cpu().data.squeeze().numpy())
    # 记录到Tensorboard
    writer.add_scalars('Loss', {'train': training_loss}, epoch+1)
    writer.add_scalars('Accuracy', {'train': training_acc}, epoch+1)
    logger.info(f"Average Training Loss of Epoch {epoch+1}: {training_loss:.6f} | Acc: {training_acc*100:.2f}%")

def train_seq2seq(model, criterion, optimizer, clip, dataloader, device, epoch, logger, log_interval, writer):
    model.train()  # 设置模型为训练模式
    losses = []  # 存储损失
    all_trg = []  # 存储所有目标
    all_pred = []  # 存储所有预测结果
    all_wer = []  # 存储所有词错误率

    for batch_idx, (imgs, target) in enumerate(dataloader):
        imgs = imgs.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        outputs = model(imgs, target)  # 前向传播

        output_dim = outputs.shape[-1]
        outputs = outputs[1:].view(-1, output_dim)
        target = target.permute(1, 0)[1:].reshape(-1)

        loss = criterion(outputs, target)
        losses.append(loss.item())

        prediction = torch.max(outputs, 1)[1]
        score = accuracy_score(target.cpu().data.squeeze().numpy(), prediction.cpu().data.squeeze().numpy())
        all_trg.extend(target)
        all_pred.extend(prediction)

        batch_size = imgs.shape[0]
        prediction = prediction.view(-1, batch_size).permute(1, 0).tolist()
        target = target.view(-1, batch_size).permute(1, 0).tolist()
        wers = [wer(target[i], prediction[i]) for i in range(batch_size)]
        all_wer.extend(wers)

        loss.backward()  # 反向传播
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  # 梯度裁剪
        optimizer.step()  # 更新参数

        if (batch_idx + 1) % log_interval == 0:
            logger.info(f"epoch {epoch+1} | iteration {batch_idx+1} | Loss {loss.item():.6f} | Acc {score*100:.2f}% | WER {sum(wers)/len(wers):.2f}%")

    training_loss = sum(losses) / len(losses)
    all_trg = torch.stack(all_trg, dim=0)
    all_pred = torch.stack(all_pred, dim=0)
    training_acc = accuracy_score(all_trg.cpu().data.squeeze().numpy(), all_pred.cpu().data.squeeze().numpy())
    training_wer = sum(all_wer) / len(all_wer)
    writer.add_scalars('Loss', {'train': training_loss}, epoch+1)
    writer.add_scalars('Accuracy', {'train': training_acc}, epoch+1)
    writer.add_scalars('WER', {'train': training_wer}, epoch+1)
    logger.info(f"Average Training Loss of Epoch {epoch+1}: {training_loss:.6f} | Acc: {training_acc*100:.2f}% | WER {training_wer:.2f}%")
