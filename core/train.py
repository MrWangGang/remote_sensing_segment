import os
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
from sklearn.metrics import precision_score, recall_score, jaccard_score, f1_score
import time
import matplotlib.pyplot as plt
from thop import profile

from vitcross_seg_modeling import VisionTransformer, CONFIGS


### --- 1. 模型和训练超参数 ---
data_directory = './data'
num_epochs = 10
batch_size = 4
lr = 1e-4

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # 这里设置了一个随机种子

# --- 2. 自定义数据集类 ---
class MyDataset(Dataset):
    def __init__(self, data_path, transform=None, mask_transform=None):
        self.data_path = data_path
        self.transform = transform
        self.mask_transform = mask_transform
        self.intensity_path = os.path.join(data_path, 'imgs', 'intensity')
        self.range_path = os.path.join(data_path, 'imgs', 'range')
        self.masks_path = os.path.join(data_path, 'masks')
        self.image_names = [f for f in os.listdir(self.intensity_path) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]
        intensity_img = Image.open(os.path.join(self.intensity_path, name)).convert('RGB')
        range_img = Image.open(os.path.join(self.range_path, name)).convert('RGB')
        mask = Image.open(os.path.join(self.masks_path, name.replace('.png', '.bmp')))

        if self.transform:
            intensity_img = self.transform(intensity_img)
            range_img = self.transform(range_img)

        if self.mask_transform:
            mask = self.mask_transform(mask)

        return intensity_img, range_img, mask.long().squeeze()


# --- 3. 指标计算函数 ---
def dice_coeff(pred, target, smooth=1.0):
    num_classes = pred.shape[1]
    target_one_hot = torch.nn.functional.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
    pred_softmax = torch.nn.functional.softmax(pred, dim=1)

    # 计算包含背景在内的所有类别的 mDice
    intersection = (pred_softmax * target_one_hot).sum(dim=[2, 3])
    union = (pred_softmax + target_one_hot).sum(dim=[2, 3])

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean()

def calculate_metrics(pred, target):
    num_classes = pred.shape[1]
    pred_labels = torch.argmax(pred, dim=1).cpu().numpy().flatten()
    target_labels = target.cpu().numpy().flatten()

    # 计算所有类别（包括背景）的 mIoU, mPrecision, mRecall, mF1
    # 使用 'macro' 平均方式，获取每个指标在所有类别上的平均值
    # 'zero_division=0' 用于处理某个类别中没有样本的情况
    iou = jaccard_score(target_labels, pred_labels, average='macro', zero_division=0)
    precision = precision_score(target_labels, pred_labels, average='macro', zero_division=0)
    recall = recall_score(target_labels, pred_labels, average='macro', zero_division=0)
    f1 = f1_score(target_labels, pred_labels, average='macro', zero_division=0)

    # mDice 单独计算
    dice = dice_coeff(pred, target).item()

    return iou, precision, recall, f1, dice

def save_metrics_to_json(metrics_list, filename='training_metrics.json'):
    with open(filename, 'w') as f:
        json.dump(metrics_list, f, indent=4)
    print(f"✅ 指标已保存到 {filename}")

def plot_combined_metrics(filename='training_metrics.json'):
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ 错误：未找到指标文件 {filename}。")
        return

    epochs = [d['epoch'] for d in data]
    train_losses = [d['train_loss'] for d in data]
    val_losses = [d['val_loss'] for d in data]
    val_ious = [d['val_iou'] for d in data]
    val_dices = [d['val_dice'] for d in data]
    val_precisions = [d['val_precision'] for d in data]
    val_recalls = [d['val_recall'] for d in data]

    # 创建一个大的宫格图，包含5个子图
    fig, axs = plt.subplots(3, 2, figsize=(15, 20), constrained_layout=True)

    # 移除最后一个空的子图
    fig.delaxes(axs[2, 1])

    # 子图1: 损失曲线
    axs[0, 0].plot(epochs, train_losses, label='Train Loss')
    axs[0, 0].plot(epochs, val_losses, label='Validation Loss')
    axs[0, 0].set_title('Loss Curves')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # 子图2: mIoU 曲线
    axs[0, 1].plot(epochs, val_ious, label='mIoU', color='orange')
    axs[0, 1].set_title('mIoU')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Score')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # 子图3: mDice 曲线
    axs[1, 0].plot(epochs, val_dices, label='mDice', color='green')
    axs[1, 0].set_title('mDice')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Score')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # 子图4: mPrecision 曲线
    axs[1, 1].plot(epochs, val_precisions, label='mPrecision', color='red')
    axs[1, 1].set_title('mPrecision')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Score')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    # 子图5: mRecall 曲线
    axs[2, 0].plot(epochs, val_recalls, label='mRecall', color='purple')
    axs[2, 0].set_title('mRecall')
    axs[2, 0].set_xlabel('Epoch')
    axs[2, 0].set_ylabel('Score')
    axs[2, 0].legend()
    axs[2, 0].grid(True)

    plt.suptitle('Training and Validation Metrics', fontsize=20, y=1.02)
    plt.savefig('combined_metrics.png', bbox_inches='tight')
    plt.close()
    print("✅ 所有指标曲线已合并并保存为 combined_metrics.png")


def save_performance_metrics(model, img_size=(224, 224), filename='model_performance.txt'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 参数量计算
    total_params = sum(p.numel() for p in model.parameters())

    # 2. FLOPs (浮点运算数) 计算
    dummy_input1 = torch.randn(1, 3, img_size[0], img_size[1]).to(device)
    dummy_input2 = torch.randn(1, 3, img_size[0], img_size[1]).to(device)
    flops, _ = profile(model, inputs=(dummy_input1, dummy_input2,), verbose=False)

    # 3. FPS (每秒帧数) 计算
    # 热身
    for _ in range(10):
        _ = model(dummy_input1, dummy_input2)

    # 计时
    start_time = time.time()
    num_runs = 100
    for _ in range(num_runs):
        _ = model(dummy_input1, dummy_input2)
    end_time = time.time()

    avg_time = (end_time - start_time) / num_runs
    fps = 1.0 / avg_time

    with open(filename, 'w') as f:
        f.write(f"Model Performance Metrics:\n")
        f.write(f"----------------------------------------\n")
        f.write(f"Total Parameters: {total_params:,}\n")
        f.write(f"FLOPs: {flops / 1e9:.2f} G\n") # 转换为 GFLOPs
        f.write(f"Average FPS (on {device}): {fps:.2f}\n")

    print(f"✅ 性能指标已保存到 {filename}")


# --- 4. 主训练函数 ---
def train_model():
    # 模型配置
    config = CONFIGS['ViT-B_16']
    config.n_classes = 2

    # 数据预处理
    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])

    # 加载和划分数据集
    full_dataset = MyDataset(data_directory, transform=img_transform, mask_transform=mask_transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 实例化模型
    model = VisionTransformer(config, img_size=224, num_classes=config.n_classes, vis=False)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"模型已准备好在 {device} 上进行训练。")

    metrics_history = []
    best_val_loss = float('inf')  # 初始化一个大的值来跟踪最佳验证损失
    best_model_path = 'best_model.pth' # 定义模型保存路径

    # 训练和验证循环
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss_list = []
        train_pbar = tqdm(train_dataloader, desc=f"第 {epoch+1}/{num_epochs} 轮 (训练)", leave=False)
        for intensity_imgs, range_imgs, masks in train_pbar:
            intensity_imgs = intensity_imgs.to(device)
            range_imgs = range_imgs.to(device)
            masks = masks.to(device)

            outputs = model(intensity_imgs, range_imgs)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_list.append(loss.item())
            train_pbar.set_postfix({'loss': np.mean(train_loss_list)})

        avg_train_loss = np.mean(train_loss_list)

        # 验证阶段
        model.eval()
        val_loss_list = []
        val_dice_list = []
        val_iou_list = []
        val_precision_list = []
        val_recall_list = []
        val_f1_list = []

        val_pbar = tqdm(val_dataloader, desc=f"第 {epoch+1}/{num_epochs} 轮 (验证)", leave=False)
        with torch.no_grad():
            for intensity_imgs, range_imgs, masks in val_pbar:
                intensity_imgs = intensity_imgs.to(device)
                range_imgs = range_imgs.to(device)
                masks = masks.to(device)

                outputs = model(intensity_imgs, range_imgs)
                val_loss = criterion(outputs, masks)
                val_loss_list.append(val_loss.item())

                iou, precision, recall, f1, dice = calculate_metrics(outputs, masks)
                val_iou_list.append(iou)
                val_precision_list.append(precision)
                val_recall_list.append(recall)
                val_f1_list.append(f1)
                val_dice_list.append(dice)

                val_pbar.set_postfix({
                    'loss': np.mean(val_loss_list),
                    'mDice': np.mean(val_dice_list),
                    'mIoU': np.mean(val_iou_list),
                    'mF1': np.mean(val_f1_list)
                })

        avg_val_loss = np.mean(val_loss_list)
        avg_val_dice = np.mean(val_dice_list)
        avg_val_iou = np.mean(val_iou_list)
        avg_val_precision = np.mean(val_precision_list)
        avg_val_recall = np.mean(val_recall_list)
        avg_val_f1 = np.mean(val_f1_list)

        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_dice": avg_val_dice,
            "val_iou": avg_val_iou,
            "val_precision": avg_val_precision,
            "val_recall": avg_val_recall,
            "val_f1": avg_val_f1
        }
        metrics_history.append(epoch_metrics)

        print(f'第 [{epoch+1}/{num_epochs}] 轮')
        print(f'  训练损失: {avg_train_loss:.4f}, 验证损失: {avg_val_loss:.4f}')
        print(f'  验证指标: mDice: {avg_val_dice:.4f}, mIoU: {avg_val_iou:.4f}, mPrecision: {avg_val_precision:.4f}, mRecall: {avg_val_recall:.4f}, mF1: {avg_val_f1:.4f}')

        # 检查是否为最佳模型并保存
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"🎉 验证损失降低至 {best_val_loss:.4f}。已保存最佳模型到 {best_model_path}")

    print("\n训练完成！")
    save_metrics_to_json(metrics_history)
    plot_combined_metrics()
    save_performance_metrics(model)


# --- 5. 运行训练 ---
if __name__ == '__main__':
    train_model()