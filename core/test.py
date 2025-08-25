import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from vitcross_seg_modeling import VisionTransformer, CONFIGS

# --- 1. 模型和预测配置 ---
model_path = 'best_model.pth'  # 训练保存的最佳模型路径
image_path = './data/imgs/intensity/im00001.png'   # 待预测的单张原图路径（强度图）
range_path = './data/imgs/range/im00001.png'       # 对应的距离图路径
mask_path = './data/masks/im00001.bmp'           # 对应的真实掩膜路径

# 图像大小，与训练时保持一致
img_size = (224, 224)

# --- 2. 模型加载和预处理 ---
def load_model():
    """加载模型并恢复训练好的权重"""
    config = CONFIGS['ViT-B_16']
    config.n_classes = 2
    model = VisionTransformer(config, img_size=224, num_classes=config.n_classes, vis=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 加载模型参数
    if not os.path.exists(model_path):
        print(f"❌ 错误：未找到模型文件 {model_path}，请确保路径正确。")
        return None

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✅ 模型已成功从 {model_path} 加载。")
    return model, device

def preprocess_image(image_path, range_path):
    """加载并预处理输入图片"""
    # 图像预处理与训练时保持一致
    img_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    intensity_img = Image.open(image_path).convert('RGB')
    range_img = Image.open(range_path).convert('RGB')

    # 原始图像，用于后续显示
    # 重要：在这里调整原始图像的大小，以匹配模型输出的掩膜尺寸，解决 IndexError
    intensity_original = intensity_img.resize(img_size)

    # 应用预处理并添加批次维度
    intensity_tensor = img_transform(intensity_img).unsqueeze(0)
    range_tensor = img_transform(range_img).unsqueeze(0)

    return intensity_tensor, range_tensor, intensity_original

# --- 3. 可视化函数 ---
def visualize_results(original_img, true_mask, predicted_mask, alpha=0.5, save_name='prediction_quadrant.png'):
    """
    四宫格可视化：
    1. 原图
    2. 真实掩膜
    3. 预测掩膜
    4. 预测掩膜叠加到原图
    """
    # 将掩膜转换为RGB格式以便叠加
    # 预测掩膜：将0（背景）和1（目标）转换为颜色
    predicted_mask_colored = np.zeros((*predicted_mask.shape, 3), dtype=np.uint8)
    predicted_mask_colored[predicted_mask == 1] = [255, 0, 0]  # 目标用红色

    # 真实掩膜：将0和1转换为颜色
    true_mask_colored = np.zeros((*true_mask.shape, 3), dtype=np.uint8)
    true_mask_colored[true_mask == 1] = [0, 255, 0] # 目标用绿色

    # 将预测掩膜叠加到原图
    original_img_np = np.array(original_img)
    overlay_img = original_img_np.copy()

    # 这一行之前会报错，现在因为 original_img 尺寸已调整，所以可以正常运行
    overlay_img[predicted_mask == 1] = (overlay_img[predicted_mask == 1] * (1 - alpha) + np.array([255, 0, 0]) * alpha).astype(np.uint8)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].imshow(original_img_np)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(true_mask_colored)
    axes[0, 1].set_title("True Mask")
    axes[0, 1].axis('off')

    axes[1, 0].imshow(predicted_mask_colored)
    axes[1, 0].set_title("Predicted Mask")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(overlay_img)
    axes[1, 1].set_title("Overlay (Red)")
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(save_name)
    print(f"✅ 可视化结果已保存到 {save_name}")
    plt.show()

# --- 4. 主函数：执行预测和可视化 ---
def main():
    model, device = load_model()
    if model is None:
        return

    # 预处理输入图片
    intensity_tensor, range_tensor, original_img = preprocess_image(image_path, range_path)

    # 将输入数据移动到正确的设备
    intensity_tensor = intensity_tensor.to(device)
    range_tensor = range_tensor.to(device)

    # 加载真实掩膜
    true_mask_raw = Image.open(mask_path)
    # 调整大小并转换为numpy数组
    true_mask = np.array(true_mask_raw.resize(img_size, Image.NEAREST))

    # 执行预测
    print("⏳ 正在进行模型预测...")
    with torch.no_grad():
        outputs = model(intensity_tensor, range_tensor)

    # 获取预测结果
    _, predicted_mask = torch.max(outputs, 1)
    predicted_mask_np = predicted_mask.squeeze(0).cpu().numpy()

    # 可视化并保存结果
    visualize_results(original_img, true_mask, predicted_mask_np)

# --- 5. 运行脚本 ---
if __name__ == '__main__':
    main()