# 遥感图像多模态分割

本项目针对双 3 通道图像输入的分割任务（如多模态遥感影像分割、医学影像多序列分割等），构建 “残差网络（ResNet）+ U-Net” 混合架构模型。通过将两个独立的 3 通道图像（如 “常规遥感 RGB 图 + 多光谱遥感图”“CT 影像 + MRI 影像”）作为输入，利用 ResNet 优化的 U-Net 编码器提取深层融合特征，结合 U-Net 解码器的细节恢复能力，实现目标区域的像素级精准分割。模型通过双模态特征互补解决单模态信息不足的问题，适用于城市建筑提取、病灶区域分割、生态地物分类等场景。

<!-- 横向排列图片容器：使用 Flexbox 确保图片水平对齐，添加间距和居中效果 -->
<div style="display: flex; justify-content: center; gap: 15px; flex-wrap: wrap;">
  <!-- 第一张图片：宽度统一设为 200px（可按需调整），保持宽高比 -->
  <img src="https://github.com/user-attachments/assets/69c7bd6e-b81f-48b4-9c45-c57915a412ab" alt="图片1" style="width: 200px; height: auto;">
  <!-- 第二张图片：与第一张尺寸一致，确保排列整齐 -->
  <img src="https://github.com/user-attachments/assets/224879e9-d759-426d-8040-45be323a8bd0" alt="图片2" style="width: 200px; height: auto;">
  <!-- 第三张图片：同样统一尺寸 -->
  <img src="https://github.com/user-attachments/assets/3d067c68-07f3-4075-b0b0-0c444c56100d" alt="图片3" style="width: 200px; height: auto;">
</div>



