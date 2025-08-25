# 遥感图像多模态分割

本项目聚焦多模态遥感图像分割任务，融合 “遥感热成像图”（反映地物温度分布，如建筑热辐射、水体热特征）与 “常规遥感图”（反映地物光谱 / 纹理信息，如植被、道路、建筑外观）两类数据，构建 “残差网络（ResNet）+ U-Net” 混合架构的分割模型。通过多模态特征互补提升分割精度，可实现建筑区域提取、水体范围划定、植被覆盖度计算等目标的像素级分割，为城市规划、生态环境监测、灾害应急评估（如火灾后热异常区域识别）等场景提供高效的遥感数据解读方案。

<!-- 横向排列图片容器：使用 Flexbox 确保图片水平对齐，添加间距和居中效果 -->
<div style="display: flex; justify-content: center; gap: 15px; flex-wrap: wrap;">
  <!-- 第一张图片：宽度统一设为 200px（可按需调整），保持宽高比 -->
  <img src="https://github.com/user-attachments/assets/69c7bd6e-b81f-48b4-9c45-c57915a412ab" alt="图片1" style="width: 200px; height: auto;">
  <!-- 第二张图片：与第一张尺寸一致，确保排列整齐 -->
  <img src="https://github.com/user-attachments/assets/224879e9-d759-426d-8040-45be323a8bd0" alt="图片2" style="width: 200px; height: auto;">
  <!-- 第三张图片：同样统一尺寸 -->
  <img src="https://github.com/user-attachments/assets/3d067c68-07f3-4075-b0b0-0c444c56100d" alt="图片3" style="width: 200px; height: auto;">
</div>



