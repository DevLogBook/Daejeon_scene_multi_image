import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def visualize_match_results(img_A, img_B, warp_AB, confidence_AB=None):
    """
    img_A, img_B: Tensor (B, 3, H, W), 归一化后的图像
    warp_AB: Tensor (B, gs, gs, 2), 模型预测的归一化坐标
    confidence_AB: Tensor (B, gs, gs), 可选，模型预测的置信度
    """
    B, C, H, W = img_A.shape
    device = warp_AB.device

    # 1. 将 32x32 的 warp 场上采样到原图分辨率 (H, W)
    # warp_AB 是 (B, 32, 32, 2)，插值前需要换位成 (B, 2, 32, 32)
    warp_resized = F.interpolate(
        warp_AB.permute(0, 3, 1, 2), 
        size=(H, W), 
        mode='bilinear', 
        align_corners=True
    ).permute(0, 2, 3, 1)  # (B, H, W, 2)

    # 2. 使用 grid_sample 搬运像素
    # 这步的意思是：根据预测的坐标，把图 B 的内容“贴”到图 A 的视角下
    warped_B = F.grid_sample(
        img_B, 
        warp_resized, 
        mode='bilinear', 
        padding_mode='zeros', 
        align_corners=True
    )

    # 3. 绘图 (展示 Batch 中的第一个样本)
    plt.figure(figsize=(15, 5))
    
    # 子图1：原始图 A
    plt.subplot(1, 3, 1)
    plt.title("Image A (Target)")
    plt.imshow(img_A[0].permute(1, 2, 0).cpu().detach().numpy())
    
    # 子图2：变形后的图 B (如果匹配准，它应该长得像 A)
    plt.subplot(1, 3, 2)
    plt.title("Warped Image B (Aligned)")
    plt.imshow(warped_B[0].permute(1, 2, 0).cpu().detach().numpy())
    
    # 子图3：置信度图 (看看模型觉得自己哪里选得准)
    plt.subplot(1, 3, 3)
    plt.title("Confidence Map")
    if confidence_AB is not None:
        conf = confidence_AB[0].cpu().detach().numpy()
        plt.imshow(conf, cmap='jet')
        plt.colorbar()
    else:
        plt.text(0.5, 0.5, "No Confidence Map", ha='center')

    plt.savefig('visres.jpg')
    print("可视化图片已保存至 visres.jpg ")

# 使用示例:
# outputs = model(img_A, img_B)
# visualize_match_results(img_A, img_B, outputs['warp_AB'], outputs['confidence_AB'])