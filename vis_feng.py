import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cv2
from pathlib import Path


class MatchingPeakVisualizer:
    """
    匹配峰可视化工具
    用于直观展示单峰/多峰匹配现象
    """
    
    def __init__(self, model, device='cuda'):
        """
        Args:
            model: 你的匹配网络，需要能输出 sim_matrix
        """
        self.model = model
        self.device = device
        self.model.eval()
    
    @torch.no_grad()
    def extract_similarity_matrix(self, img_a, img_b):
        """
        提取相似度矩阵
        Returns:
            sim_matrix: [H*W, H*W] 相似度矩阵
            feat_a, feat_b: 特征图
        """
        # 假设你的模型输出包含 sim_matrix
        output = self.model(img_a.to(self.device), img_b.to(self.device))
        
        # 如果模型没有直接输出sim_matrix，手动计算
        if 'sim_matrix' in output:
            sim_matrix = output['sim_matrix'][0]  # [H*W, H*W]
        else:
            # 从特征手动计算
            feat_a = output.get('feat_A_coarse', output.get('feat_A'))
            feat_b = output.get('feat_B_coarse', output.get('feat_B'))
            
            B, C, H, W = feat_a.shape
            fa = feat_a.flatten(2).transpose(1, 2)  # [B, H*W, C]
            fb = feat_b.flatten(2).transpose(1, 2)
            
            # 归一化后点积
            fa = F.normalize(fa, dim=-1)
            fb = F.normalize(fb, dim=-1)
            sim_matrix = torch.bmm(fa, fb.transpose(-1, -2))[0]  # [H*W, H*W]
        
        return sim_matrix.cpu(), output
    
    def visualize_single_point_matching(self, img_a, img_b, query_points, 
                                         save_path=None, figsize=(16, 12)):
        """
        可视化单个/多个查询点的匹配分布
        
        Args:
            img_a, img_b: [1, 3, H, W] 输入图像张量
            query_points: [(y, x), ...] 要可视化的查询点列表（特征图坐标）
            save_path: 保存路径
        """
        sim_matrix, output = self.extract_similarity_matrix(img_a, img_b)
        
        # 获取特征图尺寸
        N = int(np.sqrt(sim_matrix.shape[0]))
        H_feat, W_feat = N, N
        
        # 图像预处理用于显示
        img_a_np = self._tensor_to_numpy(img_a)
        img_b_np = self._tensor_to_numpy(img_b)
        H_img, W_img = img_a_np.shape[:2]
        
        # 计算缩放比例
        scale_y = H_img / H_feat
        scale_x = W_img / W_feat
        
        n_points = len(query_points)
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(n_points, 4, figure=fig, width_ratios=[1, 1, 1.2, 0.8])
        
        for idx, (qy, qx) in enumerate(query_points):
            # 查询点在展平后的索引
            query_idx = qy * W_feat + qx
            
            # 获取该点对所有目标位置的相似度
            similarity_1d = sim_matrix[query_idx]  # [H*W]
            similarity_2d = similarity_1d.reshape(H_feat, W_feat).numpy()
            
            # ===== 分析峰 =====
            peaks_info = self._analyze_peaks(similarity_2d)
            
            # ===== 绘图 =====
            
            # 1. 源图像 + 查询点标记
            ax1 = fig.add_subplot(gs[idx, 0])
            ax1.imshow(img_a_np)
            ax1.scatter([qx * scale_x], [qy * scale_y], 
                       c='red', s=200, marker='*', edgecolors='white', linewidths=2)
            ax1.set_title(f'Source Image\nQuery Point ({qy}, {qx})', fontsize=10)
            ax1.axis('off')
            
            # 2. 目标图像 + 匹配候选标记
            ax2 = fig.add_subplot(gs[idx, 1])
            ax2.imshow(img_b_np)
            
            # 标记所有峰的位置
            colors = plt.cm.Set1(np.linspace(0, 1, len(peaks_info['peaks'])))
            for i, (peak_y, peak_x, peak_val) in enumerate(peaks_info['peaks'][:5]):  # 最多显示5个峰
                ax2.scatter([peak_x * scale_x], [peak_y * scale_y],
                           c=[colors[i]], s=150, marker='o', 
                           edgecolors='white', linewidths=2,
                           label=f'Peak {i+1}: {peak_val:.3f}')
            ax2.legend(loc='upper right', fontsize=8)
            ax2.set_title(f'Target Image\n{peaks_info["num_peaks"]} peaks detected', fontsize=10)
            ax2.axis('off')
            
            # 3. 相似度热力图
            ax3 = fig.add_subplot(gs[idx, 2])
            im = ax3.imshow(similarity_2d, cmap='hot', interpolation='bilinear')
            
            # 在热力图上标记峰
            for i, (peak_y, peak_x, _) in enumerate(peaks_info['peaks'][:5]):
                ax3.scatter([peak_x], [peak_y], c=[colors[i]], s=100, 
                           marker='x', linewidths=3)
            
            plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
            ax3.set_title(f'Similarity Heatmap\nEntropy: {peaks_info["entropy"]:.3f}', fontsize=10)
            ax3.set_xlabel('Target X')
            ax3.set_ylabel('Target Y')
            
            # 4. 相似度分布直方图 + 峰分析
            ax4 = fig.add_subplot(gs[idx, 3])
            
            # 绘制排序后的相似度曲线
            sorted_sim = np.sort(similarity_1d.numpy())[::-1]
            ax4.plot(sorted_sim[:100], 'b-', linewidth=2, label='Similarity')
            ax4.axhline(y=peaks_info['peaks'][0][2], color='r', 
                       linestyle='--', label=f'Max: {peaks_info["peaks"][0][2]:.3f}')
            if len(peaks_info['peaks']) > 1:
                ax4.axhline(y=peaks_info['peaks'][1][2], color='orange',
                           linestyle='--', label=f'2nd: {peaks_info["peaks"][1][2]:.3f}')
            
            ax4.set_xlabel('Rank')
            ax4.set_ylabel('Similarity')
            ax4.set_title(f'Top-100 Similarities\nPeak Ratio: {peaks_info["peak_ratio"]:.2f}', fontsize=10)
            ax4.legend(fontsize=8)
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        plt.show()
        return fig
    
    def visualize_full_image_peaks(self, img_a, img_b, save_path=None, figsize=(14, 10)):
        """
        可视化整张图像的多峰分布情况
        生成"多峰热力图"，显示哪些区域存在匹配歧义
        """
        sim_matrix, output = self.extract_similarity_matrix(img_a, img_b)
        
        N = int(np.sqrt(sim_matrix.shape[0]))
        H_feat, W_feat = N, N
        
        # 对每个位置分析峰
        entropy_map = np.zeros((H_feat, W_feat))
        num_peaks_map = np.zeros((H_feat, W_feat))
        peak_ratio_map = np.zeros((H_feat, W_feat))
        
        for y in range(H_feat):
            for x in range(W_feat):
                idx = y * W_feat + x
                sim_1d = sim_matrix[idx].reshape(H_feat, W_feat).numpy()
                peaks_info = self._analyze_peaks(sim_1d)
                
                entropy_map[y, x] = peaks_info['entropy']
                num_peaks_map[y, x] = peaks_info['num_peaks']
                peak_ratio_map[y, x] = peaks_info['peak_ratio']
        
        # 可视化
        img_a_np = self._tensor_to_numpy(img_a)
        img_b_np = self._tensor_to_numpy(img_b)
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # 第一行：原图
        axes[0, 0].imshow(img_a_np)
        axes[0, 0].set_title('Source Image A')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(img_b_np)
        axes[0, 1].set_title('Target Image B')
        axes[0, 1].axis('off')
        
        # 置信度图（如果有）
        if 'confidence_AB' in output:
            conf = output['confidence_AB'][0, 0].cpu().numpy()
            im = axes[0, 2].imshow(conf, cmap='RdYlGn', vmin=0, vmax=1)
            axes[0, 2].set_title('Model Confidence')
            plt.colorbar(im, ax=axes[0, 2], fraction=0.046)
        else:
            axes[0, 2].axis('off')
        
        # 第二行：峰分析图
        im1 = axes[1, 0].imshow(entropy_map, cmap='hot')
        axes[1, 0].set_title('Matching Entropy\n(High = Ambiguous)')
        plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)
        
        im2 = axes[1, 1].imshow(num_peaks_map, cmap='YlOrRd', vmin=1, vmax=5)
        axes[1, 1].set_title('Number of Peaks\n(>1 = Multi-peak)')
        plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)
        
        im3 = axes[1, 2].imshow(peak_ratio_map, cmap='RdYlGn', vmin=0, vmax=1)
        axes[1, 2].set_title('Peak Ratio (1st/2nd)\n(Low = Ambiguous)')
        plt.colorbar(im3, ax=axes[1, 2], fraction=0.046)
        
        plt.suptitle('Multi-Peak Analysis for Image Matching', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        
        # 打印统计信息
        print("\n" + "="*50)
        print("Multi-Peak Statistics:")
        print(f"  Average Entropy: {entropy_map.mean():.3f}")
        print(f"  High Entropy Ratio (>2.0): {(entropy_map > 2.0).mean()*100:.1f}%")
        print(f"  Multi-peak Ratio (≥2 peaks): {(num_peaks_map >= 2).mean()*100:.1f}%")
        print(f"  Average Peak Ratio: {peak_ratio_map.mean():.3f}")
        print("="*50)
        
        return fig, {
            'entropy_map': entropy_map,
            'num_peaks_map': num_peaks_map,
            'peak_ratio_map': peak_ratio_map
        }
    
    def compare_scenes(self, unique_img_pair, repetitive_img_pair, 
                      query_point=(16, 16), save_path=None):
        """
        对比独特纹理场景 vs 重复纹理场景的匹配分布差异
        """
        fig, axes = plt.subplots(2, 4, figsize=(18, 9))
        
        for row, (img_a, img_b, title) in enumerate([
            (*unique_img_pair, 'Unique Texture (Building/Text)'),
            (*repetitive_img_pair, 'Repetitive Texture (Farmland)')
        ]):
            sim_matrix, _ = self.extract_similarity_matrix(img_a, img_b)
            N = int(np.sqrt(sim_matrix.shape[0]))
            
            qy, qx = query_point
            query_idx = qy * N + qx
            similarity_2d = sim_matrix[query_idx].reshape(N, N).numpy()
            peaks_info = self._analyze_peaks(similarity_2d)
            
            # 源图
            axes[row, 0].imshow(self._tensor_to_numpy(img_a))
            scale = img_a.shape[-1] / N
            axes[row, 0].scatter([qx * scale], [qy * scale], c='red', s=200, marker='*')
            axes[row, 0].set_title(f'{title}\nSource Image')
            axes[row, 0].axis('off')
            
            # 目标图 + 峰位置
            axes[row, 1].imshow(self._tensor_to_numpy(img_b))
            colors = ['red', 'orange', 'yellow', 'green', 'blue']
            for i, (py, px, pv) in enumerate(peaks_info['peaks'][:5]):
                axes[row, 1].scatter([px * scale], [py * scale], 
                                    c=colors[i], s=100, marker='o')
            axes[row, 1].set_title(f'Target Image\n{peaks_info["num_peaks"]} peaks')
            axes[row, 1].axis('off')
            
            # 热力图
            im = axes[row, 2].imshow(similarity_2d, cmap='hot')
            axes[row, 2].set_title(f'Similarity Map\nEntropy: {peaks_info["entropy"]:.2f}')
            plt.colorbar(im, ax=axes[row, 2], fraction=0.046)
            
            # 3D表面图
            ax3d = fig.add_subplot(2, 4, row * 4 + 4, projection='3d')
            X, Y = np.meshgrid(range(N), range(N))
            ax3d.plot_surface(X, Y, similarity_2d, cmap='hot', alpha=0.8)
            ax3d.set_title('3D Similarity Surface')
            ax3d.set_xlabel('X')
            ax3d.set_ylabel('Y')
            ax3d.set_zlabel('Sim')
            
            # 隐藏原来的2D子图
            axes[row, 3].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        return fig
    
    def _analyze_peaks(self, similarity_2d, threshold_ratio=0.8):
        """
        分析相似度图中的峰
        
        Returns:
            dict: 包含峰数量、位置、熵等信息
        """
        from scipy import ndimage
        from scipy.ndimage import maximum_filter
        
        # 1. 找局部最大值
        max_filtered = maximum_filter(similarity_2d, size=3)
        peaks_mask = (similarity_2d == max_filtered)
        
        # 2. 筛选显著峰（相似度 > 最大值 * threshold_ratio）
        max_val = similarity_2d.max()
        significant_mask = similarity_2d > (max_val * threshold_ratio)
        final_peaks_mask = peaks_mask & significant_mask
        
        # 3. 获取峰的位置和值
        peak_coords = np.where(final_peaks_mask)
        peak_values = similarity_2d[final_peaks_mask]
        
        # 按值排序
        sorted_indices = np.argsort(peak_values)[::-1]
        peaks = [(peak_coords[0][i], peak_coords[1][i], peak_values[i]) 
                 for i in sorted_indices]
        
        # 如果没有找到峰，使用最大值位置
        if len(peaks) == 0:
            max_idx = np.unravel_index(np.argmax(similarity_2d), similarity_2d.shape)
            peaks = [(max_idx[0], max_idx[1], max_val)]
        
        # 4. 计算熵
        prob = F.softmax(torch.from_numpy(similarity_2d.flatten()).float(), dim=0).numpy()
        entropy = -np.sum(prob * np.log(prob + 1e-8))
        
        # 5. 计算峰比（第一峰 / 第二峰的比值，越大说明越确定）
        if len(peaks) >= 2:
            peak_ratio = peaks[0][2] / (peaks[1][2] + 1e-8)
        else:
            peak_ratio = float('inf')
        
        return {
            'num_peaks': len(peaks),
            'peaks': peaks,  # [(y, x, value), ...]
            'entropy': entropy,
            'peak_ratio': min(peak_ratio, 10.0),  # 截断
            'max_value': max_val
        }
    
    def _tensor_to_numpy(self, tensor):
        """将图像张量转换为numpy用于显示"""
        img = tensor[0].cpu().permute(1, 2, 0).numpy()
        # 反归一化 (ImageNet标准)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)
        return img


def create_synthetic_demo():
    """
    创建合成示例演示单峰vs多峰
    无需真实模型，纯数学演示
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    N = 32  # 特征图大小
    
    # ===== 场景1: 单峰（独特纹理）=====
    # 模拟：只有一个明确的匹配位置
    y, x = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))
    
    # 单峰高斯
    peak_y, peak_x = 0.3, -0.2
    single_peak = np.exp(-((x - peak_x)**2 + (y - peak_y)**2) / 0.05)
    single_peak = single_peak / single_peak.max()
    
    axes[0, 0].text(0.5, 0.5, '🏛️\nUnique\nTexture', fontsize=16, ha='center', va='center',
                    transform=axes[0, 0].transAxes)
    axes[0, 0].set_title('Source: Building Corner')
    axes[0, 0].axis('off')
    
    im1 = axes[0, 1].imshow(single_peak, cmap='hot', vmin=0, vmax=1)
    axes[0, 1].scatter([int((peak_x + 1) / 2 * N)], [int((peak_y + 1) / 2 * N)], 
                       c='cyan', s=200, marker='*', edgecolors='white', linewidths=2)
    axes[0, 1].set_title(f'Similarity Map\n(Single Peak)')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    
    # 1D分布
    sorted_single = np.sort(single_peak.flatten())[::-1]
    axes[0, 2].plot(sorted_single[:50], 'b-', linewidth=2)
    axes[0, 2].axhline(y=sorted_single[0], color='r', linestyle='--', label=f'Max: {sorted_single[0]:.2f}')
    axes[0, 2].axhline(y=sorted_single[1], color='orange', linestyle='--', label=f'2nd: {sorted_single[1]:.2f}')
    axes[0, 2].set_title(f'Similarity Distribution\nRatio: {sorted_single[0]/sorted_single[1]:.1f}x')
    axes[0, 2].legend()
    axes[0, 2].set_xlabel('Rank')
    axes[0, 2].set_ylabel('Similarity')
    
    # 3D
    ax3d_1 = fig.add_subplot(2, 4, 4, projection='3d')
    X, Y = np.meshgrid(range(N), range(N))
    ax3d_1.plot_surface(X, Y, single_peak, cmap='hot', alpha=0.9)
    ax3d_1.set_title('3D View: Clear Single Peak')
    ax3d_1.set_zlim(0, 1)
    axes[0, 3].axis('off')
    
    # ===== 场景2: 多峰（重复纹理）=====
    # 模拟：农田中多个相似位置都有高响应
    multi_peak = np.zeros((N, N))
    peak_positions = [(-0.4, -0.4), (-0.4, 0.2), (0.2, -0.4), (0.2, 0.2), (0.4, 0.5)]
    peak_strengths = [1.0, 0.95, 0.92, 0.88, 0.85]
    
    for (py, px), strength in zip(peak_positions, peak_strengths):
        multi_peak += strength * np.exp(-((x - px)**2 + (y - py)**2) / 0.03)
    multi_peak = multi_peak / multi_peak.max()
    
    axes[1, 0].text(0.5, 0.5, '🌾🌾🌾\nRepetitive\nCrop Rows', fontsize=14, ha='center', va='center',
                    transform=axes[1, 0].transAxes, color='green')
    axes[1, 0].set_title('Source: Farmland')
    axes[1, 0].axis('off')
    
    im2 = axes[1, 1].imshow(multi_peak, cmap='hot', vmin=0, vmax=1)
    # 标记所有峰
    colors = ['cyan', 'lime', 'yellow', 'magenta', 'white']
    for i, ((py, px), _) in enumerate(zip(peak_positions, peak_strengths)):
        axes[1, 1].scatter([int((px + 1) / 2 * N)], [int((py + 1) / 2 * N)],
                          c=colors[i], s=150, marker='*', edgecolors='black', linewidths=1)
    axes[1, 1].set_title(f'Similarity Map\n({len(peak_positions)} Peaks!)')
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)
    
    # 1D分布
    sorted_multi = np.sort(multi_peak.flatten())[::-1]
    axes[1, 2].plot(sorted_multi[:50], 'b-', linewidth=2)
    axes[1, 2].axhline(y=sorted_multi[0], color='r', linestyle='--', label=f'Max: {sorted_multi[0]:.2f}')
    axes[1, 2].axhline(y=sorted_multi[1], color='orange', linestyle='--', label=f'2nd: {sorted_multi[1]:.2f}')
    axes[1, 2].set_title(f'Similarity Distribution\nRatio: {sorted_multi[0]/sorted_multi[1]:.1f}x ⚠️')
    axes[1, 2].legend()
    axes[1, 2].set_xlabel('Rank')
    axes[1, 2].set_ylabel('Similarity')
    
    # 3D
    ax3d_2 = fig.add_subplot(2, 4, 8, projection='3d')
    ax3d_2.plot_surface(X, Y, multi_peak, cmap='hot', alpha=0.9)
    ax3d_2.set_title('3D View: Multiple Peaks!')
    ax3d_2.set_zlim(0, 1)
    axes[1, 3].axis('off')
    
    plt.suptitle('Single-Peak vs Multi-Peak Matching Distribution', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('peak_comparison_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*60)
    print("INTERPRETATION:")
    print("="*60)
    print("【Single Peak】 Ratio = {:.1f}x".format(sorted_single[0]/sorted_single[1]))
    print("   → Clear winner, unambiguous matching")
    print("   → Standard attention works well")
    print()
    print("【Multi Peak】  Ratio = {:.1f}x ⚠️".format(sorted_multi[0]/sorted_multi[1]))
    print("   → Multiple candidates with similar scores")
    print("   → Model might pick wrong match!")
    print("   → Needs geometric consistency to disambiguate")
    print("="*60)
    
def load_image(image_path, size=(256, 256)):
    """
    读取并预处理图像为模型所需的张量格式
    """
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    
    # 标准归一化 (ImageNet)
    img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    
    return img_tensor.unsqueeze(0) # [1, 3, H, W]

# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 1. 先运行合成演示（无需模型）
    print("Running synthetic demo...")
    create_synthetic_demo()
    
    # 2. 如果有模型，可以进行真实可视化
    """
    # 加载你的模型
    from network import DenseMatcherTPS
    model = DenseMatcherTPS(...)
    model.load_state_dict(torch.load('checkpoint.pth'))
    
    visualizer = MatchingPeakVisualizer(model)
    
    # 加载图像对
    img_a = load_image('farmland_a.jpg')  # [1, 3, 256, 256]
    img_b = load_image('farmland_b.jpg')
    
    # 可视化特定点
    visualizer.visualize_single_point_matching(
        img_a, img_b,
        query_points=[(8, 8), (16, 16), (24, 24)],  # 特征图坐标
        save_path='peak_analysis.png'
    )
    
    # 可视化全图多峰分布
    visualizer.visualize_full_image_peaks(
        img_a, img_b,
        save_path='full_peak_map.png'
    )
    """