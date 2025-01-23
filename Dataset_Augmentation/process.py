import cv2
import numpy as np
import os

def refine_crack_mask(crack_mask, min_size=500):
    # 使用开运算去除细小的噪声和边缘
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    refined_mask = cv2.morphologyEx(crack_mask, cv2.MORPH_OPEN, kernel)
    
    # 进一步去除小的连接组件（不是裂纹的部分）
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(refined_mask, connectivity=8)
    
    filtered_mask = np.zeros_like(crack_mask)
    
    for i in range(1, num_labels):  # 从1开始，忽略背景（0）
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            filtered_mask[labels == i] = 255
    
    return filtered_mask

def process_and_refine_crack_images(crack_dir, output_dir, min_size=500):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    crack_files = sorted(os.listdir(crack_dir))
    
    for filename in crack_files:
        if filename.endswith('.png') or filename.endswith('.jpg'):
            crack_path = os.path.join(crack_dir, filename)
            crack_mask = cv2.imread(crack_path, cv2.IMREAD_GRAYSCALE)
            
            refined_crack_mask = refine_crack_mask(crack_mask, min_size)
            
            save_path = os.path.join(output_dir, filename)
            cv2.imwrite(save_path, refined_crack_mask)
            print(f"Saved refined crack mask to {save_path}")

# 定义裂纹掩码路径和输出路径
crack_dir = '/mnt/IAD_datasets/die/crack_patterns'  # 原始裂纹掩码文件夹
output_dir = '/mnt/IAD_datasets/die/refined_crack_patterns'  # 保存去除边缘后的裂纹掩码

# 运行裂纹掩码处理过程
process_and_refine_crack_images(crack_dir, output_dir, min_size=500)
