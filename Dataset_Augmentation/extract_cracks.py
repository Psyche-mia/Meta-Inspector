import cv2
import numpy as np
import os

def extract_crack(image):
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 应用高斯模糊去噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 使用Canny边缘检测
    edges = cv2.Canny(blurred, 50, 150)
    
    # 找到裂纹的轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 创建一个空白图像，填充裂纹轮廓
    crack_mask = np.zeros_like(gray)
    cv2.drawContours(crack_mask, contours, -1, (255), thickness=cv2.FILLED)
    
    return crack_mask

def process_crack_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            
            crack_mask = extract_crack(img)
            
            save_path = os.path.join(output_dir, f"crack_{filename}")
            cv2.imwrite(save_path, crack_mask)
            print(f"Saved extracted crack image to {save_path}")

# 定义输入图像和输出路径
input_dir = '/mnt/IAD_datasets/die/test/crack'
output_dir = '/mnt/IAD_datasets/die/crack_patterns'

# 运行裂纹提取处理
process_crack_images(input_dir, output_dir)
