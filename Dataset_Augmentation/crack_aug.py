# import os
# import cv2
# import numpy as np

# # 随机裂纹生成函数
# def generateRandomCracks(image, num_cracks=5, crack_length_range=(10, 50)):
#     augmented_image = image.copy()
#     h, w = image.shape[:2]

#     for _ in range(num_cracks):
#         start_point = (np.random.randint(0, w), np.random.randint(0, h))
#         angle = np.random.uniform(0, 2 * np.pi)
#         crack_length = np.random.randint(*crack_length_range)
#         end_point = (int(start_point[0] + crack_length * np.cos(angle)), 
#                     int(start_point[1] + crack_length * np.sin(angle)))

#         thickness = np.random.randint(1, 3)
#         color = (0, 0, 0)  # 假设裂纹为黑色
#         cv2.line(augmented_image, start_point, end_point, color, thickness)

#     return augmented_image

# # 数据集路径和保存路径
# input_dir = '/mnt/IAD_datasets/die/train/good'
# output_dir = '/mnt/IAD_datasets/die/augmented/crack1'

# # 创建保存路径（如果不存在）
# os.makedirs(output_dir, exist_ok=True)

# # 处理数据集中的图像
# for filename in os.listdir(input_dir):
#     if filename.endswith('.png') or filename.endswith('.jpg'):  # 根据你的数据集格式调整
#         img_path = os.path.join(input_dir, filename)
#         image = cv2.imread(img_path)

#         # 生成随机裂纹
#         augmented_image = generateRandomCracks(image)

#         # 保存增强后的图像
#         output_path = os.path.join(output_dir, filename)
#         cv2.imwrite(output_path, augmented_image)

# print(f"图像增强完成，增强后的图像已保存到 {output_dir}")

# 利用图像合成生成裂纹
import cv2
import numpy as np
import os

def add_synthetic_cracks(image, crack_image, alpha=0.7):
    # Resize the crack image to match the original image size
    crack_image = cv2.resize(crack_image, (image.shape[1], image.shape[0]))
    
    # Convert crack image to grayscale if it's not already
    if len(crack_image.shape) == 3:
        crack_image = cv2.cvtColor(crack_image, cv2.COLOR_BGR2GRAY)
    
    # Invert crack image
    crack_image = cv2.bitwise_not(crack_image)
    
    # Apply threshold to create binary crack mask
    _, crack_mask = cv2.threshold(crack_image, 200, 255, cv2.THRESH_BINARY)
    
    # Convert mask to three channels
    crack_mask_colored = cv2.cvtColor(crack_mask, cv2.COLOR_GRAY2BGR)
    
    # Blend the original image with the crack mask
    augmented_image = cv2.addWeighted(image, 1 - alpha, crack_mask_colored, alpha, 0)
    
    return augmented_image

def augment_with_crack_overlay(input_dir, crack_image_path, output_dir, alpha=0.7):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    crack_image = cv2.imread(crack_image_path, cv2.IMREAD_UNCHANGED)
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            
            augmented_img = add_synthetic_cracks(img, crack_image, alpha)
            
            save_path = os.path.join(output_dir, filename)
            cv2.imwrite(save_path, augmented_img)
            print(f"Saved augmented image to {save_path}")

# Define the paths
input_dir = '/mnt/IAD_datasets/die/train/good'
output_dir = '/mnt/IAD_datasets/die/augmented/crack'
crack_image_path = '/path/to/crack_pattern.png'  # Replace with your crack pattern image path

# Run the augmentation process
augment_with_crack_overlay(input_dir, crack_image_path, output_dir, alpha=0.7)
