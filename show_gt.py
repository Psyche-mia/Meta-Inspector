from PIL import Image
import numpy as np
import os

def process_binary_mask(input_path, output_path, threshold=128):
    """
    处理二值掩码图像，确保其为二值图像，并保存处理后的图像。

    :param input_path: 输入掩码图像的路径
    :param output_path: 输出处理后图像的路径
    :param threshold: 用于二值化的阈值（默认 128）
    """
    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        print(f"输入文件不存在: {input_path}")
        return

    try:
        # 打开图像文件
        img = Image.open(input_path)
        print(f"打开图像: {input_path}")

        # 确认图像模式为 'L'（灰度）
        if img.mode != 'L':
            print(f"图像模式不是 'L'，当前模式: {img.mode}。转换为 'L' 模式。")
            img = img.convert('L')
        else:
            print("图像模式为 'L' (灰度模式)。")
        
        # 转换为 NumPy 数组
        img_np = np.array(img)
        print(img_np.shape)
        print(img_np)
        # 打印原始图像的唯一像素值
        unique_values = np.unique(img_np)
        print(f"原始图像的唯一像素值: {unique_values}")

        # 检查是否已经是二值图像（只有两个唯一值）
        if len(unique_values) > 2:
            print("图像不是二值的。应用阈值处理以转换为二值图像。")
            # 应用阈值进行二值化
            binary_np = np.where(img_np > threshold, 255, 0).astype(np.uint8)
        else:
            print("图像已经是二值的。无需进一步处理。")
            binary_np = img_np

        # 打印处理后图像的唯一像素值
        unique_after = np.unique(binary_np)
        print(f"处理后图像的唯一像素值: {unique_after}")

        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"创建输出目录: {output_dir}")

        # 转换回 PIL 图像并保存
        binary_img = Image.fromarray(binary_np, mode='L')
        binary_img.save(output_path)
        print(f"处理后的二值掩码已保存到: {output_path}")

    except Exception as e:
        print(f"处理图像时发生错误: {e}")

# 使用示例
input_mask_path = "gt_mask/000_mask.png"
output_mask_path = "gt_mask/000_mask_processed.png"
process_binary_mask(input_mask_path, output_mask_path)
