#!/usr/bin/env python3
"""
从分割的字符图像生成训练数据的脚本
"""

import os
import cv2
import numpy as np
from typing import List, Tuple
import re


def create_manual_training_data(characters_dir: str = 'output/characters', 
                               output_script: str = 'create_training_data.py'):
    """
    从分割的字符图像创建训练数据脚本
    
    Args:
        characters_dir: 字符图像目录
        output_script: 输出训练脚本路径
    """
    # 获取所有字符图像文件
    image_files = []
    for ext in ['.png', '.bmp', '.jpg', '.jpeg']:
        for root, dirs, files in os.walk(characters_dir):
            for file in files:
                if file.lower().endswith(ext):
                    image_files.append(os.path.join(root, file))
    
    print(f"找到 {len(image_files)} 个字符图像文件")
    
    # 生成训练数据脚本
    script_content = f'''#!/usr/bin/env python3
"""
自动生成的SVM训练数据脚本
"""

import os
import cv2
from ocr_svm import train_svm_model

# 训练数据
training_data = [
'''
    
    for img_path in sorted(image_files):
        filename = os.path.basename(img_path)
        # 尝试从文件名中提取字符标签，例如 "000_A.bmp" -> "A"
        match = re.search(r'_([A-Za-z0-9])\.', filename)
        if match:
            label = match.group(1)
            def create_manual_training_data(characters_dir: str = 'output/characters', 
                               output_script: str = 'create_training_data.py'):
    """
    从分割的字符图像创建训练数据脚本
    
    Args:
        characters_dir: 字符图像目录
        output_script: 输出训练脚本路径
    """
    # 检查是否是patch目录结构（按字符分类的子目录）
    subdirs = [d for d in os.listdir(characters_dir) 
               if os.path.isdir(os.path.join(characters_dir, d))]
    
    image_files = []
    
    # 如果子目录名称是单个字符，则认为是patch目录结构
    single_char_dirs = [d for d in subdirs if len(d) == 1 and d.isalnum()]
    
    if single_char_dirs:
        print(f"检测到patch目录结构，找到 {len(single_char_dirs)} 个字符类别")
        for char_dir in sorted(single_char_dirs):
            char_path = os.path.join(characters_dir, char_dir)
            for ext in ['.png', '.bmp', '.jpg', '.jpeg']:
                for file in os.listdir(char_path):
                    if file.lower().endswith(ext):
                        img_path = os.path.join(char_path, file)
                        image_files.append((img_path, char_dir))  # (路径, 标签)
                        print(f"  找到 {char_dir}: {file}")
    else:
        # 获取所有字符图像文件（旧的命名格式）
        for ext in ['.png', '.bmp', '.jpg', '.jpeg']:
            for root, dirs, files in os.walk(characters_dir):
                for file in files:
                    if file.lower().endswith(ext):
                        img_path = os.path.join(root, file)
                        image_files.append((img_path, None))  # (路径, 未标注)
    
    print(f"找到 {len(image_files)} 个字符图像文件")
    
    # 生成训练数据脚本
    script_content = f'''#!/usr/bin/env python3
"""
自动生成的SVM训练数据脚本
"""

import os
import cv2
from ocr_svm import train_svm_model

# 训练数据
training_data = [
'''
    
    for img_path, label in image_files:
        filename = os.path.basename(img_path)
        if label is not None:
            # 对于patch目录结构，标签已经确定
            script_content += f'    ("{img_path}", "{label}"),  # {filename}
'
        else:
            # 尝试从文件名中提取字符标签，例如 "000_A.bmp" -> "A"
            match = re.search(r'_([A-Za-z0-9])\.', filename)
            if match:
                label = match.group(1)
                script_content += f'    ("{img_path}", "{label}"),  # {filename}
'
            else:
                # 如果无法从文件名推断标签，使用占位符
                script_content += f'    ("{img_path}", "?"),  # {filename} - 需要手动标注
'
    
    script_content += ''']

def main():
    print("开始加载训练数据...")
    
    images = []
    labels = []
    
    for img_path, label in training_data:
        if label == "?":
            print(f"跳过待标注的图像: {img_path}")
            continue  # 跳过未标注的图像
            
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
            labels.append(label)
            print(f"加载: {img_path} -> {label}")
        else:
            print(f"无法加载图像: {img_path}")
    
    if len(images) == 0:
        print("没有有效的训练数据!")
        return
    
    print(f"
开始训练SVM模型...")
    print(f"训练样本数: {{len(images)}}")
    print(f"类别数: {{len(set(labels))}}")
    print(f"类别: {{sorted(set(labels))}}")
    
    train_svm_model(images, labels)
    print("
SVM模型训练完成!")

if __name__ == "__main__":
    main()
'''

    # 写入脚本文件
    with open(output_script, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print(f"训练数据脚本已生成: {output_script}")
    print(f"请编辑该脚本以修正标签，然后运行: python {output_script}")
    
    script_content += ''']

def main():
    print("开始加载训练数据...")
    
    images = []
    labels = []
    
    for img_path, label in training_data:
        if label == "?":
            print(f"跳过待标注的图像: {img_path}")
            continue  # 跳过未标注的图像
            
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
            labels.append(label)
            print(f"加载: {img_path} -> {label}")
        else:
            print(f"无法加载图像: {img_path}")
    
    if len(images) == 0:
        print("没有有效的训练数据!")
        return
    
    print(f"\n开始训练SVM模型...")
    print(f"训练样本数: {{len(images)}}")
    print(f"类别数: {{len(set(labels))}}")
    print(f"类别: {{sorted(set(labels))}}")
    
    train_svm_model(images, labels)
    print("\nSVM模型训练完成!")

if __name__ == "__main__":
    main()
'''

    # 写入脚本文件
    with open(output_script, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print(f"训练数据脚本已生成: {output_script}")
    print(f"请编辑该脚本以修正标签，然后运行: python {output_script}")


def main():
    print("字符图像训练数据生成器")
    print("=" * 40)
    
    characters_dir = input("请输入字符图像目录 (默认: output/characters): ").strip()
    if not characters_dir:
        characters_dir = 'output/characters'
    
    output_script = input("请输入输出脚本名称 (默认: create_training_data.py): ").strip()
    if not output_script:
        output_script = 'create_training_data.py'
    
    if not os.path.exists(characters_dir):
        print(f"目录不存在: {characters_dir}")
        return
    
    create_manual_training_data(characters_dir, output_script)


if __name__ == "__main__":
    main()
