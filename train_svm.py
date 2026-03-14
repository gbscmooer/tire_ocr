#!/usr/bin/env python3
"""
SVM模型训练脚本
用于从Clist.txt和字符图像训练SVM模型
"""

import os
import cv2
import numpy as np
from ocr import train_svm_model
import glob
from typing import List, Tuple

# 导入 ocr.py 中的 load_clist 函数
try:
    from pure.pure_ocr import load_clist
except ImportError:
    # 如果 ocr.py 不存在或没有 load_clist 函数，定义一个简单的替代函数
    def load_clist(clist_path: str = 'Clist.txt'):
        """简单实现 load_clist 函数"""
        templates = {}
        if not os.path.exists(clist_path):
            print(f"[load_clist] Warning: Clist.txt not found at {clist_path}")
            return templates

        try:
            with open(clist_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    # 解析格式: 3:((2:2:3:4:3),(5:1:1:1:1:2:3))
                    if ':' not in line:
                        continue
                    parts = line.split(':', 1)
                    label = parts[0].strip()
                    templates[label] = {'x': [], 'y': []}  # 简化版本
        except Exception as e:
            print(f"[load_clist] Error parsing Clist.txt: {e}")

        print(f"[load_clist] Loaded {len(templates)} character templates")
        return templates


def load_training_data_from_clist_and_images(clist_path: str = 'Clist.txt', 
                                            images_dir: str = 'patch') -> Tuple[List[np.ndarray], List[str]]:
    """
    从Clist.txt和字符图像目录加载训练数据
    
    Args:
        clist_path: Clist.txt文件路径
        images_dir: 字符图像目录
    
    Returns:
        (images, labels) 元组
    """
    # 从Clist.txt加载标签信息
    templates = load_clist(clist_path)
    if not templates:
        print(f"警告: 无法从 {clist_path} 加载模板")
        return [], []
    
    # 获取字符标签列表
    expected_labels = list(templates.keys())
    print(f"从Clist.txt找到 {len(expected_labels)} 个字符模板: {expected_labels}")
    
    # 检查是否是patch目录结构（按字符分类的子目录）
    if not os.path.exists(images_dir):
        print(f"错误: 目录不存在 {images_dir}")
        return [], []
        
    subdirs = [d for d in os.listdir(images_dir) 
               if os.path.isdir(os.path.join(images_dir, d))]
    
    images = []
    labels = []
    
    # 如果子目录名称是单个字符或数字，则认为是patch目录结构
    single_char_dirs = [d for d in subdirs if (len(d) == 1 and (d.isalnum())) or d == '-']
    
    if single_char_dirs:
        print(f"检测到patch目录结构，处理 {len(single_char_dirs)} 个字符类别: {single_char_dirs}")
        for char_dir in sorted(single_char_dirs):
            # 检查该字符是否在Clist中
            if char_dir in expected_labels:
                char_path = os.path.join(images_dir, char_dir)
                # 获取该目录下的所有图像
                for ext in ['*.png', '*.bmp', '*.jpg', '*.jpeg']:
                    char_image_files = glob.glob(os.path.join(char_path, ext))
                    char_image_files.extend(glob.glob(os.path.join(char_path, ext.upper())))
                    
                    for img_path in char_image_files:
                        try:
                            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                            if img is not None:
                                images.append(img)
                                labels.append(char_dir)
                                print(f"  加载 {char_dir}: {os.path.basename(img_path)}")
                        except Exception as e:
                            print(f"  跳过无效图像 {img_path}: {e}")
            else:
                print(f"  跳过未知字符类别: {char_dir}")
    else:
        # 使用旧的文件组织方式
        image_files = []
        for ext in ['*.png', '*.bmp', '*.jpg', '*.jpeg']:
            image_files.extend(glob.glob(os.path.join(images_dir, ext)))
            image_files.extend(glob.glob(os.path.join(images_dir, ext.upper())))
        
        for img_path in image_files:
            # 从文件名推断标签（例如：000_A.bmp -> A）
            filename = os.path.basename(img_path)
            parts = filename.split('_')
            if len(parts) >= 2:
                # 取第二个部分作为标签（去掉扩展名）
                label = parts[1].split('.')[0]
                
                # 只处理在Clist中的标签
                if label in expected_labels:
                    try:
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            images.append(img)
                            labels.append(label)
                            print(f"  加载: {filename} -> {label}")
                    except Exception as e:
                        print(f"  跳过无效图像 {img_path}: {e}")
                else:
                    print(f"  跳过未知标签 {label} 在文件 {filename}")
            else:
                print(f"  无法从文件名解析标签: {filename}")
    
    print(f"共加载 {len(images)} 个训练样本")
    return images, labels


def load_training_data_from_manual_pairs(image_label_pairs: List[Tuple[str, str]]) -> Tuple[List[np.ndarray], List[str]]:
    """
    从图像路径和标签对列表加载训练数据
    
    Args:
        image_label_pairs: [(image_path, label), ...] 列表
    
    Returns:
        (images, labels) 元组
    """
    images = []
    labels = []
    
    for img_path, label in image_label_pairs:
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                labels.append(label)
                print(f"  加载: {img_path} -> {label}")
            else:
                print(f"  无法加载图像: {img_path}")
        except Exception as e:
            print(f"  跳过无效图像 {img_path}: {e}")
    
    return images, labels


def main():
    print("SVM OCR 模型训练器")
    print("=" * 50)
    
    print("从 patch 目录加载训练数据...")
    images, labels = load_training_data_from_clist_and_images()
    
    if not images:
        print("没有加载到任何训练数据!")
        return
    
    if len(set(labels)) < 2:
        print("需要至少2个不同的字符类别进行训练!")
        return
    
    print(f"\n开始训练SVM模型...")
    print(f"训练样本数: {len(images)}")
    print(f"类别数: {len(set(labels))}")
    print(f"类别: {sorted(set(labels))}")
    
    try:
        # 训练模型
        train_svm_model(images, labels)
        print("\nSVM模型训练完成!")
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
