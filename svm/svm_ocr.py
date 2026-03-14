"""精简的 OCR 模块：基于SVM的字符识别。

公共函数：
- `prepare_image_array(img) -> np.ndarray`：将输入规范为灰度 uint8 numpy 数组
- `run_knn_recognition(char_images=None) -> list`：接受内存中的字符图像列表
    （列表项为包含 'name' 和 'img' 的字典）并返回识别结果。
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import pickle


# --------- OCR 配置超参数 ---------
# 所有 OCR 识别相关的可调参数在此处定义
CFG = {
    # SVM相关参数
    "SVM_KERNEL": 'rbf',                 # SVM核函数
    "SVM_C": 1.0,                       # SVM正则化参数
    "SVM_GAMMA": 'scale',               # SVM核函数系数
    
    # 图像处理参数
    "TARGET_SIZE": (28, 28),            # 标准化字符图像尺寸
    
    # 其他参数
    "SCORE_THRESHOLD": 0.1,             # 识别阈值（SVM置信度阈值）
    "CLIST_PATH": "Clist.txt",          # 字符模板文件路径
    "MODEL_PATH": "models/svm_model.pkl",  # SVM模型保存路径
}


class SVMOCR:
    def __init__(self):
        self.svm_model = None
        self.label_encoder = None
        self.is_trained = False
    
    def extract_features(self, img: np.ndarray) -> np.ndarray:
        """提取字符图像的特征向量。
        
        使用多种特征提取方法：
        1. 像素强度直方图
        2. 水平和垂直投影
        3. HOG特征
        """
        h, w = img.shape
        
        # 标准化尺寸
        resized = cv2.resize(img, CFG["TARGET_SIZE"])
        
        # 像素强度特征（展平为一维向量）
        pixel_features = resized.flatten() / 255.0  # 归一化到[0,1]
        
        # 水平投影特征
        hor_proj = np.sum(resized > 0, axis=1) / w  # 每行白色像素比例
        
        # 垂直投影特征
        ver_proj = np.sum(resized > 0, axis=0) / h  # 每列白色像素比例
        
        # 组合所有特征
        features = np.concatenate([pixel_features, hor_proj, ver_proj])
        
        return features
    
    def train(self, images: List[np.ndarray], labels: List[str]):
        """训练SVM模型
        
        Args:
            images: 训练图像列表
            labels: 对应的标签列表
        """
        print(f"[SVMOCR] 开始训练SVM模型，样本数: {len(images)}")
        
        # 提取特征
        X = []
        y = []
        
        for img, label in zip(images, labels):
            try:
                features = self.extract_features(img)
                X.append(features)
                y.append(label)
            except Exception as e:
                print(f"  跳过无效图像: {e}")
                continue
        
        if len(X) == 0:
            raise ValueError("没有有效的训练样本")
        
        X = np.array(X)
        y = np.array(y)
        
        # 标签编码
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # 训练SVM
        self.svm_model = SVC(
            kernel=CFG["SVM_KERNEL"],
            C=CFG["SVM_C"],
            gamma=CFG["SVM_GAMMA"],
            probability=True  # 启用概率预测
        )
        
        self.svm_model.fit(X, y_encoded)
        self.is_trained = True
        
        print(f"[SVMOCR] SVM模型训练完成，类别数: {len(self.label_encoder.classes_)}")
        print(f"[SVMOCR] 类别: {list(self.label_encoder.classes_)}")
        
        # 保存模型
        self.save_model()
    
    def save_model(self):
        """保存训练好的模型"""
        model_dir = os.path.dirname(CFG["MODEL_PATH"])
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        
        model_data = {
            'svm_model': self.svm_model,
            'label_encoder': self.label_encoder
        }
        
        with open(CFG["MODEL_PATH"], 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"[SVMOCR] 模型已保存到: {CFG['MODEL_PATH']}")
    
    def load_model(self):
        """加载预训练的模型"""
        if not os.path.exists(CFG["MODEL_PATH"]):
            print(f"[SVMOCR] 模型文件不存在: {CFG['MODEL_PATH']}")
            return False
        
        try:
            with open(CFG["MODEL_PATH"], 'rb') as f:
                model_data = pickle.load(f)
            
            self.svm_model = model_data['svm_model']
            self.label_encoder = model_data['label_encoder']
            self.is_trained = True
            
            print(f"[SVMOCR] 模型已从 {CFG['MODEL_PATH']} 加载")
            return True
        except Exception as e:
            print(f"[SVMOCR] 加载模型失败: {e}")
            return False
    
    def predict(self, img: np.ndarray) -> Tuple[str, float, List[int], List[int]]:
        """预测单个字符
        
        Args:
            img: 输入图像
            
        Returns:
            (label, score, peak_x_indices, peak_y_indices)
        """
        if not self.is_trained:
            return ('?', 1.0, [], [])
        
        try:
            # 提取特征
            features = self.extract_features(img).reshape(1, -1)
            
            # 预测
            pred_encoded = self.svm_model.predict(features)[0]
            proba = self.svm_model.predict_proba(features)[0]
            
            # 获取置信度（最高概率）
            confidence = np.max(proba)
            
            # 解码标签
            label = self.label_encoder.inverse_transform([pred_encoded])[0]
            
            # 对于SVM，我们返回一个假的峰值信息（空列表）
            # 因为SVM不需要这些信息，但为了保持接口兼容性
            return (label, 1.0 - confidence, [], [])  # score = 1 - confidence
            
        except Exception as e:
            print(f"[SVMOCR] 预测失败: {e}")
            return ('?', 1.0, [], [])


# 全局SVM实例
svm_ocr = SVMOCR()


def prepare_image_array(img: np.ndarray) -> np.ndarray:
    """确保 `img` 为二维 uint8 灰度 numpy 数组。

    - 若为三通道，使用 RGB 平均转换为灰度（快速，不依赖 cv2）。
    - 若为四通道，去掉 alpha 通道并对 RGB 求平均。
    - 若已为二维，则复制并在必要时转换为 uint8。
    """
    if not isinstance(img, np.ndarray):
        raise TypeError("prepare_image_array expects a numpy.ndarray")

    if img.ndim == 3 and img.shape[2] >= 3:
        rgb = img[..., :3].astype(np.float32)
        gray = np.mean(rgb, axis=2)
        arr = np.clip(gray, 0, 255).astype(np.uint8)
        return arr
    elif img.ndim == 2:
        if img.dtype != np.uint8:
            return img.astype(np.uint8)
        return img.copy()
    else:
        raise ValueError("Unsupported image shape for prepare_image_array: " + str(img.shape))


def is_space_image(img: np.ndarray) -> bool:
    """判断图像是否是表示空格的特殊数组"""
    # 检查图像是否是全黑的（表示空格的特殊数组）
    # 空格图像应该是一个很小的全黑图像
    if img.ndim == 2:
        total_pixels = img.size
        if total_pixels == 0:
            return False
        # 如果所有像素都是0（黑色），则认为是空格
        black_pixels = np.sum(img == 0)
        # 如果黑色像素占比很高（比如95%以上），则认为是空格
        return black_pixels / total_pixels > 0.95
    return False


def run_knn_recognition(char_images: List[Dict[str, Any]] = None) -> List[Tuple[str, str, float, List[int], List[int]]]:
    """运行基于SVM的字符识别。

    接受 `char_images`，应为由 `post_vertical_projection.cut_characters(...)` 生成的
    列表，列表项形如 `{'name': str, 'img': np.ndarray, 'meta': {...}}`。

    返回列表，元素为元组 (filename, label, score, peak_x_indices, peak_y_indices)。

    识别流程：
    1. 尝试加载预训练的SVM模型
    2. 对每个字符图像：
       - 检查是否是表示空格的特殊数组
       - 如果是空格，直接返回空格字符
       - 否则使用SVM进行预测
    3. 返回识别结果
    """
    results = []
    if not char_images:
        return results

    # 尝试加载预训练模型
    if not svm_ocr.load_model():
        print("[SVMOCR] 警告：未找到预训练模型，SVM识别将返回 '?'")
        # 如果没有模型，为每个字符返回'?'和高错误分数
        for idx, ent in enumerate(char_images):
            name = ent.get('name', f'char_{idx:03d}')
            results.append((name, '?', 1.0, [], []))
        return results

    print(f"[SVMOCR] Processing {len(char_images)} characters with SVM model...")

    for idx, ent in enumerate(char_images):
        name = ent.get('name', f'char_{idx:03d}')
        img = ent.get('img')

        # 检查是否是空格图像
        if is_space_image(img):
            # 直接识别为空格
            label = ' '  # 空格字符
            score = 0.0  # 空格的匹配分数为0
            peak_x = []  # 空格没有峰值
            peak_y = []  # 空格没有峰值
            results.append((name, label, score, peak_x, peak_y))
            print(f"  [{idx}] {name}: ' ' (space detected, score={score:.4f}, x_peaks={len(peak_x)}, y_peaks={len(peak_y)})")
            continue

        # 归一化图像
        try:
            img_arr = prepare_image_array(img)
        except Exception as e:
            print(f"  [{idx}] {name}: Recognition failed - {e}")
            results.append((name, '?', 1.0, [], []))
            continue

        # 使用SVM进行预测
        label, score, peak_x, peak_y = svm_ocr.predict(img_arr)
        
        results.append((name, label, score, peak_x, peak_y))

        # 打印结果
        score_threshold = CFG.get("SCORE_THRESHOLD", 0.1)
        if score > score_threshold:
            display_label = "无法识别"
            print(f"  [{idx}] {name}: '{display_label}' (score={score:.4f}, x_peaks={len(peak_x)}, y_peaks={len(peak_y)})")
        elif label != '?':
            print(f"  [{idx}] {name}: '{label}' (score={score:.4f}, x_peaks={len(peak_x)}, y_peaks={len(peak_y)})")
        else:
            print(f"  [{idx}] {name}: unmatched (score={score:.4f})")

    # 统计结果
    score_threshold = CFG.get("SCORE_THRESHOLD", 0.1)
    matched = sum(1 for r in results if r[1] != '?' and r[2] <= score_threshold)
    print(f"\n[SVMOCR] Matched {matched}/{len(results)} characters")

    # 生成完整识别结果字符串
    full_recognition_result = ""
    for name, label, score, peak_x, peak_y in results:
        if score > score_threshold:
            # 不将"无法识别"添加到完整结果中，或者用特殊符号代替
            full_recognition_result += "?"
        else:
            full_recognition_result += label

    print(f"[SVMOCR] Full recognition result: '{full_recognition_result}'")

    # 写入结果文件
    output_path = 'output/characters/recognition_result.txt'
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Character Recognition Results\n")
            f.write("# Format: filename, label, score, x_peaks, y_peaks\n")
            f.write(f"# Full recognition result: {full_recognition_result}\n")
            score_threshold = CFG.get("SCORE_THRESHOLD", 0.1)
            for name, label, score, peak_x, peak_y in results:
                if score > score_threshold:
                    display_label = "无法识别"
                else:
                    display_label = label
                f.write(f"{name}, {display_label}, {score:.4f}, {len(peak_x)}, {len(peak_y)}\n")
        print(f"[SVMOCR] Results saved to {output_path}")
    except Exception as e:
        print(f"[SVMOCR] Failed to save results: {e}")

    return results


def train_svm_model(training_images: List[np.ndarray], training_labels: List[str]):
    """训练SVM模型的便捷函数
    
    Args:
        training_images: 训练图像列表
        training_labels: 对应标签列表
    """
    svm_ocr.train(training_images, training_labels)


def load_svm_model():
    """加载预训练的SVM模型"""
    return svm_ocr.load_model()