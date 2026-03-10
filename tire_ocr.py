import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import random

class TireOCR:
    """橡胶胎面点阵字符识别主类"""
    
    def __init__(self, output_dir: str = "output"):
        """
        初始化 OCR 处理器
        Args:
            output_dir: 输出目录，保存中间结果
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 存储中间结果
        self.original_image = None          # 原始图像
        self.binary_image = None            # 二值化图像
        self.tread_roi_image = None         # 胎面区域图像（黑色区域）
        self.enhanced_image = None          # 增强后图像
        self.tophat_image = None            # Top-Hat 变换结果
        self.morph_image = None             # 形态学处理结果
        self.segmented_chars = []           # 分割后的字符图像列表
        self.char_positions = []            # 字符在原图中的位置
        self.label_roi_image = None         # 标签区域图像（白色长条）
        self.label_binary = None            # 标签二值化图像
        
        # 编号区域定位信息
        self.label_left_global = None       # 编号区域在原始图像中的左边界
        self.label_right_global = None      # 编号区域在原始图像中的右边界
        self.tread_left_global = None       # 胎面区域在原始图像中的左边界
        self.tread_right_global = None      # 胎面区域在原始图像中的右边界
        self.label_left_relative = None     # 编号区域在胎面ROI中的相对左偏移
        self.label_right_relative = None    # 编号区域在胎面ROI中的相对右偏移

        # 字符识别相关
        self.classifier = None              # 训练好的分类器 (SVM)
        self.hog_descriptor = None          # HOG 特征描述器
        self.class_labels = []              # 类别标签
        self.is_trained = False
    
    # ==================== 第一阶段：ROI 提取与预处理 ====================
    def stage1_extract_roi(self, image_path: str) -> np.ndarray:
        """
        第一阶段：ROI 提取与预处理
        1. 全局二值化（Otsu）得到黑白图像（胎面为黑0，背景为白255）
        2. 从图像中心 x = 宽度/2 向两边扫描，找到黑色胎面区域的左右边界
        3. 提取胎面 ROI 并灰度化 + CLAHE 增强
        
        Args:
            image_path: 输入图像路径
        
        Returns:
            胎面 ROI 增强后的图像
        """
        # 读取图像
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise FileNotFoundError(f"无法读取图像: {image_path}")
        
        # 1. 全局二值化（Otsu）
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        _, self.binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 2. 使用中心向左扫描找到编号区域（白色文字区域），然后基于编号区域锁定整个胎面区域
        height, width = self.binary_image.shape
        center_x = width // 2
        
        # 第一步：找到编号区域（白色文字区域）的左右边界
        label_left = None
        label_right = None
        current_x = center_x
        
        # 状态机：0-寻找右边界，1-寻找左边界
        state = 0
        consecutive_label_cols = 0
        consecutive_tread_cols = 0
        
        while current_x > 0:
            black_pixels = np.sum(self.binary_image[:, current_x] == 0)
            black_ratio = black_pixels / height
            
            if state == 0:
                if black_ratio < 0.05:
                    # 遇到无关白色边缘，说明没有编号区域
                    print("  未发现编号区域 (遇到空白边缘)")
                    return None
                elif 0.10 <= black_ratio <= 0.90:
                    # 找到可能的编号区域右边界
                    consecutive_label_cols += 1
                    if consecutive_label_cols >= 3:  # 连续3列都在标签范围内才确认
                        label_right = current_x + 2  # 取最早的一列
                        state = 1
                        print(f"  发现编号区域右边界: {label_right}, 连续{consecutive_label_cols}列")
                else:
                    consecutive_label_cols = 0  # 重置计数
                    
            elif state == 1:
                if black_ratio > 0.98:
                    consecutive_tread_cols += 1
                    
                    # 增加容忍度机制：连续5个像素列整体>98%才确认为左边界
                    if consecutive_tread_cols >= 5:
                        # 检查这5列的平均黑色像素比例
                        check_start = current_x
                        check_end = current_x + 4  # 包括当前列和前4列
                        total_black_ratio = 0
                        check_count = 0
                        
                        # 检查范围是否有效
                        for check_x in range(check_start, check_end + 1):
                            if check_x >= 0:
                                check_black_pixels = np.sum(self.binary_image[:, check_x] == 0)
                                check_black_ratio = check_black_pixels / height
                                total_black_ratio += check_black_ratio
                                check_count += 1
                        
                        if check_count > 0:
                            avg_black_ratio = total_black_ratio / check_count
                            if avg_black_ratio > 0.98:
                                label_left = current_x + 2  # 取最早的一列
                                print(f"  发现编号区域左边界: {label_left}, 连续{consecutive_tread_cols}列，平均黑色像素比例={avg_black_ratio:.3f}")
                                break
                else:
                    consecutive_tread_cols = 0  # 重置计数
                    
            current_x -= 1
        
        # 如果没找到编号区域，尝试更宽松的阈值
        if label_left is None or label_right is None:
            print("  未完整发现编号区域界限，尝试宽松阈值...")
            # 重置状态
            label_left = None
            label_right = None
            current_x = center_x
            state = 0
            while current_x > 0:
                black_pixels = np.sum(self.binary_image[:, current_x] == 0)
                black_ratio = black_pixels / height
                
                if state == 0:
                    if black_ratio < 0.05:
                        print("  宽松模式: 遇到空白边缘")
                        return None
                    elif 0.05 <= black_ratio <= 0.95:  # 放宽阈值
                        if label_right is None:
                            label_right = current_x
                            state = 1
                elif state == 1:
                    if black_ratio > 0.95:  # 放宽阈值
                        label_left = current_x
                        break
                        
                current_x -= 1
        
        if label_left is None or label_right is None:
            print("  未完整发现编号区域界限")
            return None
            
        print(f"  编号区域: 左边界={label_left}, 右边界={label_right}, 宽度={label_right-label_left}")
        
        # 保存编号区域在原始图像中的位置
        self.label_left_global = label_left
        self.label_right_global = label_right
        
        # 第二步：基于编号区域向内扩展定位整个胎面区域
        # 我们希望胎面区域只包含编号区域周围的部分，而不是整个胎面
        # 所以应该基于编号区域向内缩小范围，而不是向外扩展到整个胎面
        
        # 计算基于编号区域向内扩展的宽度
        # 编号区域宽度通常较小，我们向两边扩展固定宽度
        expand_left = 50  # 向左扩展50像素
        expand_right = 50  # 向右扩展50像素
        
        # 计算胎面区域边界
        left_bound = max(0, label_left - expand_left)
        right_bound = min(width - 1, label_right + expand_right)
        
        # 确保胎面区域包含编号区域
        if left_bound > label_left:
            left_bound = max(0, label_left - 10)  # 如果左边扩展不够，至少保证包含编号区域
        if right_bound < label_right:
            right_bound = min(width - 1, label_right + 10)  # 如果右边扩展不够，至少保证包含编号区域
        
        # 按照经验，ROI的左边界再向左扩展整体宽度的20%（用户反馈需要更多扩展）
        region_width = right_bound - left_bound
        left_extension = int(region_width * 0.20)  # 整体宽度的20%
        left_bound = max(0, left_bound - left_extension)
        
        print(f"  原始胎面区域: {left_bound+left_extension}-{right_bound}, 宽度={region_width}")
        print(f"  左边界扩展: {left_extension}像素 (整体宽度的20%)")
        
        # 确保宽度合理
        min_width = 80  # 胎面区域最小宽度
        max_width = 350  # 胎面区域最大宽度（考虑了向左扩展）
        region_width = right_bound - left_bound  # 重新计算宽度
        
        if region_width < min_width:
            # 如果区域太窄，向两边扩展
            needed = min_width - region_width
            left_bound = max(0, left_bound - needed // 2)
            right_bound = min(width - 1, right_bound + needed // 2)
            print(f"  区域太窄，扩展宽度到{right_bound-left_bound}")
        elif region_width > max_width:
            # 如果区域太宽，向内缩小（主要从右边缩小）
            excess = region_width - max_width
            right_bound = right_bound - excess
            print(f"  警告: 胎面区域宽度({region_width})过大，已缩小到{right_bound-left_bound}")
        
        # 确保左边界小于右边界
        if left_bound >= right_bound:
            # 如果出现异常，使用默认宽度
            left_bound = max(0, label_left - 50)
            right_bound = min(width - 1, label_right + 50)
            print(f"  边界异常，使用默认范围: {left_bound}-{right_bound}")
        
        # 保存胎面区域边界
        self.tread_left_global = left_bound
        self.tread_right_global = right_bound
        
        # 计算编号区域在胎面ROI中的相对位置
        self.label_left_relative = label_left - left_bound
        self.label_right_relative = label_right - left_bound
        
        print(f"  胎面区域: 左边界={left_bound}, 右边界={right_bound}, 宽度={right_bound-left_bound}")
        print(f"  编号区域在胎面内位置: 相对左偏移={self.label_left_relative}, 相对右偏移={self.label_right_relative}")
        
        # 提取整个高度的胎面区域
        self.tread_roi_image = self.original_image[:, left_bound:right_bound]
        
        # 3. 处理上下边界
        # 获取胎面区域的二进制图像部分
        tread_binary = self.binary_image[:, left_bound:right_bound]
        tread_height, tread_width = tread_binary.shape
        
        print("  处理上下边界...")
        
        # 上边界：从上往下扫描行，寻找真正的胎面边缘
        # 胎面区域应该是黑色像素比例很高的区域
        # 我们需要找到从背景（黑色像素低）到胎面（黑色像素高）的过渡
        # 但实际上，图像已经裁剪到胎面区域，所以我们需要找胎面的真正上下边缘
        
        # 方法：寻找连续多行黑色像素比例 < 90% 的区域，认为是胎面边缘
        top_bound = 0
        consecutive_low_black_rows = 0
        threshold_low_black = 0.90  # 黑色像素比例低于90%认为是边缘
        
        for y in range(tread_height):
            black_pixels = np.sum(tread_binary[y, :] == 0)
            black_ratio = black_pixels / tread_width
            
            if black_ratio < threshold_low_black:
                consecutive_low_black_rows += 1
                if consecutive_low_black_rows >= 5:  # 连续5行黑色像素都低，认为是上边界
                    top_bound = y - 2  # 取最早的一行
                    print(f"    发现上边界: y={top_bound}, 连续{consecutive_low_black_rows}行黑色像素<{threshold_low_black}")
                    break
            else:
                consecutive_low_black_rows = 0  # 重置计数
        
        # 如果没找到明显的边界，使用默认值
        if top_bound == 0:
            print("    未发现明显上边界，使用默认上边界")
            top_bound = 0
        
        # 下边界：从下往上扫描行
        bottom_bound = tread_height - 1
        consecutive_low_black_rows = 0
        
        for y in range(tread_height - 1, -1, -1):
            black_pixels = np.sum(tread_binary[y, :] == 0)
            black_ratio = black_pixels / tread_width
            
            if black_ratio < threshold_low_black:
                consecutive_low_black_rows += 1
                if consecutive_low_black_rows >= 5:  # 连续5行黑色像素都低，认为是下边界
                    bottom_bound = y + 2  # 取最早的一行
                    print(f"    发现下边界: y={bottom_bound}, 连续{consecutive_low_black_rows}行黑色像素<{threshold_low_black}")
                    break
            else:
                consecutive_low_black_rows = 0  # 重置计数
        
        # 如果没找到明显的边界，使用默认值
        if bottom_bound == tread_height - 1:
            print("    未发现明显下边界，使用默认下边界")
            bottom_bound = tread_height - 1
        
        # 检查边界是否有效
        if bottom_bound <= top_bound:
            print("  警告: 上下边界异常，使用默认边界")
            top_bound = 0
            bottom_bound = tread_height - 1
        
        # 上下边界分别向外扩展几个像素
        expand_top = 20
        expand_bottom = 20
        top_bound = max(0, top_bound - expand_top)
        bottom_bound = min(tread_height - 1, bottom_bound + expand_bottom)
        
        print(f"  扩展后的上下边界: 上边界={top_bound}, 下边界={bottom_bound}, 高度={bottom_bound-top_bound+1}")
        
        # 使用新的上下边界裁剪胎面区域
        self.tread_roi_image = self.tread_roi_image[top_bound:bottom_bound+1, :]
        
        # 4. 图像增强
        # 灰度化
        roi_gray = cv2.cvtColor(self.tread_roi_image, cv2.COLOR_BGR2GRAY)
        
        # CLAHE 增强
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.enhanced_image = clahe.apply(roi_gray)
        
        # 保存中间结果
        cv2.imwrite(os.path.join(self.output_dir, "1_binary.png"), self.binary_image)
        
        # 保存裁剪前的胎面区域用于对比
        tread_full_height = self.original_image[:, left_bound:right_bound]
        cv2.imwrite(os.path.join(self.output_dir, "2_roi_full.png"), tread_full_height)
        cv2.imwrite(os.path.join(self.output_dir, "2_roi.png"), self.tread_roi_image)
        cv2.imwrite(os.path.join(self.output_dir, "3_enhanced.png"), self.enhanced_image)
        
        return self.enhanced_image
    
    def stage1_visualize(self):
        """可视化第一阶段处理结果"""
        if self.original_image is None:
            print("请先运行 stage1_extract_roi()")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original image
        axes[0, 0].imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')
        
        # Binary image
        axes[0, 1].imshow(self.binary_image, cmap='gray')
        axes[0, 1].set_title("Binary Image (Otsu)")
        axes[0, 1].axis('off')
        
        # Tread ROI image
        if self.tread_roi_image is not None:
            axes[1, 0].imshow(cv2.cvtColor(self.tread_roi_image, cv2.COLOR_BGR2RGB))
            axes[1, 0].set_title(f"Tread Region ({self.tread_roi_image.shape[1]}x{self.tread_roi_image.shape[0]})")
            axes[1, 0].axis('off')
        
        # Enhanced image
        if self.enhanced_image is not None:
            axes[1, 1].imshow(self.enhanced_image, cmap='gray')
            axes[1, 1].set_title("CLAHE Enhancement")
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "stage1_result.png"), dpi=150)
        plt.show()
    
    # ==================== 第二阶段：标签区域定位（优先使用第一阶段结果） ====================
    def stage2_label_localization(self) -> Optional[np.ndarray]:
        """
        第二阶段：在胎面ROI内部定位编号区域（白色文字区域）
        优先使用第一阶段找到的编号区域位置，如果不可用则使用亮度分析
        
        Returns:
            标签区域二值化图像，若无标签则返回 None
        """
        if self.enhanced_image is None:
            raise ValueError("请先运行 stage1_extract_roi()")
        
        print("  定位编号区域...")
        
        height, width = self.enhanced_image.shape
        
        # 优先使用第一阶段保存的编号区域位置
        start = None
        end = None
        
        if self.label_left_relative is not None and self.label_right_relative is not None:
            start = self.label_left_relative
            end = self.label_right_relative
            label_width = end - start
            print(f"  使用第一阶段定位的编号区域: 列 {start}-{end}, 宽度={label_width}")
        else:
            # 如果第一阶段没有找到编号区域，使用亮度分析
            print("  使用亮度分析定位编号区域...")
            
            # 分析每列的亮度特征
            col_means = np.mean(self.enhanced_image, axis=0)
            
            # 寻找亮度明显高于平均的区域（编号区域是白色点阵，应该比较亮）
            mean_brightness = np.mean(col_means)
            std_brightness = np.std(col_means)
            bright_threshold = mean_brightness + std_brightness * 0.3
            
            # 标记亮列
            bright_cols = np.where(col_means > bright_threshold)[0]
            print(f"    亮度阈值: {bright_threshold:.1f}, 亮列数量: {len(bright_cols)}")
            
            if len(bright_cols) == 0:
                print("    未找到明显的高亮列")
                return None
            
            # 找到连续的亮列区域（可能的编号区域）
            bright_regions = []
            if len(bright_cols) > 0:
                current_start = bright_cols[0]
                current_end = bright_cols[0]
                
                for i in range(1, len(bright_cols)):
                    if bright_cols[i] == bright_cols[i-1] + 1:
                        current_end = bright_cols[i]
                    else:
                        region_width = current_end - current_start + 1
                        if region_width >= 10:  # 最小宽度10像素
                            bright_regions.append((current_start, current_end, region_width))
                        current_start = bright_cols[i]
                        current_end = bright_cols[i]
                
                # 添加最后一个区域
                region_width = current_end - current_start + 1
                if region_width >= 10:
                    bright_regions.append((current_start, current_end, region_width))
            
            if not bright_regions:
                print("    未找到足够宽的高亮区域")
                return None
            
            # 选择最可能是编号区域的区域
            print(f"    找到 {len(bright_regions)} 个候选区域:")
            region_scores = []
            
            for i, (s, e, region_width) in enumerate(bright_regions):
                avg_brightness = np.mean(col_means[s:e+1])
                # 评分：宽度 * 平均亮度 * (1 - |宽度-40|/100) [鼓励宽度接近40像素]
                width_penalty = 1 - abs(region_width - 40) / 100
                score = region_width * avg_brightness * max(0.1, width_penalty)
                region_scores.append((i, s, e, region_width, avg_brightness, score))
                print(f"      区域 {i}: 列 {s}-{e}, 宽度={region_width}, 亮度={avg_brightness:.1f}, 得分={score:.1f}")
            
            # 选择得分最高的区域
            region_scores.sort(key=lambda x: x[5], reverse=True)
            best_idx, s, e, region_width, avg_brightness, score = region_scores[0]
            print(f"    选择区域 {best_idx}: 列 {s}-{e}, 宽度={region_width}, 亮度={avg_brightness:.1f}")
            
            start = s
            end = e
        
        # 扩展区域以确保包含整个编号区域
        margin = 15  # 扩展边界
        x_start = max(0, start - margin)
        x_end = min(width - 1, end + margin)
        
        # 确保宽度至少为20像素
        min_width = 20
        if x_end - x_start < min_width:
            # 扩展宽度
            center = (x_start + x_end) // 2
            x_start = max(0, center - min_width // 2)
            x_end = min(width - 1, center + min_width // 2)
        
        # 垂直方向：使用整个高度（编号区域通常从上到下）
        y_start = 0
        y_end = height - 1
        
        # 提取标签 ROI
        self.label_roi_image = self.tread_roi_image[y_start:y_end+1, x_start:x_end+1]
        
        # 检查提取的ROI是否有效
        if self.label_roi_image is None or self.label_roi_image.size == 0:
            print(f"  错误: 提取的编号区域ROI为空")
            self.label_roi_image = None
            self.label_binary = None
            return None
            
        print(f"  提取编号区域ROI: 位置({x_start},{y_start})-({x_end},{y_end}), 尺寸={self.label_roi_image.shape}")
        
        # 在编号区域内提取二值化字符（改进版，针对点阵字符）
        label_gray = cv2.cvtColor(self.label_roi_image, cv2.COLOR_BGR2GRAY)
        
        # 增强对比度
        label_enhanced = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(label_gray)
        
        # 使用自适应阈值提取字符（在编号区域内）
        label_binary = cv2.adaptiveThreshold(label_enhanced, 255,
                                             cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, 21, 5)
        
        # 确保字符是白色（255），背景是黑色（0）
        # 对于编号区域，背景应该是黑色（胎面），字符是白色
        white_pixels = np.sum(label_binary == 255)
        black_pixels = np.sum(label_binary == 0)
        if white_pixels < black_pixels:  # 如果字符是黑色，背景是白色
            label_binary = cv2.bitwise_not(label_binary)
        
        # 形态学闭运算，连接离散的点阵字符
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        label_binary = cv2.morphologyEx(label_binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # 可选：开运算去除小噪声
        label_binary = cv2.morphologyEx(label_binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        self.label_binary = label_binary
        
        # 保存中间结果
        cv2.imwrite(os.path.join(self.output_dir, "4_label_binary.png"), self.label_binary)
        
        return self.label_binary
    
    def stage2_visualize(self):
        """可视化第二阶段处理结果"""
        if self.enhanced_image is None or self.label_binary is None:
            print("请先运行 stage2_label_localization()")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Enhanced image
        axes[0, 0].imshow(self.enhanced_image, cmap='gray')
        axes[0, 0].set_title("Enhanced Image (Input)")
        axes[0, 0].axis('off')
        
        # Label ROI image
        if self.label_roi_image is not None:
            axes[0, 1].imshow(cv2.cvtColor(self.label_roi_image, cv2.COLOR_BGR2RGB))
            axes[0, 1].set_title("Label Region ROI")
            axes[0, 1].axis('off')
        
        # Binary label result
        axes[1, 0].imshow(self.label_binary, cmap='gray')
        axes[1, 0].set_title("Binary Label (after morphology)")
        axes[1, 0].axis('off')
        
        # Histogram analysis
        axes[1, 1].hist(self.enhanced_image.ravel(), 256, [0, 256])
        axes[1, 1].set_title("Enhanced Image Histogram")
        axes[1, 1].set_xlabel("Pixel Value")
        axes[1, 1].set_ylabel("Frequency")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "stage2_result.png"), dpi=150)
        plt.show()
    
    # ==================== 第三阶段：字符区域分割（仅区域划分锁定） ====================
    def stage3_character_segmentation(self) -> List[np.ndarray]:
        """
        第三阶段：字符区域分割
        1. 在标签区域内进行垂直投影分析，寻找字符间的切割点
        2. 针对点阵字符特征调整过滤条件（点阵字符通常非常窄）
        3. 归一化处理 (64x64)，但不进行字符识别
        
        Returns:
            归一化后的字符图像列表，若无标签则返回空列表
        """
        if self.label_binary is None:
            print("  无标签区域，跳过字符分割")
            self.segmented_chars = []
            self.char_positions = []
            return []
        
        print("  开始标签内字符区域分割...")
        binary = self.label_binary.copy()
        height, width = binary.shape
        
        # 1. 垂直投影分析 - 针对点阵字符调整
        # 计算每列的白色像素数量（字符像素）
        vertical_projection = np.sum(binary > 0, axis=0)
        
        # 寻找字符之间的空白列（切割点）
        # 对于点阵字符，空白列的阈值可以更小，因为字符本身很窄
        blank_threshold = height * 0.05  # 降低到5%
        print(f"    空白列阈值: {blank_threshold}")
        
        # 另外，如果整个列都是白色（垂直投影值等于高度），可能是背景噪声
        # 将这些列也视为空白
        full_white_cols = vertical_projection == height
        is_blank = (vertical_projection < blank_threshold) | full_white_cols
        
        print(f"    空白列数量: {is_blank.sum()}/{len(is_blank)}")
        
        # 生成切割区间
        cut_positions = []
        in_char = False
        start = 0
        
        for i in range(width):
            if not is_blank[i] and not in_char:
                # 字符开始
                start = i
                in_char = True
            elif is_blank[i] and in_char:
                # 字符结束
                end = i
                if end - start >= 1:  # 最小宽度：1个像素（点阵字符可能只有1-2像素宽）
                    cut_positions.append((start, end))
                in_char = False
        
        # 如果最后一个字符延伸到末尾
        if in_char and width - start >= 1:
            cut_positions.append((start, width))
        
        print(f"  初步找到 {len(cut_positions)} 个候选字符区域")
        
        # 2. 连通域过滤 - 针对点阵字符调整
        self.segmented_chars = []
        self.char_positions = []
        
        for idx, (left, right) in enumerate(cut_positions):
            # 提取字符区域
            char_region = binary[:, left:right]
            
            # 水平投影，裁剪上下空白
            horizontal_projection = np.sum(char_region > 0, axis=1)
            # 对于点阵字符，水平空白阈值可以更低
            h_blank_threshold = max(1, (right - left) * 0.05)  # 降低到5%
            top = 0
            bottom = height - 1
            
            # 寻找上边界
            for i in range(height):
                if horizontal_projection[i] > h_blank_threshold:
                    top = max(0, i - 1)  # 稍微扩大一点
                    break
            
            # 寻找下边界
            for i in range(height - 1, -1, -1):
                if horizontal_projection[i] > h_blank_threshold:
                    bottom = min(height - 1, i + 1)
                    break
            
            # 提取字符图像
            char_img = binary[top:bottom+1, left:right]
            
            # 检查字符图像是否有效
            if char_img.size == 0:
                print(f"    跳过区域 {idx}: 字符图像为空")
                continue
                
            char_height, char_width = char_img.shape
            
            # 点阵字符可能非常窄，所以降低最小宽度要求
            if char_height < 3 or char_width < 1:  # 最小高度3像素，最小宽度1像素
                print(f"    跳过区域 {idx}: 尺寸过小 ({char_width}x{char_height})")
                continue
            
            aspect_ratio = char_width / char_height
            area = char_height * char_width
            
            # 点阵字符的宽高比通常很小（0.001-0.3），因为它们是垂直排列的点
            # 放宽宽高比限制
            min_aspect_ratio = 0.001  # 允许非常窄的字符（之前是0.005）
            max_aspect_ratio = 0.5    # 点阵字符不会太宽
            
            if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
                print(f"    跳过区域 {idx}: 宽高比异常 {aspect_ratio:.3f} (期望: {min_aspect_ratio:.3f}-{max_aspect_ratio:.3f})")
                continue
            
            # 面积过滤 - 点阵字符面积可能很小
            min_area = 5  # 最小面积5像素
            if area < min_area:
                print(f"    跳过区域 {idx}: 面积过小 {area} (最小: {min_area})")
                continue
            
            # 3. 归一化处理 (64x64)
            normalized_char = self._normalize_char(char_img)
            
            self.segmented_chars.append(normalized_char)
            self.char_positions.append((left, top, right, bottom))
        
        print(f"  分割出 {len(self.segmented_chars)} 个有效字符区域")
        
        # 保存分割结果图像
        if self.segmented_chars:
            # 绘制带有边界框的图像
            img_with_boxes = cv2.cvtColor(self.label_binary, cv2.COLOR_GRAY2RGB)
            for idx, (left, top, right, bottom) in enumerate(self.char_positions):
                cv2.rectangle(img_with_boxes, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(img_with_boxes, str(idx), (left, top-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.imwrite(os.path.join(self.output_dir, "5_segmented_chars.png"), img_with_boxes)
        
        return self.segmented_chars
    
    def _normalize_char(self, char_img: np.ndarray, target_size: int = 64) -> np.ndarray:
        """
        将字符图像归一化为固定大小
        """
        # 检查输入图像是否有效
        if char_img.size == 0:
            # 返回全黑图像
            return np.zeros((target_size, target_size), dtype=np.uint8)
        
        # 添加 padding，保持宽高比
        h, w = char_img.shape
        
        # 确保高度和宽度大于0
        if h <= 0 or w <= 0:
            return np.zeros((target_size, target_size), dtype=np.uint8)
        
        # 计算缩放比例，避免除零
        scale_h = target_size / h if h > 0 else 0
        scale_w = target_size / w if w > 0 else 0
        scale = min(scale_h, scale_w)
        
        # 确保缩放比例有效
        if scale <= 0:
            return np.zeros((target_size, target_size), dtype=np.uint8)
        
        new_h, new_w = int(h * scale), int(w * scale)
        
        # 确保新尺寸大于0
        if new_h <= 0 or new_w <= 0:
            return np.zeros((target_size, target_size), dtype=np.uint8)
        
        # 缩放图像
        resized = cv2.resize(char_img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # 创建目标图像，填充为背景色（黑色）
        normalized = np.zeros((target_size, target_size), dtype=np.uint8)
        
        # 计算放置位置（居中）
        y_offset = (target_size - new_h) // 2
        x_offset = (target_size - new_w) // 2
        
        # 放置字符
        normalized[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return normalized
    
    def stage3_visualize(self):
        """可视化第三阶段处理结果"""
        if not self.segmented_chars:
            print("请先运行 stage3_character_segmentation()")
            return
        
        # Create visualization image
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Original binary character image
        if self.label_binary is not None:
            axes[0, 0].imshow(self.label_binary, cmap='gray')
            axes[0, 0].set_title("Binary Label Image")
            axes[0, 0].axis('off')
            
            # Draw character bounding boxes on image
            img_with_boxes = cv2.cvtColor(self.label_binary, cv2.COLOR_GRAY2RGB)
            for idx, (left, top, right, bottom) in enumerate(self.char_positions):
                cv2.rectangle(img_with_boxes, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(img_with_boxes, str(idx), (left, top-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            axes[0, 1].imshow(img_with_boxes)
            axes[0, 1].set_title(f"Character Bounding Boxes ({len(self.char_positions)} chars)")
            axes[0, 1].axis('off')
        
        # 2. Display some segmented characters
        num_chars_to_show = min(16, len(self.segmented_chars))
        if num_chars_to_show > 0:
            # Calculate grid size
            grid_size = int(np.ceil(np.sqrt(num_chars_to_show)))
            
            # Create character grid
            char_size = 64
            margin = 5
            grid_img = np.zeros((grid_size * (char_size + margin), 
                                grid_size * (char_size + margin)), dtype=np.uint8)
            
            for i in range(num_chars_to_show):
                row = i // grid_size
                col = i % grid_size
                y_start = row * (char_size + margin)
                x_start = col * (char_size + margin)
                grid_img[y_start:y_start+char_size, x_start:x_start+char_size] = self.segmented_chars[i]
            
            axes[1, 0].imshow(grid_img, cmap='gray')
            axes[1, 0].set_title(f"Segmented Characters (first {num_chars_to_show})")
            axes[1, 0].axis('off')
        
        # 3. Character size distribution
        if self.char_positions:
            widths = [right - left for left, _, right, _ in self.char_positions]
            heights = [bottom - top for _, top, _, bottom in self.char_positions]
            
            axes[1, 1].scatter(widths, heights, alpha=0.6)
            axes[1, 1].set_xlabel("Width (pixels)")
            axes[1, 1].set_ylabel("Height (pixels)")
            axes[1, 1].set_title("Character Size Distribution")
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "stage3_result.png"), dpi=150)
        plt.show()
    
    # ==================== 第四阶段：传统字符识别 (HOG+SVM) ====================
    def _generate_synthetic_dataset(self, num_samples_per_char: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成合成点阵字符数据集
        Args:
            num_samples_per_char: 每个字符生成的样本数量
        
        Returns:
            images: 归一化后的字符图像列表
            labels: 对应的标签（0-9, A-Z 映射为 0-35）
        """
        chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        char_to_label = {char: idx for idx, char in enumerate(chars)}
        
        images = []
        labels = []
        
        # 创建空白图像用于绘制
        canvas_size = 64
        font_choices = [
            cv2.FONT_HERSHEY_SIMPLEX,
            cv2.FONT_HERSHEY_PLAIN,
            cv2.FONT_HERSHEY_DUPLEX,
            cv2.FONT_HERSHEY_COMPLEX,
            cv2.FONT_HERSHEY_TRIPLEX
        ]
        
        print(f"  生成合成字符数据集 ({len(chars)} 个字符，每个 {num_samples_per_char} 个样本)...")
        
        for char in chars:
            for sample_idx in range(num_samples_per_char):
                # 随机选择字体和大小
                font = random.choice(font_choices)
                font_scale = random.uniform(1.5, 3.0)
                thickness = random.randint(1, 3)
                
                # 创建空白图像（黑色背景）
                img = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
                
                # 计算文本大小和位置
                (text_width, text_height), baseline = cv2.getTextSize(char, font, font_scale, thickness)
                x = (canvas_size - text_width) // 2
                y = (canvas_size + text_height) // 2
                
                # 绘制白色字符
                cv2.putText(img, char, (x, y), font, font_scale, 255, thickness)
                
                # 添加噪声以模拟真实点阵字符
                # 1. 随机点阵化：将部分像素设为0（模拟点阵缺失）
                mask = np.random.random((canvas_size, canvas_size)) > 0.85  # 15% 的像素被移除
                img[mask] = 0
                
                # 2. 添加高斯噪声
                noise = np.random.normal(0, 10, (canvas_size, canvas_size)).astype(np.int32)
                img = np.clip(img.astype(np.int32) + noise, 0, 255).astype(np.uint8)
                
                # 3. 形态学操作模拟点阵
                kernel_size = random.choice([1, 3])
                if kernel_size > 1:
                    kernel = np.ones((kernel_size, kernel_size), np.uint8)
                    img = cv2.erode(img, kernel, iterations=1)
                
                # 4. 二值化
                _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
                
                images.append(img)
                labels.append(char_to_label[char])
        
        return np.array(images), np.array(labels)
    
    def _extract_hog_features(self, images: np.ndarray) -> np.ndarray:
        """
        提取 HOG 特征
        Args:
            images: 归一化后的字符图像列表
        
        Returns:
            hog_features: HOG 特征向量
        """
        if self.hog_descriptor is None:
            # 初始化 HOG 描述器
            winSize = (64, 64)
            blockSize = (16, 16)
            blockStride = (8, 8)
            cellSize = (8, 8)
            nbins = 9
            
            self.hog_descriptor = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
        
        hog_features = []
        for img in images:
            # 计算 HOG 特征
            hog_feature = self.hog_descriptor.compute(img)
            hog_features.append(hog_feature.flatten())
        
        return np.array(hog_features)
    
    def _train_classifier(self, num_samples_per_char: int = 30):
        """
        训练 HOG+SVM 分类器
        """
        print("  训练 HOG+SVM 分类器...")
        
        # 生成合成数据集
        images, labels = self._generate_synthetic_dataset(num_samples_per_char)
        
        # 提取 HOG 特征
        features = self._extract_hog_features(images)
        
        # 训练 SVM
        svm = cv2.ml.SVM_create()
        svm.setType(cv2.ml.SVM_C_SVC)
        svm.setKernel(cv2.ml.SVM_LINEAR)
        svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-6))
        
        svm.train(features.astype(np.float32), cv2.ml.ROW_SAMPLE, labels.astype(np.int32))
        
        self.classifier = svm
        self.class_labels = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.is_trained = True
        
        print(f"  分类器训练完成，支持 {len(self.class_labels)} 个字符")
    
    def stage4_hog_svm_recognition(self, char_images: List[np.ndarray]) -> str:
        """
        第四阶段：使用 HOG+SVM 进行字符识别
        Args:
            char_images: 归一化后的字符图像列表
        
        Returns:
            识别出的字符串
        """
        if not self.is_trained:
            self._train_classifier()
        
        if not char_images:
            return ""
        
        # 准备输入图像
        images_array = np.array(char_images)
        
        # 提取 HOG 特征
        features = self._extract_hog_features(images_array)
        
        # 预测
        _, predictions = self.classifier.predict(features.astype(np.float32))
        predictions = predictions.flatten().astype(int)
        
        # 转换为字符
        result = ""
        for pred in predictions:
            if 0 <= pred < len(self.class_labels):
                result += self.class_labels[pred]
            else:
                result += "?"
        
        return result
    
    # ==================== 完整处理流程 ====================
    def full_pipeline(self, image_path: str) -> str:
        """
        完整处理流程（重构版）
        Args:
            image_path: 输入图像路径
            
        Returns:
            识别出的字符串
        """
        print("=== 开始处理图像 ===")
        # 第一阶段：胎面 ROI 提取
        enhanced = self.stage1_extract_roi(image_path)
        if enhanced is None:
            print("第一阶段完成: 未发现编号区域，直接判定为'无'")
            return "无"
        
        print("第一阶段完成: 胎面 ROI 提取与增强")
        
        # 第二阶段：标签区域定位
        label_binary = self.stage2_label_localization()
        if label_binary is not None:
            print("第二阶段完成: 找到标签区域")
        else:
            print("第二阶段完成: 未发现标签区域")
            return "无标签"
        
        # 第三阶段：字符区域分割
        char_images = self.stage3_character_segmentation()
        if char_images:
            print(f"第三阶段完成: 分割出 {len(char_images)} 个字符区域")
        else:
            print("第三阶段完成: 无字符区域可分割")
            return "无字符"
        
        # 第四阶段：HOG+SVM 识别
        result = self.stage4_hog_svm_recognition(char_images)
        print(f"第四阶段完成: 识别结果 = '{result}'")
        
        # 保存最终结果图像
        self._save_final_result(result)
        
        return result
    
    def _save_final_result(self, result: str):
        """保存最终识别结果图像"""
        if self.label_binary is None or not self.char_positions:
            return
        
        # 创建结果图像
        result_img = cv2.cvtColor(self.label_binary, cv2.COLOR_GRAY2RGB)
        
        # 绘制边界框和标签
        for idx, (left, top, right, bottom) in enumerate(self.char_positions):
            cv2.rectangle(result_img, (left, top), (right, bottom), (0, 255, 0), 2)
            if idx < len(result):
                cv2.putText(result_img, result[idx], (left, top-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # 添加整体识别结果文本
        cv2.putText(result_img, f"识别结果: {result}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        cv2.imwrite(os.path.join(self.output_dir, "6_final_result.png"), result_img)
        print(f"  最终结果已保存到 {self.output_dir}/6_final_result.png")
    
    def run_and_visualize(self, image_path: str):
        """
        运行完整流程并可视化所有结果
        """
        result = self.full_pipeline(image_path)
        
        # 可视化各阶段结果
        if hasattr(self, 'original_image') and self.original_image is not None:
            self.stage1_visualize()
        
        if hasattr(self, 'label_binary') and self.label_binary is not None:
            self.stage2_visualize()
        
        if self.segmented_chars:
            self.stage3_visualize()
        
        print(f"\n=== 处理完成 ===")
        print(f"原始图像: {image_path}")
        print(f"识别结果: {result}")
        print(f"中间结果保存在: {self.output_dir}/")
        
        return result


def main():
    """主函数，演示重构后的处理流程"""
    import sys
    
    # 检查参数
    if len(sys.argv) < 2:
        print("用法: python tire_ocr.py <图像路径>")
        print("示例: python tire_ocr.py data/20250115083551210.png")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # 创建 OCR 处理器
    ocr = TireOCR()
    
    try:
        # 执行完整流程
        result = ocr.run_and_visualize(image_path)
        
    except Exception as e:
        print(f"处理失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()