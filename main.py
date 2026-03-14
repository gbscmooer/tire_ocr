import cv2
import numpy as np
import os
import threading
import queue
import time
from typing import Tuple, Optional

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
        
        # ==================== 集中配置参数 ====================
        self.cfg = {
            # ROI 边界扩展 (stage1)
            'expand_left': 0,           # 向左扩展像素
            'expand_right': 40,         # 向右扩展像素
            'min_width': 80,           # 胎面区域最小宽度
            'max_width': 350,          # 胎面区域最大宽度
            'expand_top': 20,          # 上边界扩展像素
            'expand_bottom': 20,       # 下边界扩展像素
            
            # Stage2 形态学参数
            'stage2_open_kernel': (2, 2),
            'stage2_open_iterations': 1,
            'stage2_dilation_kernel': (7, 9),
            'stage2_dilation_iterations': 1,
            'stage2_close_kernel': (9, 7),
            'stage2_close_iterations': 2,
            
            # Stage5 形态学参数 (字符连通) - 降权避免横向粘连
            'stage5_open_kernel': (5, 1),        # 轻量开运算去噪
            'stage5_open_iterations': 1,
            'stage5_dilation_kernel': (1,1),    # 窄宽度，重点垂直连接（原(9,3)太宽）
            'stage5_dilation_iterations': 1,
            'stage5_close_kernel': (1, 2),       # 调整为垂直方向连接
            'stage5_close_iterations': 1,
            'stage5_h_erosion_kernel': (3, 1),   # 横向腐蚀核：收缩点阵字符横向宽度
            'stage5_h_erosion_iterations':5,    # 横向腐蚀迭代次数
            'stage5_v_erosion_kernel': (1, 3),   # 纵向腐蚀核：收缩点阵字符纵向高度，分开纵向粘连
            'stage5_v_erosion_iterations':4,    # 纵向腐蚀迭代次数
            
            # 轮廓过滤参数
            'contour_min_w': 8,
            'contour_min_h': 12,
            'contour_max_w': 90,
            'contour_max_h': 60,
            'contour_min_area': 80,
            'contour_min_fill_ratio': 0.15,
            'contour_aspect_min': 0.15,
            'contour_aspect_max': 5.0,
            'contour_merge_gap': 10,           # 轮廓合并的最大间隙
            'contour_y_std_factor': 1.0,       # Y轴过滤的标准差倍数
            
            # 字符分割参数
            'split_width_threshold': 35,       # 触发分割的字符宽度阈值
            'split_valley_ratio': 0.5,         # 水平投影波谷阈值比例
            'split_edge_margin': 10,           # 分割点距边缘的最小距离
            'split_min_char_width': 20,         # 分割后字符的最小宽度
            
            # 识别参数
            'confidence_threshold': 0.01,      # 置信度阈值
            'knn_k': 3,                        # KNN 的 k 值
            
            # 其他阈值
            'white_pixel_ratio_invert': 0.5,   # 白色像素比例超过此值则反转图像
            'white_pixel_ratio_low': 0.1,      # 白色像素比例低于此值则检查反转
            'white_pixel_ratio_label': 0.05,   # 编号区域白色像素比例阈值
            # 全局调试开关，True 时会输出中间产物并启用额外调试绘图
            'debug': False,
        }
        
        # 存储中间结果
        self.original_image = None          # 原始图像
        self.binary_image = None            # 二值化图像
        self.tread_roi_image = None         # 胎面区域图像（黑色区域）
        self.enhanced_image = None          # 增强后图像
    # (tophat intermediate not used in current pipeline)
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

    # 形态学处理结果 (initialized in stage4)
    
    # ==================== 第一阶段：ROI 提取与预处理 ====================
    def stage1_extract_roi(self, image_path: str) -> np.ndarray:
        """
        第一阶段：ROI 提取与预处理
        1. 全局二值化（Otsu）得到黑白图像（胎面为黑0，背景为白255）
        2. 从图像中心 x = 宽度/2 向两边扫描，找到黑色胎面区域的左右边界
        3. 提取胎面 ROI 并灰度化 + 全局二值化 + 顺时针旋转90度
        
        Args:
            image_path: 输入图像路径
        
        Returns:
            胎面 ROI 增强后的图像 (尺寸: 2396x229)
        """
        # 读取图像：支持传入文件路径或已加载的 numpy 图像
        if isinstance(image_path, np.ndarray):
            self.original_image = image_path.copy()
        else:
            self.original_image = cv2.imread(image_path)
            if self.original_image is None:
                raise FileNotFoundError(f"无法读取图像: {image_path}")
        
        # 1. 全局二值化（Otsu）
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        _, self.binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 2. 使用中心向左扫描找到编号区域（白色文字区域），然后基于编号区域锁定整个胎面区域
        height, width = self.binary_image.shape
        center_x = (width // 2 ) -120  # 从中心向左偏移120像素开始扫描，避免中心可能的白色区域干扰
        
        # 第一步：找到编号区域（白色文字区域）的左右边界
        label_left = None
        label_right = None
        min_valid_roi_width = 100  # ROI最小合法宽度（右边界-左边界）
        current_x = center_x
        
        # 状态机：0-寻找右边界，1-寻找左边界
        state = 0
        consecutive_label_cols = 0
        consecutive_tread_cols = 0
        
        while current_x > 0:
            black_pixels = np.sum(self.binary_image[:, current_x] == 0)
            black_ratio = black_pixels / height
            
            if state == 0:
                if black_ratio < 0.01:
                    # 遇到无关白色边缘，说明没有编号区域
                    print("  未发现编号区域 (遇到空白边缘)")
                    return None
                elif 0.01 <= black_ratio <= 0.99:
                    # 找到可能的编号区域右边界
                    consecutive_label_cols += 1
                    if consecutive_label_cols >= 2:  # 连续2列都在标签范围内才确认
                        label_right = current_x + 2  # 取最早的一列
                        state = 1
                        print(f"  发现编号区域右边界: {label_right}, 连续{consecutive_label_cols}列")
                else:
                    consecutive_label_cols = 0  # 重置计数
                    
            elif state == 1:
                if black_ratio > 0.995:
                    consecutive_tread_cols += 1
                    
                    # 增加容忍度机制：连续10个像素列都>99.5%才确认为左边界
                    if consecutive_tread_cols >= 15:
                        # 检查这10列是否都满足黑色像素比例 > 99.5%
                        check_start = current_x
                        check_end = current_x + 14  # 包括当前列和前9列
                        all_valid = True
                        check_count = 0
                        
                        # 独立判断每一列
                        for check_x in range(check_start, check_end + 1):
                            if check_x >= 0:
                                check_black_pixels = np.sum(self.binary_image[:, check_x] == 0)
                                check_black_ratio = check_black_pixels / height
                                if check_black_ratio <= 0.995:  # 如果任何一列不满足条件
                                    all_valid = False
                                    break
                                check_count += 1
                        
                        if all_valid and check_count == 15:  # 所有10列都满足条件
                            candidate_left = current_x + 2  # 取最早的一列
                            candidate_width = (label_right - candidate_left) if label_right is not None else 0
                            if candidate_width < min_valid_roi_width:
                                # ROI过窄：重置右边界为待定，从当前左边界位置重新向左扫描
                                print(
                                    f"  候选左边界 {candidate_left} 非法: 宽度={candidate_width}<"
                                    f"{min_valid_roi_width}，重置右边界并继续扫描"
                                )
                                label_left = None
                                label_right = None
                                state = 0
                                consecutive_label_cols = 0
                                consecutive_tread_cols = 0
                                current_x = candidate_left
                                continue
                            else:
                                label_left = candidate_left
                                print(f"  发现编号区域左边界: {label_left}, 连续{consecutive_tread_cols}列，所有列黑色像素比例均>99.5%")
                                break
                        
                        # 检查范围是否有效
                        total_black_ratio = 0
                        for check_x in range(check_start, check_end + 1):
                            if check_x >= 0:
                                check_black_pixels = np.sum(self.binary_image[:, check_x] == 0)
                                check_black_ratio = check_black_pixels / height
                                total_black_ratio += check_black_ratio
                                check_count += 1
                        
                        if check_count > 0:
                            avg_black_ratio = total_black_ratio / check_count
                            if avg_black_ratio > 0.995:  # 平均黑色像素比例必须非常高，说明确实是胎面区域
                                candidate_left = current_x + 2  # 取最早的一列
                                candidate_width = (label_right - candidate_left) if label_right is not None else 0
                                if candidate_width < min_valid_roi_width:
                                    # ROI过窄：重置右边界为待定，从当前左边界位置重新向左扫描
                                    print(
                                        f"  候选左边界 {candidate_left} 非法: 宽度={candidate_width}<"
                                        f"{min_valid_roi_width}，重置右边界并继续扫描"
                                    )
                                    label_left = None
                                    label_right = None
                                    state = 0
                                    consecutive_label_cols = 0
                                    consecutive_tread_cols = 0
                                    current_x = candidate_left
                                    continue
                                else:
                                    label_left = candidate_left
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
                    if black_ratio < 0.01:
                        print("  宽松模式: 遇到空白边缘")
                        return None
                    elif 0.01 <= black_ratio <= 0.99:  # 放宽阈值
                        if label_right is None:
                            label_right = current_x
                            state = 1
                elif state == 1:
                    if black_ratio > 0.95:  # 放宽阈值
                        candidate_left = current_x
                        candidate_width = (label_right - candidate_left) if label_right is not None else 0
                        if candidate_width < min_valid_roi_width:
                            # ROI过窄：重置右边界为待定，从当前左边界位置重新向左扫描
                            label_left = None
                            label_right = None
                            state = 0
                            current_x = candidate_left
                            continue
                        else:
                            label_left = candidate_left
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
        expand_left = self.cfg['expand_left']  # 向左扩展像素
        expand_right = self.cfg['expand_right']  # 向右扩展像素
        
        # 计算胎面区域边界
        left_bound = max(0, label_left - expand_left)
        right_bound = min(width - 1, label_right + expand_right)
        
        # 确保胎面区域包含编号区域
        if left_bound > label_left:
            left_bound = max(0, label_left - 10)  # 如果左边扩展不够，至少保证包含编号区域
        if right_bound < label_right:
            right_bound = min(width - 1, label_right + 10)  # 如果右边扩展不够，至少保证包含编号区域
        
        # 按照经验，ROI的左边界再向左扩展整体宽度的0%（修改后的参数）
        region_width = right_bound - left_bound
        left_extension = int(region_width * 0)  # 整体宽度的0%（修改后的参数）
        left_bound = max(0, left_bound - left_extension)
        
        print(f"  原始胎面区域: {left_bound+left_extension}-{right_bound}, 宽度={region_width}")
        print(f"  左边界扩展: {left_extension}像素 (整体宽度的0%)")
        
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
        
        # 方法：寻找连续多行黑色像素比例 < 99% 的区域，认为是胎面边缘
        top_bound = 0
        consecutive_low_black_rows = 0
        threshold_low_black = 0.99  # 黑色像素比例低于99%认为是边缘
        
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
        
        # 4. 图像增强（改为全局二值化+旋转）
        # 灰度化
        roi_gray = cv2.cvtColor(self.tread_roi_image, cv2.COLOR_BGR2GRAY)
        
        # 全局二值化（Otsu）替代CLAHE
        _, self.enhanced_image = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 顺时针旋转90度（原始尺寸229x2396 -> 旋转后2396x229）
        self.enhanced_image = cv2.rotate(self.enhanced_image, cv2.ROTATE_90_CLOCKWISE)
        
        # 保存最终增强图像（根据 cfg['debug'] 决定是否写盘）
        if self.cfg.get('debug', False):
            cv2.imwrite(os.path.join(self.output_dir, "3_enhanced.png"), self.enhanced_image)
            print(f"  增强图像已保存到 3_enhanced.png")
        else:
            print(f"  增强图像已生成（未写入磁盘，debug=False）")
        
        return self.enhanced_image
    
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
        margin = 30  # 扩展边界，避免截断
        x_start = max(0, start - margin)
        x_end = min(width - 1, end + margin)
        
        # 确保宽度至少为100像素，避免过窄导致旋转后高度太小
        min_width = 100
        if x_end - x_start < min_width:
            # 扩展宽度
            center = (x_start + x_end) // 2
            x_start = max(0, center - min_width // 2)
            x_end = min(width - 1, center + min_width // 2)
        
        # 垂直方向：使用整个高度（编号区域通常从上到下）
        y_start = 0
        y_end = height - 1
        
        # 调试输出
        print(f"    编号区域原始范围: {start}-{end}, 宽度={end-start}")
        print(f"    扩展后范围: {x_start}-{x_end}, 宽度={x_end-x_start}")
        print(f"    ROI高度范围: {y_start}-{y_end}, 高度={y_end-y_start+1}")
        
        # 直接使用已经旋转过的增强图像（3_enhanced.png），不再进行旋转
        # 第一阶段已经顺时针旋转90度，尺寸为2396x229
        print(f"  使用已旋转的增强图像: 尺寸={self.enhanced_image.shape}")
        
        # 由于第一阶段已经进行了全局二值化，self.enhanced_image已经是二值图像
        # 直接使用它作为基础，但确保字符是白色（255），背景是黑色（0）
        label_binary = self.enhanced_image.copy()
        
        # 确保字符是白色（255），背景是黑色（0）
        # 对于点阵字符，白色像素比例通常很低（因为是点状）
        # 检查整个图像的白色像素比例来决定是否反转
        white_pixels_total = np.sum(label_binary == 255)
        total_pixels = label_binary.size
        white_ratio_total = white_pixels_total / total_pixels if total_pixels > 0 else 0
        print(f"    整体图像白色像素比例: {white_ratio_total:.3f}")
        
        # 点阵字符：如果白色像素比例 > 50%，说明背景是白色，需要反转
        # 如果白色像素比例 < 30%，说明字符是白色在黑色背景上，这正是我们想要的
        if white_ratio_total > 0.5:  # 背景是白色，需要反转
            print("    背景为白色，反转图像使字符变白，背景变黑")
            label_binary = cv2.bitwise_not(label_binary)
        elif white_ratio_total < 0.1:  # 白色像素太少，可能是反转了两次或其他问题
            print("    白色像素太少，检查是否需要反转...")
            # 检查编号区域
            label_region = label_binary[:, x_start:x_end+1]
            white_ratio_in_label = np.sum(label_region == 255) / label_region.size if label_region.size > 0 else 0
            print(f"    编号区域内部白色像素比例: {white_ratio_in_label:.3f}")
            if white_ratio_in_label < 0.05:  # 编号区域也几乎没有白色像素
                print("    编号区域也缺少白色像素，反转图像")
                label_binary = cv2.bitwise_not(label_binary)
        
        # 先进行开运算去除小噪声，分离字符
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, self.cfg['stage2_open_kernel'])
        label_binary = cv2.morphologyEx(label_binary, cv2.MORPH_OPEN, kernel_open, 
                                       iterations=self.cfg['stage2_open_iterations'])
        
        # 增强形态学：先膨胀再闭运算，强制连接点阵字符成块
        # 使用较大的核连接垂直和水平方向上的离散点
        kernel_dilation = cv2.getStructuringElement(cv2.MORPH_RECT, self.cfg['stage2_dilation_kernel'])
        label_binary = cv2.dilate(label_binary, kernel_dilation, 
                                 iterations=self.cfg['stage2_dilation_iterations'])
        
        # 再进行闭运算填充内部空隙，形成完整字符块
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, self.cfg['stage2_close_kernel'])
        label_binary = cv2.morphologyEx(label_binary, cv2.MORPH_CLOSE, kernel_close, 
                                       iterations=self.cfg['stage2_close_iterations'])
        
        self.label_binary = label_binary
        
        # 更新相关变量以保持兼容性
        if len(self.enhanced_image.shape) == 2:
            enhanced_bgr = cv2.cvtColor(self.enhanced_image, cv2.COLOR_GRAY2BGR)
        else:
            enhanced_bgr = self.enhanced_image
        self.label_roi_image = enhanced_bgr  # 保存原始增强图像作为ROI
        
        return self.label_binary
    
    # (stage3_character_segmentation and its helper _normalize_char were removed
    #  because the current pipeline performs segmentation via the post-processing
    #  module. Keep the rest of the class focused on stages 1/2/4 and orchestration.)

    def _get_kernel_size(self, key: str) -> Tuple[int, int]:
        """读取并规范化形态学核尺寸，确保传给 OpenCV 的 ksize 为正整数元组。"""
        raw = self.cfg.get(key)
        if not isinstance(raw, (tuple, list)) or len(raw) != 2:
            raise ValueError(f"配置项 {key} 必须是长度为2的 tuple/list，当前值: {raw}")

        try:
            kx = max(1, int(round(raw[0])))
            ky = max(1, int(round(raw[1])))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"配置项 {key} 含有非法核尺寸值: {raw}") from exc

        return (kx, ky)
    
    def waterfall_segmentation(self, image: np.ndarray) -> np.ndarray:
        """
        滴水算法分离粘连的字符
        
        基本思路：
        1. 计算水平投影找到可能的分割位置
        2. 在每个候选位置模拟水滴下落，找到最佳分割路径
        3. 沿路径分割字符
        
        Args:
            image: 输入二值图像（字符为白色255，背景为黑色0）
            
        Returns:
            分离后的图像
        """
        h, w = image.shape
        result = image.copy()
        
        # 1. 计算水平投影（垂直方向的像素和）
        vertical_projection = np.sum(image > 0, axis=0).astype(np.int32)
        
        # 2. 找到波谷位置（候选分割点）
        min_val = np.min(vertical_projection)
        max_val = np.max(vertical_projection)
        threshold = min_val + (max_val - min_val) * 0.2  # 波谷阈值
        
        # 找到所有低于阈值的列
        candidate_cols = []
        for col in range(1, w - 1):
            if vertical_projection[col] < threshold:
                # 检查是否是局部极小值
                if vertical_projection[col] < vertical_projection[col - 1] and \
                   vertical_projection[col] < vertical_projection[col + 1]:
                    candidate_cols.append(col)
        
        if not candidate_cols:
            return result
        
        print(f"  滴水算法: 找到 {len(candidate_cols)} 个候选分割点")
        
        # 3. 对每个候选点进行滴水分割
        for col in candidate_cols:
            # 从顶部开始模拟水滴下落
            path = []
            row = 0
            current_col = col
            
            while row < h:
                path.append((row, current_col))
                
                # 检查是否到达底部
                if row >= h - 1:
                    break
                
                # 决定下一步方向：左、下、右
                left_col = max(0, current_col - 1)
                right_col = min(w - 1, current_col + 1)
                
                # 计算三个方向的"高度"（白色像素数量）
                down_height = 0
                left_height = 0
                right_height = 0
                
                # 向下方向
                if row + 1 < h:
                    if result[row + 1, current_col] > 0:
                        down_height = 1
                
                # 向左下方向
                if row + 1 < h and left_col >= 0:
                    if result[row + 1, left_col] > 0:
                        left_height = 1
                
                # 向右下方向
                if row + 1 < h and right_col < w:
                    if result[row + 1, right_col] > 0:
                        right_height = 1
                
                # 选择方向：优先选择没有像素的方向（避开字符）
                if down_height == 0:
                    # 向下是空的，继续向下
                    row += 1
                elif left_height == 0 and right_height == 0:
                    # 左右都是空的，选择离初始列更近的方向
                    if abs(left_col - col) < abs(right_col - col):
                        current_col = left_col
                    else:
                        current_col = right_col
                    row += 1
                elif left_height == 0:
                    # 左边是空的
                    current_col = left_col
                    row += 1
                elif right_height == 0:
                    # 右边是空的
                    current_col = right_col
                    row += 1
                else:
                    # 三个方向都有像素，选择梯度最小的方向
                    # 计算左右两侧的像素密度差
                    left_density = np.sum(result[row, max(0, current_col-3):current_col] > 0)
                    right_density = np.sum(result[row, current_col:min(w, current_col+3)] > 0)
                    
                    if left_density < right_density:
                        current_col = max(0, current_col - 1)
                    elif right_density < left_density:
                        current_col = min(w - 1, current_col + 1)
                    else:
                        # 密度相等，向下
                        pass
                    row += 1
            
            # 4. 沿路径分割（将路径上的像素置为0）
            for r, c in path:
                # 只分割路径附近的像素（避免破坏字符结构）
                for dc in range(-1, 2):
                    nc = c + dc
                    if 0 <= nc < w:
                        result[r, nc] = 0
        
        return result
    
    # ==================== 第四阶段：形态学处理输出 ====================
    def stage4_morphology_prepare(self, image: np.ndarray = None) -> Optional[np.ndarray]:
        """
        第四阶段：形态学处理输出
        执行开运算、膨胀、闭运算等形态学操作，连接点阵字符成块，
        并保存结果为 output/morphology_result.png
        
        Args:
            image: 输入图像（二值化），如果为 None 则使用 self.enhanced_image
            
        Returns:
            形态学处理后的图像，如果处理失败则返回 None
        """
        if image is None:
            if self.enhanced_image is None:
                print("  错误: 没有可用的增强图像")
                return None
            image = self.enhanced_image
        
        height, width = image.shape
        print(f"  开始形态学处理: 图像尺寸 {width}x{height}")
        
        # 检查图像是否需要反转（确保字符为白色，背景为黑色）
        # 计算白色像素比例
        white_pixels = np.sum(image == 255)
        total_pixels = image.size
        white_ratio = white_pixels / total_pixels
        print(f"  图像白色像素比例: {white_ratio:.3f}")
        
        working_image = image.copy()
        # 如果白色像素比例 > 0.5，可能背景是白色，需要反转
        if white_ratio > 0.5:
            print("  图像反转（使字符为白色，背景为黑色）")
            working_image = cv2.bitwise_not(working_image)
        
        # 1. 形态学操作强制连接点阵字符成块
        # 首先进行开运算去除小噪声
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, self._get_kernel_size('stage5_open_kernel'))
        opened_image = cv2.morphologyEx(working_image, cv2.MORPH_OPEN, kernel_open, 
                                       iterations=self.cfg['stage5_open_iterations'])
        
        # 滴水算法：分离粘连的字符（在腐蚀之前）
        separated_image = self.waterfall_segmentation(opened_image)
        
        # 横向腐蚀：收缩点阵字符的横向宽度，使字符间隙更清晰，便于垂直投影分割
        kernel_h_erosion = cv2.getStructuringElement(cv2.MORPH_RECT, self._get_kernel_size('stage5_h_erosion_kernel'))
        eroded_image = cv2.erode(separated_image, kernel_h_erosion,
                                 iterations=self.cfg['stage5_h_erosion_iterations'])

        # 纵向腐蚀：收缩点阵字符的纵向高度，分开纵向粘连的字符
        kernel_v_erosion = cv2.getStructuringElement(cv2.MORPH_RECT, self._get_kernel_size('stage5_v_erosion_kernel'))
        eroded_image = cv2.erode(eroded_image, kernel_v_erosion,
                                 iterations=self.cfg['stage5_v_erosion_iterations'])
        
        # 再进行闭运算填充内部空隙，形成字符块
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, self._get_kernel_size('stage5_close_kernel'))
        closed_image = cv2.morphologyEx(eroded_image, cv2.MORPH_CLOSE, kernel_close, 
                                       iterations=self.cfg['stage5_close_iterations'])
        
        # 保存形态学处理后的图像（根据 cfg['debug'] 决定是否写盘）
        if self.cfg.get('debug', False):
            cv2.imwrite(os.path.join(self.output_dir, "morphology_result.png"), closed_image)
            print(f"  形态学处理结果已保存到 morphology_result.png")
        else:
            print(f"  形态学处理结果已生成（未写入磁盘，debug=False）")
        self.morph_image = closed_image  # 保存 morphology 结果用于后续检测
        
        return closed_image
    
    # ==================== 完整处理流程 ====================
    def full_pipeline(self, image_path: str) -> str:
        """
        简化处理流程：只运行第一阶段
        Args:
            image_path: 输入图像路径
            
        Returns:
            处理状态字符串
        """
        print("=== 开始处理图像（简化版） ===")
        # 第一阶段：胎面 ROI 提取
        enhanced = self.stage1_extract_roi(image_path)
        if enhanced is None:
            print("处理完成: 未发现编号区域")
            return "无编号区域"
        
        if self.cfg.get('debug', False):
            print(f"处理完成: 生成增强图像 {self.enhanced_image.shape}，已保存为 3_enhanced.png")
        else:
            print(f"处理完成: 生成增强图像 {self.enhanced_image.shape}（未写入磁盘）")
        return "成功生成增强图像"
    
    def full_contours_pipeline(self, image_path: str) -> str:
        """
        完整轮廓检测处理流程：运行所有阶段，生成完整输出
        Args:
            image_path: 输入图像路径
            
        Returns:
            处理状态字符串（非识别结果）
        """
        print("=== 开始完整轮廓检测处理流程 ===")
        
        # 第一阶段：ROI提取与预处理
        print("\n=== 第一阶段：ROI提取 ===")
        enhanced = self.stage1_extract_roi(image_path)
        if enhanced is None:
            print("第一阶段失败：未发现编号区域")
            return ""
        
        if self.cfg.get('debug', False):
            print("第一阶段完成，生成: 3_enhanced.png")
        else:
            print("第一阶段完成，增强图像已生成（未写入磁盘）")
        
        # 第二阶段：标签区域定位
        print("\n=== 第二阶段：标签区域定位 ===")
        label_binary = self.stage2_label_localization()
        if label_binary is None:
            print("第二阶段失败：无法定位标签区域")
            return ""
        
        print("第二阶段完成")
        
        # 第四阶段：形态学处理输出
        print("\n=== 第四阶段：形态学处理输出 ===")
        morphology_result = self.stage4_morphology_prepare()
        
        if morphology_result is not None:
            print(f"  形态学处理完成，结果尺寸: {morphology_result.shape}")
            print("第四阶段完成，生成: morphology_result.png")
            print(f"处理完成，所有输出文件保存在: {self.output_dir}/")
            return "形态学处理完成"
        else:
            print("第四阶段失败：形态学处理异常")
            return ""


def main():
    """主函数，演示完整的前处理流程（截止形态学输出）"""
    import sys
    import argparse
    
    # 检查参数
    if len(sys.argv) < 2:
        print("用法: python tire_ocr.py <图像路径或目录> [模式] [-debug] [-s]")
        print("示例:")
        print("  python tire_ocr.py data/                         # 遍历 data/ 下的 .bmp/.png 文件，运行完整流程")
        print("  python tire_ocr.py 20250115083551210.bmp simple  # 只运行第一阶段（ROI提取）")
        print("  python tire_ocr.py 20250115083551210.bmp full    # 运行完整流程（至形态学输出）")
        print("  python tire_ocr.py 20250115083551210.bmp full -debug  # 运行完整流程并启用调试模式")
        print("  python tire_ocr.py 20250115083551210.bmp full -debug -s  # 运行完整流程并保存分割字符到分类目录")
        sys.exit(1)

    path_arg = sys.argv[1]
    mode = "full"
    debug_mode = False
    save_split_mode = False
    
    # 检查是否包含 -debug 和 -s 参数
    original_args = sys.argv[1:]  # 排除脚本名
    if "-debug" in original_args:
        debug_mode = True
    if "-s" in original_args:
        save_split_mode = True
    
    # 移除 -debug 和 -s 参数，然后重新解析
    args_copy = [arg for arg in original_args if arg != "-debug" and arg != "-s"]
    if len(args_copy) > 0:
        path_arg = args_copy[0]
    if len(args_copy) > 1:
        mode = args_copy[1].lower()

    # If a directory is provided, iterate its .bmp/.png files and process each.
    def process_file(image_path: str, mode: str, preloaded_image: Optional[np.ndarray] = None, debug_mode: bool = False, save_split_mode: bool = False):
        display_name = image_path if isinstance(image_path, str) else '<in-memory>'
        print(f"\n--- 处理文件: {display_name} (模式={mode}) ---")
        ocr = TireOCR()
        # 设置 debug 模式
        ocr.cfg['debug'] = debug_mode
        try:
            # If preloaded_image is provided, pass it through to stage1 via the pipeline
            entry = preloaded_image if preloaded_image is not None else image_path
            if mode == "simple":
                print("=== 运行简化流程（只处理第一阶段）===")
                result = ocr.full_pipeline(entry)
            else:
                print("=== 运行完整流程（至形态学输出）===")
                result = ocr.full_contours_pipeline(entry)

            if ocr.enhanced_image is None:
                print("检测到第一阶段失败：未发现编号区域，跳过后续处理。")
                return

            # 尝试调用内置后处理（如果存在）——与之前逻辑相同
            try:
                post_mod_path = os.path.join(os.path.dirname(__file__), "post_vertical_projection.py")
                if os.path.exists(post_mod_path):
                    try:
                        import post_vertical_projection as pvp
                    except Exception:
                        import importlib.util
                        spec = importlib.util.spec_from_file_location("post_vertical_projection", post_mod_path)
                        pvp = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(pvp)

                    try:
                        pvp.CFG['DEBUG_DRAW_SEGMENTS'] = ocr.cfg.get('debug', False)
                    except Exception:
                        pass

                    morph_img = getattr(ocr, 'morph_image', None)
                    enhanced_img = getattr(ocr, 'enhanced_image', None)
                    if morph_img is None or enhanced_img is None:
                        print("后处理跳过：缺少中间产物（morph_image 或 enhanced_image）")
                    else:
                        _, proj_morph = pvp.vertical_projection(morph_img, None)
                        peaks_for_cut = pvp.detect_and_cluster_peaks(
                            proj_morph,
                            binary_thresh_factor=pvp.CFG.get("BINARY_THRESH_FACTOR", 0.02),
                            merge_distance=pvp.CFG.get("MERGE_DISTANCE", 20),
                            min_width=pvp.CFG.get("MIN_WIDTH", 5),
                            gaussian_k=pvp.CFG.get("GAUSSIAN_KERNEL", 11),
                            gaussian_sigma=pvp.CFG.get("GAUSSIAN_SIGMA", 2.0),
                        )
                        print(f"\nSegmenting characters using {len(peaks_for_cut)} peaks...")
                        # 设置调试参数
                        try:
                            pvp.CFG['DEBUG_DRAW_SEGMENTS'] = ocr.cfg.get('debug', False)
                        except Exception:
                            pass
                        
                        # 如果 enhanced_image 存在，直接使用图像数据而不是文件路径
                        if enhanced_img is not None:
                            summary, char_images = pvp.cut_characters(peaks_for_cut, enhanced_img, os.path.join(ocr.output_dir, 'characters'))
                        else:
                            # 如果没有图像数据，尝试从文件加载
                            try:
                                summary, char_images = pvp.cut_characters(peaks_for_cut, os.path.join(ocr.output_dir, '3_enhanced.png'), os.path.join(ocr.output_dir, 'characters'))
                            except TypeError:
                                summary, char_images = pvp.cut_characters(peaks_for_cut, enhanced_img, os.path.join(ocr.output_dir, 'characters'))
                        
                        # 运行KNN识别（如果存在字符图像）
                        try:
                            if 'char_images' in locals() and char_images:
                                print("Running in-file KNN recognition on in-memory images...")
                                try:
                                    pvp.run_knn_recognition(char_images=char_images, save_split_mode=save_split_mode)
                                except Exception as _knn_err:
                                    print(f"run_knn_recognition() failed: {_knn_err}")
                            else:
                                # 回退到扫描磁盘目录
                                cut_dir = os.path.join(ocr.output_dir, 'characters')
                                if os.path.exists(cut_dir) and any(f.lower().endswith(('.png', '.bmp')) for f in os.listdir(cut_dir)):
                                    print("Running in-file KNN recognition (from disk)...")
                                    try:
                                        pvp.run_knn_recognition(save_split_mode=save_split_mode)
                                    except Exception as _knn_err:
                                        print(f"run_knn_recognition() failed: {_knn_err}")
                                else:
                                    print("KNN skipped: no in-memory images and 'output/characters' not found or empty.")
                        except Exception as _e:
                            print(f"KNN execution check failed: {_e}")
                else:
                    print("未找到 post_vertical_projection.py，跳过后处理。")
            except Exception as ex:
                print(f"内置后处理失败: {ex}")
        except Exception as e:
            print(f"处理文件失败: {display_name}, 错误: {e}")

    if os.path.isdir(path_arg):
        # gather bmp/png files
        files = sorted(os.listdir(path_arg))
        images = [os.path.join(path_arg, f) for f in files if f.lower().endswith(('.bmp', '.png'))]
        if not images:
            print(f"目录 {path_arg} 中未找到 .bmp 或 .png 文件")
            sys.exit(1)

        # Buffered producer-consumer: preload up to buffer_size images
        buffer_size = 20
        q = queue.Queue(maxsize=buffer_size)

        def producer(file_list):
            for fp in file_list:
                try:
                    img = cv2.imread(fp)
                    if img is None:
                        print(f"预加载失败，无法读取: {fp}")
                        continue
                    # block if queue full
                    q.put((fp, img))
                except Exception as ex:
                    print(f"预加载出错: {fp}, {ex}")
            # signal end
            q.put(None)

        prod_thread = threading.Thread(target=producer, args=(images,), daemon=True)
        prod_thread.start()

        # consume
        while True:
            item = q.get()
            if item is None:
                break
            fp, img = item
            process_file(fp, mode, preloaded_image=img, debug_mode=debug_mode, save_split_mode=save_split_mode)
        # ensure producer finished
        prod_thread.join()
    else:
        # single file
        if not os.path.exists(path_arg):
            print(f"文件不存在: {path_arg}")
            sys.exit(1)
        process_file(path_arg, mode, debug_mode=debug_mode, save_split_mode=save_split_mode)
    


if __name__ == "__main__":
    main()