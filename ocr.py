"""精简的 OCR 模块：基于投影波高匹配的字符识别。

公共函数：
- `prepare_image_array(img) -> np.ndarray`：将输入规范为灰度 uint8 numpy 数组
- `run_knn_recognition(char_images=None) -> list`：接受内存中的字符图像列表
    （列表项为包含 'name' 和 'img' 的字典）并返回识别结果。
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional


# --------- OCR 配置超参数 ---------
# 所有 OCR 识别相关的可调参数在此处定义
CFG = {
    # 峰值检测相关参数
    "PEAK_WINDOW": 2,                  # 用于计算导数均值的窗口大小
    "PEAK_MIN_HEIGHT_THRESHOLD": 0.05, # 最小峰值高度阈值（相对于最大值的比例）
    "PEAK_MERGE_DISTANCE": 10,          # 合并相邻峰值的距离阈值
    
    # 投影平滑参数
    "SMOOTH_KERNEL_SIZE": 5,           # 高斯平滑核大小（奇数）
    "SMOOTH_SIGMA": 0.8,               # 高斯核标准差
    
    # 误差计算参数
    "MAX_DEVIATION": 0.2,              # 最大允许偏差（已弃用，保留用于兼容性）
    "MAX_RATIO_THRESHOLD": 2,        # 波高比例阈值（超过此值返回inf惩罚）
    
    # 其他参数
    "SCORE_THRESHOLD": 15.0,           # 识别阈值，超过此值输出"无法识别"
    "CLIST_PATH": "Clist.txt",         # 字符模板文件路径
}


# 全局模板缓存
_TEMPLATE_CACHE: Optional[Dict[str, Dict[str, List[int]]]] = None


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


def load_clist(clist_path: str = 'Clist.txt') -> Dict[str, Dict[str, List[int]]]:
    """解析 Clist.txt 文件，加载字符模板特征。

    文件格式示例：
        3:((2:2:3:4:3),(5:1:1:1:1:2:3))
    表示数字3的特征：
        - x方向投影波高度：[2, 2, 3, 4, 3]
        - y方向投影波高度：[5, 1, 1, 1, 1, 2, 3]

    返回字典：
        {
            '3': {'x': [2, 2, 3, 4, 3], 'y': [5, 1, 1, 1, 1, 2, 3]},
            ...
        }
    """
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
                feature_str = parts[1].strip()
                # 提取两个括号内的内容
                if not (feature_str.startswith('((') and feature_str.endswith('))')):
                    continue
                inner = feature_str[2:-2]  # 去掉外层的 (( 和 ))
                # 分割x和y方向
                inner_parts = inner.split('),(')
                if len(inner_parts) != 2:
                    continue
                x_str = inner_parts[0]
                y_str = inner_parts[1]
                # 解析波高序列
                x_ratios = [int(x.strip()) for x in x_str.split(':') if x.strip().isdigit()]
                y_ratios = [int(y.strip()) for y in y_str.split(':') if y.strip().isdigit()]
                if x_ratios and y_ratios:
                    templates[label] = {'x': x_ratios, 'y': y_ratios}
    except Exception as e:
        print(f"[load_clist] Error parsing Clist.txt: {e}")

    print(f"[load_clist] Loaded {len(templates)} character templates")
    return templates


def get_templates(clist_path: str = 'Clist.txt') -> Dict[str, Dict[str, List[int]]]:
    """获取字符模板（带缓存）。"""
    global _TEMPLATE_CACHE
    if _TEMPLATE_CACHE is None:
        _TEMPLATE_CACHE = load_clist(clist_path)
    return _TEMPLATE_CACHE


def compute_projection(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """计算字符图像的水平和垂直投影。

    参数：
        img: 二值图像（uint8，0为背景，255为前景）

    返回：
        (proj_x, proj_y):
            - proj_x: 水平投影（每列白色像素占比），形状 (W,)
            - proj_y: 垂直投影（每行白色像素占比），形状 (H,)
    """
    # 确保为二值图像
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    if img.ndim == 3:
        # 转为灰度
        img = np.mean(img, axis=2).astype(np.uint8)

    h, w = img.shape
    mask = (img > 0).astype(np.uint8)

    # 水平投影：沿行方向求和，得到每列的白色像素数
    proj_x = np.sum(mask, axis=0).astype(float) / max(1, h)
    # 垂直投影：沿列方向求和，得到每行的白色像素数
    proj_y = np.sum(mask, axis=1).astype(float) / max(1, w)

    return proj_x, proj_y


def smooth_projection_1d(proj: np.ndarray, kernel_size: int = None, sigma: float = None) -> np.ndarray:
    """对一维投影进行高斯平滑。"""
    if kernel_size is None:
        kernel_size = CFG.get("SMOOTH_KERNEL_SIZE", 7)
    if sigma is None:
        sigma = CFG.get("SMOOTH_SIGMA", 1.0)
    
    if proj.size == 0:
        return proj
    if kernel_size % 2 == 0:
        kernel_size += 1
    half = kernel_size // 2
    xs = np.arange(-half, half + 1)
    gauss = np.exp(-(xs.astype(float) ** 2) / (2.0 * sigma ** 2))
    gauss = gauss / np.sum(gauss)
    return np.convolve(proj.astype(float), gauss, mode='same')


def find_peaks_1d(proj: np.ndarray, window: int = None, min_height_threshold: float = None,
                  merge_distance: int = None) -> List[int]:
    """在一维数组中查找局部极大值的索引（基于导数符号变化）。

    参数：
        proj: 一维投影数组
        window: 用于计算导数均值的窗口大小
        min_height_threshold: 最小峰值高度阈值（相对于最大值的比例）
        merge_distance: 合并相邻峰值的距离阈值

    返回：
        峰值索引列表
    """
    # 从 CFG 读取参数
    if window is None:
        window = CFG.get("PEAK_WINDOW", 2)
    if min_height_threshold is None:
        min_height_threshold = CFG.get("PEAK_MIN_HEIGHT_THRESHOLD", 0.05)
    if merge_distance is None:
        merge_distance = CFG.get("PEAK_MERGE_DISTANCE", 3)
    
    if proj.size < 3:
        return []

    # 平滑
    smoothed = smooth_projection_1d(proj)

    # 计算导数
    d = np.diff(smoothed)
    n = len(smoothed)
    peaks = []

    # 计算最大值作为阈值基准
    max_val = float(np.max(smoothed)) if np.max(smoothed) > 0 else 1.0
    threshold = max_val * min_height_threshold

    for j in range(1, n - 1):
        # 左侧导数窗口
        left_start = max(0, j - window)
        left = d[left_start:j] if j > left_start else np.array([], dtype=float)
        # 右侧导数窗口
        right_end = min(len(d), j + window)
        right = d[j:right_end] if right_end > j else np.array([], dtype=float)

        left_mean = float(np.mean(left)) if left.size > 0 else 0.0
        right_mean = float(np.mean(right)) if right.size > 0 else 0.0

        # 导数由正变负，且当前值大于阈值
        if left_mean > 0 and right_mean < 0 and smoothed[j] > threshold:
            peaks.append(j)

    # 合并相邻的峰值（距离小于merge_distance的峰值只保留最高的一个）
    if not peaks:
        return []

    merged_peaks = []
    i = 0
    while i < len(peaks):
        # 找到一组相邻的峰值
        group_start = i
        while i + 1 < len(peaks) and peaks[i + 1] - peaks[i] <= merge_distance:
            i += 1
        group_end = i

        # 在这组中找到值最大的峰值索引
        group = peaks[group_start:group_end + 1]
        best_idx = max(group, key=lambda idx: smoothed[idx])
        merged_peaks.append(best_idx)
        i += 1

    return merged_peaks


def extract_peak_heights(proj: np.ndarray, peak_indices: List[int]) -> List[float]:
    """提取峰值高度序列。

    参数：
        proj: 投影数组
        peak_indices: 峰值索引列表

    返回：
        峰值高度列表（归一化后的值）
    """
    if not peak_indices:
        return []

    heights = [float(proj[i]) for i in peak_indices]
    max_h = max(heights) if heights else 1.0
    if max_h == 0:
        return heights

    # 归一化到 [0, 1]
    normalized = [h / max_h for h in heights]
    return normalized


def compute_error(detected: List[float], template: List[int], max_deviation: float = None, max_ratio_threshold: float = None) -> float:
    """计算检测到的波高序列与模板之间的误差。

    规则：
    - 长度不一致：返回指数惩罚（10^长度差）
    - 两个序列都归一化到最小值为1.0，然后逐元素比对
    - 直接累加逐元素绝对差之和
    - 不使用容差检查和指数惩罚

    参数：
        detected: 检测到的波高序列（原始高度，已归一化到[0,1]）
        template: 模板波高序列（整数，表示相对波高）
        max_deviation: 此参数已弃用，保留用于兼容性

    返回：
        误差值（越小越好）
    """
    # 从 CFG 读取参数
    if max_deviation is None:
        max_deviation = CFG.get("MAX_DEVIATION", 0.2)
    if max_ratio_threshold is None:
        max_ratio_threshold = CFG.get("MAX_RATIO_THRESHOLD", 2.0)
    
    if not template or not detected:
        return 1e3

    # 长度不一致的处理
    if len(detected) != len(template):
        length_diff = abs(len(detected) - len(template))
        # 长度差=1时尝试匹配前len(template)个或后len(template)个峰
        if length_diff == 1:
            detected_arr = np.array(detected, dtype=float)
            template_float = np.array(template, dtype=float)
            
            # 归一化检测序列
            min_d = float(np.min(detected_arr))
            if min_d == 0:
                max_d = float(np.max(detected_arr))
                if max_d == 0:
                    return 1e3
                detected_norm = detected_arr / max_d
            else:
                detected_norm = detected_arr / min_d
            
            # 归一化模板
            min_t = float(np.min(template_float))
            if min_t == 0:
                max_t = float(np.max(template_float))
                if max_t == 0:
                    return 1e3
                template_norm = template_float / max_t
            else:
                template_norm = template_float / min_t
            
            # 尝试匹配前len(template)个峰或后len(template)个峰
            if len(detected) > len(template):
                # 检测到的峰比模板多，尝试前len(template)个或后len(template)个
                detected_front = detected_norm[:len(template)]
                detected_back = detected_norm[-len(template):]
            else:
                # 检测到的峰比模板少，用模板的前len(detected)个或后len(detected)个
                template_front = template_norm[:len(detected)]
                template_back = template_norm[-len(detected):]
                detected_front = detected_norm
                detected_back = detected_norm
                # 重新计算误差
                error_front = np.sum(np.abs(detected_front - template_front))
                error_back = np.sum(np.abs(detected_back - template_back))
                return min(error_front, error_back)
            
            # 计算前段和后段的匹配误差
            error_front = np.sum(np.abs(detected_front - template_norm))
            error_back = np.sum(np.abs(detected_back - template_norm))
            
            # 取最小误差
            return min(error_front, error_back)
        else:
            # 长度差>=2时使用指数惩罚
            return 3.0 ** length_diff

    # 归一化 detected（最小值为1.0）
    detected_arr = np.array(detected, dtype=float)
    min_d = float(np.min(detected_arr))
    if min_d == 0:
        # 如果最小值为0，无法归一化到最小值为1，改用最大值归一化
        max_d = float(np.max(detected_arr))
        if max_d == 0:
            return 1e3
        detected_norm = detected_arr / max_d
    else:
        detected_norm = detected_arr / min_d

    # 归一化模板（最小值为1.0）
    template_float = np.array(template, dtype=float)
    min_t = float(np.min(template_float))
    if min_t == 0:
        # 如果最小值为0，无法归一化到最小值为1，改用最大值归一化
        max_t = float(np.max(template_float))
        if max_t == 0:
            return 1e3
        template_norm = template_float / max_t
    else:
        template_norm = template_float / min_t

    # 计算逐元素差并累加
    diffs = np.abs(detected_norm - template_norm)
    
    # 检查是否有差异过大的元素（比例超过阈值）
    for i, (d, t) in enumerate(zip(detected_norm, template_norm)):
        if t > 0:  # 避免除零
            ratio = d / t
            if ratio >= max_ratio_threshold or ratio <= (1.0 / max_ratio_threshold):
                # 差异过大，返回无限大惩罚
                return float('inf')
    
    return float(np.sum(diffs))


def match_character(proj_x: np.ndarray, proj_y: np.ndarray,
                    templates: Dict[str, Dict[str, List[int]]],
                    max_deviation: float = None,
                    min_height_threshold: float = None) -> Tuple[str, float, List[int], List[int]]:
    """匹配字符，返回最匹配的标签和误差。

    参数：
        proj_x: 水平投影
        proj_y: 垂直投影
        templates: 字符模板字典
        max_deviation: 最大允许偏差（如果为None则从CFG读取）
        min_height_threshold: 最小峰值高度阈值（如果为None则从CFG读取）

    返回：
        (label, score, peak_x_indices, peak_y_indices)
    """
    # 从 CFG 读取参数
    if max_deviation is None:
        max_deviation = CFG.get("MAX_DEVIATION", 0.2)
    if min_height_threshold is None:
        min_height_threshold = CFG.get("PEAK_MIN_HEIGHT_THRESHOLD", 0.05)
    max_ratio_threshold = CFG.get("MAX_RATIO_THRESHOLD", 2.0)
    
    # 查找峰值（使用阈值过滤噪声）
    peak_x_indices = find_peaks_1d(proj_x, min_height_threshold=min_height_threshold)
    peak_y_indices = find_peaks_1d(proj_y, min_height_threshold=min_height_threshold)

    # 提取波高序列
    heights_x = extract_peak_heights(proj_x, peak_x_indices)
    heights_y = extract_peak_heights(proj_y, peak_y_indices)

    if not templates:
        return ('?', float('inf'), peak_x_indices, peak_y_indices)

    best_label = '?'
    best_score = float('inf')

    # 对每个模板进行匹配
    for label, template in templates.items():
        template_x = template.get('x', [])
        template_y = template.get('y', [])

        # 计算x方向误差
        error_x = compute_error(heights_x, template_x, max_deviation, max_ratio_threshold)
        # 计算y方向误差
        error_y = compute_error(heights_y, template_y, max_deviation, max_ratio_threshold)

        # 综合误差（加权平均）
        total_error = error_x + error_y

        if total_error < best_score:
            best_score = total_error
            best_label = label

    return (best_label, best_score, peak_x_indices, peak_y_indices)


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
    """运行基于投影波高匹配的字符识别。

    接受 `char_images`，应为由 `post_vertical_projection.cut_characters(...)` 生成的
    列表，列表项形如 `{'name': str, 'img': np.ndarray, 'meta': {...}}`。

    返回列表，元素为元组 (filename, label, score, peak_x_indices, peak_y_indices)。

    识别流程：
    1. 加载Clist.txt模板
    2. 对每个字符图像：
       - 检查是否是表示空格的特殊数组
       - 如果是空格，直接返回空格字符
       - 否则转换为灰度并二值化
       - 计算水平和垂直投影
       - 查找峰值并提取波高序列
       - 与模板进行匹配
    3. 返回识别结果
    """
    results = []
    if not char_images:
        return results

    # 加载模板
    templates = get_templates('Clist.txt')
    if not templates:
        print("[run_knn_recognition] No templates loaded, recognition disabled")
        return results

    print(f"[run_knn_recognition] Processing {len(char_images)} characters with {len(templates)} templates...")

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
            results.append((name, '?', float('inf'), [], []))
            continue

        # 二值化（使用简单阈值）
        _, binary = cv2.threshold(img_arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 计算投影
        try:
            proj_x, proj_y = compute_projection(binary)
        except Exception as e:
            print(f"  [{idx}] {name}: Projection failed - {e}")
            results.append((name, '?', float('inf'), [], []))
            continue

        # 匹配字符
        label, score, peak_x, peak_y = match_character(proj_x, proj_y, templates)

        results.append((name, label, score, peak_x, peak_y))

        # 打印结果
        score_threshold = CFG.get("SCORE_THRESHOLD", 10.0)
        if score > score_threshold:
            display_label = "无法识别"
            print(f"  [{idx}] {name}: '{display_label}' (score={score:.4f}, x_peaks={len(peak_x)}, y_peaks={len(peak_y)})")
        elif label != '?':
            print(f"  [{idx}] {name}: '{label}' (score={score:.4f}, x_peaks={len(peak_x)}, y_peaks={len(peak_y)})")
        else:
            print(f"  [{idx}] {name}: unmatched (score={score:.4f})")

    # 统计结果
    score_threshold = CFG.get("SCORE_THRESHOLD", 10.0)
    matched = sum(1 for r in results if r[1] != '?' and r[2] <= score_threshold)
    print(f"\n[run_knn_recognition] Matched {matched}/{len(results)} characters")

    # 生成完整识别结果字符串
    full_recognition_result = ""
    for name, label, score, peak_x, peak_y in results:
        if score > score_threshold:
            # 不将"无法识别"添加到完整结果中，或者用特殊符号代替
            full_recognition_result += "?"
        else:
            full_recognition_result += label

    print(f"[run_knn_recognition] Full recognition result: '{full_recognition_result}'")

    # 写入结果文件
    output_path = 'output/characters/recognition_result.txt'
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Character Recognition Results\n")
            f.write("# Format: filename, label, score, x_peaks, y_peaks\n")
            f.write(f"# Full recognition result: {full_recognition_result}\n")
            score_threshold = CFG.get("SCORE_THRESHOLD", 10.0)
            for name, label, score, peak_x, peak_y in results:
                if score > score_threshold:
                    display_label = "无法识别"
                else:
                    display_label = label
                f.write(f"{name}, {display_label}, {score:.4f}, {len(peak_x)}, {len(peak_y)}\n")
        print(f"[run_knn_recognition] Results saved to {output_path}")
    except Exception as e:
        print(f"[run_knn_recognition] Failed to save results: {e}")

    return results