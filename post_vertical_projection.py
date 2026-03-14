import os
import sys
import cv2
import numpy as np



# --------- 配置（可调超参数） ---------
# 所有重要的绘图 / 峰值检测 参数在此处定义。
CFG = {
    # 绘图尺寸与输出参数
    "FIGSIZE_PER_COL": 6,         # 每列子图的宽度（英寸）；用于生成投影可视化图像时计算总宽度
    "FIG_HEIGHT": 6,              # 绘图总体高度（英寸）
    "HEIGHT_RATIOS": [2, 1],      # 子图行高度比：上行显示原图（2），下行显示投影（1）
    "PLOT_DPI": 100,              # 保存绘图时的分辨率（DPI）

    # 投影/峰值检测基础阈值与平滑参数
    # BINARY_THRESH_FACTOR: 在增强后的投影上将小值置零时使用的相对阈值（相对于当前曲线最大值）
    #                阈值计算为 factor * max(projection)。较小值会保留更多弱峰，较大值会过滤弱噪声。
    "BINARY_THRESH_FACTOR": 0.01,

    # MERGE_DISTANCE: 在基于连通区间的分段处理中，如果两个连通区间之间的列距离小于该值，则会合并为同一区间。
    #                单位：像素列。用于消除因微小间隙导致的过度分割。
    "MERGE_DISTANCE": 3,

    # MIN_WIDTH: 过滤噪声的最小宽度（像素列），连通区间宽度小于该值会被丢弃为噪声。
    "MIN_WIDTH": 5,

    # GAUSSIAN_KERNEL / GAUSSIAN_SIGMA: 用于对 1D 投影进行高斯平滑的核大小与标准差。
    #    GAUSSIAN_KERNEL: 必须为奇数，越大平滑越强、细节越少。单位：采样点（列或行）。
    #    GAUSSIAN_SIGMA: 高斯核的标准差，控制平滑程度（相对核宽）。
    "GAUSSIAN_KERNEL": 9,
    "GAUSSIAN_SIGMA": 1.0,

    # SMOOTH_ZERO_THRESHOLD: 在对投影做非线性增强与局部对比度计算后，可用此绝对阈值把小值直接置零。
    #                       值应在 0..1 区间内，表示投影归一化后的绝对下限。若为 None 则回退到使用相对阈值 BINARY_THRESH_FACTOR。
    "SMOOTH_ZERO_THRESHOLD": 0.02,

    # NONLINEAR_GAMMA: 对平滑曲线做非线性重缩放的 gamma 值（幂律），通常 <1 用于抬升低幅峰、压缩高幅峰。
    #                  取值范围建议约 0.1 .. 1.5；较小会增强弱峰，但可能放大噪声。
    "NONLINEAR_GAMMA": 0.1,

    # LOCAL_CONTRAST_WINDOW: 用于局部对比度计算的滑动窗口大小（采样点数，建议为奇数）。
    #                        窗口越大，对比度增强越平滑；窗口越小，局部对比越敏感。
    "LOCAL_CONTRAST_WINDOW": 31,

        # 字符切割 / 峰分组相关阈值（单位均为像素或归一化比率）
# CHAR_SEP_GAP: 两个峰之间若间距 > CHAR_SEP_GAP，则视为字符边界（分隔符/强制断点）。
#               该值影响字符分段和强制分割的边界判定。
    "MIN_GAP": 8,            # 相邻峰间距小于此值则合并为一个峰（像素）
    "CHAR_SEP_GAP": 40,      # 相邻峰间距大于此值则强制分割（像素）
    "SPACE_THRESHOLD": 120,   # 相邻峰间距大于此值则认为空格（像素）

    # PNG_COMPRESSION: 写入 PNG 时使用的 zlib 压缩级别（0-9）。
    #                  0 表示不压缩（最快），9 表示最大压缩（最慢但文件小）。
    "PNG_COMPRESSION": 0,    # 0 => no compression (fast), 9 => max compression (smallest)

    # 输入/输出目录与文件名
    "INPUT_CHARS_DIR": 'output/characters',                           # 默认从此目录读取字符样本（用于识别/模板匹配）
    "OUTPUT_TXT": 'output/characters/recognition_result.txt',         # 识别结果写入路径（文本）
    "PROJ_OUT_DIR": 'output/characters/projections',                  # 每字符投影可视化图像输出目录

    # 调试开关：是否绘制并保存带分段矩形与标签的调试图像（segmentation_debug.png）
    "DEBUG_DRAW_SEGMENTS": True,

    # 每字符投影可视化与检测的独立平滑参数（便于对单字符投影做更细粒度控制）
    "CHAR_PROJ_GAUSSIAN_KERNEL": 11,    # 单字符投影平滑核大小（奇数）
    "CHAR_PROJ_GAUSSIAN_SIGMA": 3.0,    # 单字符投影高斯标准差
    # CHAR_PROJ_SMOOTH_ZERO_THRESHOLD: 对单字符平滑曲线应用的绝对阈值（若不为 None 则小于该值设为 0）
    "CHAR_PROJ_SMOOTH_ZERO_THRESHOLD": 0.02,
    # 最小宽度：当在单字符投影上检测连通区时的最小保留宽度
    "CHAR_PROJ_MIN_WIDTH": 2,

    # PEAK_SIGN_WINDOW: 在使用导数符号变化检测极值点时使用的窗口大小（采样点数）。
    # 如果为 1 则保持原有行为（检查 d[j-1] 和 d[j]）；更大的值会对导数在左右窗口内取均值后再判断符号变化，
    # 有助于提高在嘈杂信号上的鲁棒性（避免由单点噪声造成的误检）。
    "PEAK_SIGN_WINDOW": 2,
}

# --------------------------------------------------------------


def plot_vertical_projection_with_peaks(image_path, projection, peaks, segments=None, out_path=None):
    """
    使用OpenCV绘制垂直投影图，标记检测到的峰值和字符分段
    
    参数:
        image_path: 原始图像路径或numpy数组
        projection: 垂直投影数组
        peaks: 检测到的峰值列表 [(x, height), ...]
        segments: 字符分段信息，可选
        out_path: 输出图像路径
    """
    # 读取原始图像
    if isinstance(image_path, str):
        original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        original_img = image_path
        if original_img.ndim == 3:
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    
    if original_img is None:
        print("无法读取原始图像")
        return

    # 定义图像尺寸参数
    img_height, img_width = original_img.shape
    proj_height = 200  # 投影图的高度
    combined_height = img_height + proj_height  # 组合图像的高度
    combined_img = np.ones((combined_height, max(img_width, len(projection)), 3), dtype=np.uint8) * 255

    # 转换原始图像为彩色并放置在组合图像的顶部
    original_bgr = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    combined_img[0:img_height, 0:img_width] = original_bgr
    
    # 如果有分段信息，在原始图像上绘制分段矩形
    if segments:
        for i, seg in enumerate(segments):
            x_start = seg['x_start']
            x_end = seg['x_end']
            
            # 创建一个矩形表示字符区域
            cv2.rectangle(combined_img, (x_start, 0), (x_end, img_height), (0, 0, 255), 2)
            
            # 添加分段标签
            cv2.putText(combined_img, f'{i}', (x_start + 2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # 绘制垂直投影
    if len(projection) > 1:
        # 归一化投影值到投影图的高度范围内
        max_proj_val = max(projection) if max(projection) != 0 else 1
        proj_normalized = [int(val / max_proj_val * (proj_height - 20)) for val in projection]
        
        # 将投影绘制到组合图像的底部
        proj_start_y = img_height + 5
        for x in range(1, len(proj_normalized)):
            pt1 = (x - 1, proj_start_y + (proj_height - 20) - proj_normalized[x - 1])
            pt2 = (x, proj_start_y + (proj_height - 20) - proj_normalized[x])
            cv2.line(combined_img, pt1, pt2, (255, 0, 0), 1)

        # 标记峰值
        if peaks:
            for peak_x, peak_val in peaks:
                peak_x = int(peak_x)
                peak_y = int(peak_val / max_proj_val * (proj_height - 20))
                peak_y = proj_start_y + (proj_height - 20) - peak_y
                cv2.circle(combined_img, (peak_x, peak_y), 3, (0, 0, 255), -1)

        # 标记字符分段边界
        if segments:
            for seg in segments:
                # 垂直线在投影图区域
                cv2.line(combined_img, (seg['x_start'], proj_start_y), 
                         (seg['x_start'], proj_start_y + proj_height), (0, 255, 0), 1, cv2.LINE_4)
                cv2.line(combined_img, (seg['x_end'], proj_start_y), 
                         (seg['x_end'], proj_start_y + proj_height), (0, 255, 0), 1, cv2.LINE_4)

    # 添加标题
    cv2.putText(combined_img, 'Vertical Projection with Peaks and Character Segments', 
                (10, img_height + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # 保存图像
    if out_path:
        cv2.imwrite(out_path, combined_img)
        print(f"垂直投影调试图已保存: {out_path}")
    else:
        print("垂直投影调试图已生成")


def vertical_projection(image, out_path=None):
    """计算图像的垂直投影。

    约定（推荐）：传入的 `image` 应为二值图像（numpy.ndarray），形状为 (H, W)，dtype 为 `uint8`，
    前景/白色像素为 255，背景为 0。该函数也接受灰度或（偶发的）彩色图像作为回退输入：
    - 如果传入的是二值图（值域为 {0,1} 或 {0,255}），函数将直接使用该二值图。
    - 否则函数会对灰度图使用 Otsu 二值化以产生前景掩码。

    参数 `image` 可以是文件路径 (str) 或 numpy.ndarray（灰度或彩色）。
    返回值为 `(gray_img, projection_array)`，其中 `gray_img` 为用于计算的灰度/二值图 (H, W) uint8，
    `projection_array` 为长度 W 的浮点数组（每列白色像素占比，范围 0..1）。
    """
    # 如果传入的是路径则加载图像
    if isinstance(image, str):
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image}")
    elif isinstance(image, np.ndarray):
        img = image
        # 如有必要转换为灰度
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            img = img.copy()
    else:
        raise TypeError("vertical_projection expects a file path or numpy.ndarray")

    # 确保为 uint8 类型
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    # 使用 Otsu 二值化：前景（白色）像素为 255
    unique_vals = np.unique(img)
    if unique_vals.size <= 2:
        # 处理 {0,1} -> {0,255} 的情况或已经是 {0,255}
        if unique_vals.max() <= 1:
            bw = (img.astype(np.uint8) * 255).astype(np.uint8)
        else:
            bw = img.astype(np.uint8).copy()
    else:
        # 回退：对灰度图进行 Otsu 二值化以得到前景掩码
        _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 计算每列白色像素占比
    height = bw.shape[0]
    white_mask = (bw == 255).astype(np.uint8)
    projection_counts = np.sum(white_mask, axis=0)
    projection = projection_counts.astype(float) / float(max(1, height))

    return img, projection


def detect_and_cluster_peaks(projection, binary_thresh_factor=None, merge_distance=None, min_width=None,
                             gaussian_k=None, gaussian_sigma=None):
    """基于导数过零点的波峰检测（包含高斯平滑、非线性增强与局部对比度增强）。

    算法说明：
    - 对输入的一维投影使用 1D 高斯核平滑以降低噪声。
    - 对平滑结果做非线性压缩（gamma）并进行局部对比度增强，以提升低幅峰值。
    - 使用阈值把较小的值置零，按连通区间在区间内查找导数由正变负的位置作为峰点候选。
    - 丢弃过窄区间的噪声并在区间内取强度最大的点作为该区间的代表峰。
    - 合并距离小于 MIN_GAP 的相邻峰，取较高者的位置

    参数（可从 CFG 读取默认值）：
    - binary_thresh_factor: 二值阈值因子，阈值 = factor * max(projection)
    - merge_distance: 保留以兼容但当前未显式使用
    - min_width: 最小保留区间宽度（像素）
    - gaussian_k: 高斯核大小（奇数）
    - gaussian_sigma: 高斯核标准差

    返回：列表，每项为 (x_peak, y_value)
    """
    if projection is None or projection.size == 0:
        return []

    max_val = float(np.max(projection))
    if max_val <= 0:
        return []

    if binary_thresh_factor is None:
        binary_thresh_factor = CFG.get("BINARY_THRESH_FACTOR", 0.02)
    if merge_distance is None:
        merge_distance = CFG.get("MERGE_DISTANCE", 20)
    if min_width is None:
        min_width = CFG.get("MIN_WIDTH", 5)
    if gaussian_k is None:
        gaussian_k = CFG.get("GAUSSIAN_KERNEL", 11)
    if gaussian_sigma is None:
        gaussian_sigma = CFG.get("GAUSSIAN_SIGMA", 2.0)

    thresh = float(binary_thresh_factor) * max_val

    # 1D Gaussian smoothing
    k = int(gaussian_k)
    if k % 2 == 0:
        k += 1
    half = k // 2
    xs = np.arange(-half, half + 1)
    gauss = np.exp(-(xs.astype(float) ** 2) / (2.0 * float(gaussian_sigma) ** 2))
    gauss = gauss / np.sum(gauss)
    proj_smooth = np.convolve(projection.astype(float), gauss, mode='same')

    # 非线性重缩放 + 局部对比度增强
    eps = 1e-8
    smooth_max = float(np.max(proj_smooth))
    if smooth_max <= 0:
        return []
    gamma = float(CFG.get("NONLINEAR_GAMMA", 0.65))
    gamma = max(0.1, min(1.5, gamma))
    # gamma < 1: 压缩高峰并抬升低峰，缓解动态范围过大
    nonlinear = np.power(np.clip(proj_smooth / smooth_max, 0.0, 1.0), gamma)

    w = int(CFG.get("LOCAL_CONTRAST_WINDOW", 81))
    if w < 3:
        w += 1
    if w % 2 == 0:
        w += 1
    kernel = np.ones(w, dtype=float) / float(w)
    local_mean = np.convolve(nonlinear, kernel, mode='same')
    local_dev = np.convolve(np.abs(nonlinear - local_mean), kernel, mode='same')
    enhanced = np.maximum(0.0, (nonlinear - local_mean) / (local_dev + eps))
    enh_max = float(np.max(enhanced))
    if enh_max > 0:
        enhanced = enhanced / enh_max

    # 阈值化：优先使用绝对阈值 SMOOTH_ZERO_THRESHOLD（如果设置），否则使用相对阈值 thresh
    proc = enhanced.copy()
    smooth_zero_thresh = CFG.get("SMOOTH_ZERO_THRESHOLD", None)
    if smooth_zero_thresh is not None:
        proc[proc < float(smooth_zero_thresh)] = 0.0
    else:
        proc[proc < float(binary_thresh_factor)] = 0.0

    # 导数符号切换法：导数从正到负的位置作为候选峰值
    # 使用可配置的窗口对导数取局部均值后判断符号变化，以提高在噪声下的鲁棒性
    smoothed = enhanced
    d = np.diff(smoothed)
    peak_sign_window = int(CFG.get('PEAK_SIGN_WINDOW', 1))
    if peak_sign_window < 1:
        peak_sign_window = 1
    peaks = []
    n = len(smoothed)
    i = 0

    # 导数法在细字符场景下区间会偏窄，这里放宽最小宽度约束以避免漏检
    effective_min_width = max(3, int(min_width) // 4)

    # 先按阈值化后的非零区间遍历，再在区间内找导数过零点
    while i < n:
        if proc[i] == 0.0:
            i += 1
            continue
        start = i
        while i < n and proc[i] != 0.0:
            i += 1
        end = i - 1
        if (end - start + 1) < effective_min_width:
            continue

        # 在 [start+1, end-1] 内找导数由正到负的点（对应峰索引 j）
        local_peaks = []
        lo = max(start + 1, 1)
        hi = min(end - 1, n - 2)
        for j in range(lo, hi + 1):
            # 计算导数在左右窗口内的均值：左侧为 d[j-window .. j-1], 右侧为 d[j .. j+window-1]
            left_start = max(0, j - peak_sign_window)
            left = d[left_start:j] if j > left_start else np.array([], dtype=float)
            right_end = min(len(d), j + peak_sign_window)
            right = d[j:right_end] if right_end > j else np.array([], dtype=float)
            left_mean = float(np.mean(left)) if left.size > 0 else 0.0
            right_mean = float(np.mean(right)) if right.size > 0 else 0.0
            if left_mean > 0.0 and right_mean < 0.0 and smoothed[j] > 0.0:
                local_peaks.append(j)

        if local_peaks:
            # 一个连通区间可能有多个候选峰，取平滑值最大的那个
            best_idx = max(local_peaks, key=lambda idx: smoothed[idx])
        else:
            # 平顶峰/弱噪声场景下兜底为区间最大值
            seg_idx = int(np.argmax(smoothed[start:end + 1]))
            best_idx = start + seg_idx

        peaks.append((float(best_idx), float(projection[best_idx])))

    # 兜底：若分段结果为空，则直接在全局平滑曲线上找导数由正到负且高于阈值的点
    if not peaks:
        # 全局兜底：在整个平滑曲线上应用窗口化的符号变化检测
        for j in range(1, n - 1):
            left_start = max(0, j - peak_sign_window)
            left = d[left_start:j] if j > left_start else np.array([], dtype=float)
            right_end = min(len(d), j + peak_sign_window)
            right = d[j:right_end] if right_end > j else np.array([], dtype=float)
            left_mean = float(np.mean(left)) if left.size > 0 else 0.0
            right_mean = float(np.mean(right)) if right.size > 0 else 0.0
            if left_mean > 0.0 and right_mean < 0.0 and smoothed[j] >= float(binary_thresh_factor):
                peaks.append((float(j), float(projection[j])))

    # 添加峰合并逻辑：合并距离小于 MIN_GAP 的相邻峰，取较高者的位置
    min_gap = int(CFG.get('MIN_GAP', 8))
    if len(peaks) > 1:
        merged_peaks = []
        i = 0
        while i < len(peaks):
            current_peak = peaks[i]
            j = i + 1
            
            # 检查是否有相邻峰需要合并
            while j < len(peaks):
                next_peak = peaks[j]
                if next_peak[0] - current_peak[0] <= min_gap:
                    # 比较两个峰的高度，保留较高的
                    if next_peak[1] > current_peak[1]:
                        current_peak = next_peak  # 保留较高的峰
                    j += 1
                else:
                    break
            
            merged_peaks.append(current_peak)
            i = j
        
        peaks = merged_peaks

    return peaks


def smooth_projection(projection, gaussian_k=None, gaussian_sigma=None):
    """返回一维投影的高斯平滑结果（浮点数组）。"""
    if projection is None or projection.size == 0:
        return projection
    if gaussian_k is None:
        gaussian_k = CFG.get("GAUSSIAN_KERNEL", 11)
    if gaussian_sigma is None:
        gaussian_sigma = CFG.get("GAUSSIAN_SIGMA", 2.0)
    k = int(gaussian_k)
    if k % 2 == 0:
        k += 1
    half = k // 2
    xs = np.arange(-half, half + 1)
    gauss = np.exp(-(xs.astype(float) ** 2) / (2.0 * float(gaussian_sigma) ** 2))
    gauss = gauss / np.sum(gauss)
    return np.convolve(projection.astype(float), gauss, mode='same')





# ---------------- KNN 风格的投影匹配（文件内实现） ----------------
def find_peaks_from_smoothed(arr):
    """在平滑的一维数组中查找局部极大值的索引（基于导数符号变化）。"""
    peaks_idx = []
    if arr is None or len(arr) < 3:
        return peaks_idx
    d = np.diff(arr)
    peak_sign_window = int(CFG.get('PEAK_SIGN_WINDOW', 1))
    if peak_sign_window < 1:
        peak_sign_window = 1
    n = len(arr)
    for j in range(1, n - 1):
        # 左侧导数窗口：d[max(0, j-window) .. j-1]
        left_start = max(0, j - peak_sign_window)
        left = d[left_start:j] if j > left_start else np.array([], dtype=float)
        # 右侧导数窗口：d[j .. j+window-1]
        right_end = min(len(d), j + peak_sign_window)
        right = d[j:right_end] if right_end > j else np.array([], dtype=float)
        left_mean = float(np.mean(left)) if left.size > 0 else 0.0
        right_mean = float(np.mean(right)) if right.size > 0 else 0.0
        if left_mean > 0 and right_mean < 0 and arr[j] > 0:
            peaks_idx.append(j)
    return peaks_idx


def run_knn_recognition(char_images=None):
    """运行基于投影的表驱动识别（原先位于 KNN.py 中）。

    功能：扫描 `INPUT_CHARS_DIR` 目录中的字符图像（png/bmp），对每个字符计算横向/纵向投影、平滑、峰值检测，
    并与 `Clist.txt` 中的模板按比例进行匹配。
    识别结果写入 `OUTPUT_TXT`，并在控制台打印简要摘要。
    """
    # Delegate actual recognition logic to ocr.py to decouple implementations.
    try:
        import ocr as _ocr
    except Exception:
        # fallback: use importlib to load ocr.py
        import importlib.util
        import sys
        import os
        # Get the directory of this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        ocr_path = os.path.join(current_dir, 'ocr.py')
        if os.path.exists(ocr_path):
            spec = importlib.util.spec_from_file_location("ocr", ocr_path)
            _ocr = importlib.util.module_from_spec(spec)
            sys.modules['ocr'] = _ocr
            spec.loader.exec_module(_ocr)
        else:
            raise ImportError(f"Cannot find ocr.py at {ocr_path}")
    return _ocr.run_knn_recognition(char_images=char_images)



def classify_peaks_to_chars(peaks, img_width):
    """
    根据 peaks 的横坐标和高度，使用状态机将 peaks 分组为字符。
    保持原有字符分割逻辑，不因空格而改变原有的字符分组。
    
    状态机规则：
    1. 如果相邻峰间距大于CHAR_SEP_GAP强制进行分割
    2. 每隔5个峰就进行一次分割
    3. 每扫描过一个波峰就要判断一次状态机
    
    返回 list of dict:
    {
      'type': 'char',
      'x_start': int,
      'x_end': int,
      'peaks': [(x, h), ...]  # peaks belonging to this segment
    }
    """
    if not peaks:
        return []
    
    # 字符分割阈值
    CHAR_SEP_GAP = float(CFG.get("CHAR_SEP_GAP", 37))  # 相邻峰间距大于此值则强制分割
    
    # Sort peaks by x
    peaks = sorted(peaks, key=lambda p: p[0])
    n = len(peaks)
    xs = [float(p[0]) for p in peaks]
    
    segments = []
    
    # 状态机状态
    i = 0  # 当前峰的索引
    while i < n:
        # 开始收集一个新的字符
        collected_indices = [i]
        collected_peaks = [peaks[i]]
        
        # 状态机：收集属于当前字符的波峰
        j = i
        while j + 1 < n:
            # 检查相邻峰间距
            gap = xs[j + 1] - xs[j]
            
            # 条件1：如果间距大于CHAR_SEP_GAP，强制分割
            if gap > CHAR_SEP_GAP:
                break
            
            # 条件2：如果已收集5个峰，强制分割
            if len(collected_indices) >= 5:
                break
            
            # 继续收集
            j += 1
            collected_indices.append(j)
            collected_peaks.append(peaks[j])
        
        # 确定字符边界
        # 左边界：如果是第一个字符，从0开始；否则取前一个字符的右边界
        if i == 0:
            x_start = 0
        else:
            # 取前一个波峰和当前第一个波峰的中点
            x_start = int((xs[i - 1] + xs[i]) / 2)
        
        # 右边界：如果是最后一个字符，到图像末尾；否则取当前最后一个波峰和下一个波峰的中点
        if j == n - 1:
            x_end = img_width
        else:
            x_end = int((xs[j] + xs[j + 1]) / 2)
        
        # 确保边界有效
        if x_end <= x_start:
            x_end = min(img_width, x_start + 10)
        
        # 创建字符段
        segments.append({
            'type': 'char',
            'x_start': x_start,
            'x_end': x_end,
            'peaks': collected_peaks
        })
        
        # 更新状态，跳到下一个未处理的峰
        i = j + 1
    
    return segments


def create_space_array():
    """创建一个特殊的numpy数组来表示空格"""
    # 创建一个黑色的1x1像素图像表示空格（全为0）
    # 使用与字符图像相似的格式，但所有像素都是黑色（0）
    # 这样OCR模块可以识别这个特殊数组并返回空格
    space_img = np.zeros((20, 20), dtype=np.uint8)  # 20x20的全黑图像表示空格
    return space_img


def cut_characters(peaks, src_image_path, out_dir, write_files=False):
    """
    根据 peaks 信息对 src_image_path 进行字符切割，
    保存切割结果到 out_dir。
    
    参数：
        peaks: list of (x, height) from detect_and_cluster_peaks()
        src_image_path: 要切割的图像路径（如 output/3_enhanced.png）
        out_dir: 输出目录（如 output/characters/）
    """
    # src_image_path 可以是文件路径或 numpy ndarray
    img = None
    if isinstance(src_image_path, np.ndarray):
        img = src_image_path.copy()
        # 如果是单通道图像，为了绘图一致性转换为 BGR
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img = cv2.imread(src_image_path)
    if img is None:
        print(f"[cut_characters] Cannot load image: {src_image_path}")
        return ([], [])
    
    img_height, img_width = img.shape[:2]
    
    # 获取原始的峰信息用于分析
    peaks_sorted = sorted(peaks, key=lambda p: p[0]) if peaks else []
    
    # 根据原有逻辑获取字符分段，但是考虑大间隙作为空格
    SPACE_THRESHOLD = float(CFG.get("SPACE_THRESHOLD", 150))
    CHAR_SEP_GAP = float(CFG.get("CHAR_SEP_GAP", 46))
    
    segments = []
    
    if not peaks_sorted:
        segments = []
    else:
        i = 0  # 当前峰的索引
        while i < len(peaks_sorted):
            # 开始收集一个新的字符
            collected_peaks = [peaks_sorted[i]]
            
            # 状态机：收集属于当前字符的波峰
            j = i
            while j + 1 < len(peaks_sorted):
                # 检查相邻峰间距
                gap = peaks_sorted[j + 1][0] - peaks_sorted[j][0]
                
                # 如果间隙大于空格阈值，需要在此处分割并插入空格
                if gap > SPACE_THRESHOLD:
                    # 先完成当前段
                    # 确定字符边界
                    if i == 0:
                        x_start = 0
                    else:
                        x_start = int((peaks_sorted[i - 1][0] + peaks_sorted[i][0]) / 2)
                    
                    if j == len(peaks_sorted) - 1:
                        x_end = img_width
                    else:
                        x_end = int((peaks_sorted[j][0] + peaks_sorted[j + 1][0]) / 2)
                    
                    if x_end <= x_start:
                        x_end = min(img_width, x_start + 10)
                    
                    segments.append({
                        'type': 'char',
                        'x_start': x_start,
                        'x_end': x_end,
                        'peaks': collected_peaks
                    })
                    
                    # 插入空格段
                    space_x_start = int(peaks_sorted[j][0])
                    space_x_end = int(peaks_sorted[j + 1][0])
                    segments.append({
                        'type': 'space',
                        'x_start': space_x_start,
                        'x_end': space_x_end,
                        'peaks': []
                    })
                    
                    # 移动到下一个段
                    i = j + 1
                    collected_peaks = [peaks_sorted[i]]  # 开始新的收集
                    j = i  # 重新开始内循环
                    continue
                
                # 如果间距大于字符分割阈值，强制分割（但不作为空格）
                elif gap > CHAR_SEP_GAP:
                    break
                
                # 如果已收集5个峰，强制分割
                if len(collected_peaks) >= 5:
                    break
                
                # 继续收集
                j += 1
                collected_peaks.append(peaks_sorted[j])
            
            # 确定最终字符段边界
            if i == 0:
                x_start = 0
            else:
                x_start = int((peaks_sorted[i - 1][0] + peaks_sorted[i][0]) / 2)
            
            if j == len(peaks_sorted) - 1:
                x_end = img_width
            else:
                x_end = int((peaks_sorted[j][0] + peaks_sorted[j + 1][0]) / 2)
            
            if x_end <= x_start:
                x_end = min(img_width, x_start + 10)
            
            segments.append({
                'type': 'char',
                'x_start': x_start,
                'x_end': x_end,
                'peaks': collected_peaks
            })
            
            # 更新状态，跳到下一个未处理的峰
            i = j + 1
    
    # 如果启用调试模式，生成投影图
    if CFG.get('DEBUG_DRAW_SEGMENTS', False):
        try:
            # 计算当前图像的垂直投影
            gray_for_proj = img
            if gray_for_proj.ndim == 3:
                gray_for_proj = cv2.cvtColor(gray_for_proj, cv2.COLOR_BGR2GRAY)
            
            # 重新计算垂直投影用于绘图
            white_mask = (gray_for_proj == 255).astype(np.uint8)
            proj_vals = np.sum(white_mask, axis=0).astype(float) / float(max(1, gray_for_proj.shape[0]))
            
            # 生成调试图像
            debug_img_path = os.path.join(out_dir if out_dir else '.', 'debug_vertical_projection.png')
            plot_vertical_projection_with_peaks(
                src_image_path, 
                proj_vals, 
                peaks, 
                segments, 
                debug_img_path
            )
        except Exception as e:
            print(f"生成调试投影图失败: {e}")
    
    # 仅在需要写入文件时准备输出目录
    if write_files:
        os.makedirs(out_dir, exist_ok=True)
        # Clear previous results (remove old PNG/BMP files)
        for f in os.listdir(out_dir):
            if f.lower().endswith(('.png', '.bmp')):
                try:
                    os.remove(os.path.join(out_dir, f))
                except Exception:
                    pass
    
    char_count = 0
    summary = []
    # collect in-memory character images when not writing to disk
    char_images = []  # list of dicts: {'name': str, 'img': ndarray, 'meta': dict}
    
    for seg_idx, seg in enumerate(segments):
        x_start = max(0, int(seg['x_start']))
        x_end = min(img_width, int(seg['x_end']))
        
        if x_start >= x_end:
            continue
        
        char_type = seg['type']
        
        if char_type == 'space':
            # 创建空格图像
            char_img = create_space_array()
            out_name = f"{char_count:03d}_space.bmp"
        else:
            char_img = img[:, x_start:x_end]
            out_name = f"{char_count:03d}_{char_type}.bmp"
        
        out_full = os.path.join(out_dir, out_name) if out_dir else out_name

        # 可选：写入 BMP（不压缩）；否则收集到内存中返回
        if write_files:
            try:
                cv2.imwrite(out_full, char_img)
            except Exception:
                cv2.imwrite(out_full, char_img)
        else:
            # store a copy (ensure contiguous array)
            char_images.append({'name': out_name, 'img': char_img.copy(), 'meta': {
                'index': char_count,
                'type': char_type,
                'x_start': x_start,
                'x_end': x_end,
                'width': x_end - x_start,
                'num_peaks': len(seg['peaks'])
            }})

        # 如果启用分段调试绘图，则为每个字符生成投影可视化（横/纵）并保存到
        # CFG['PROJ_OUT_DIR']，每字符一张图
        if CFG.get('DEBUG_DRAW_SEGMENTS', False) and char_type != 'space':
            try:
                # Ensure projection output dir exists
                proj_out = CFG.get('PROJ_OUT_DIR', os.path.join(out_dir, 'projections'))
                if not proj_out:
                    proj_out = os.path.join(out_dir, 'projections') if out_dir else 'output/characters/projections'
                os.makedirs(proj_out, exist_ok=True)

                # 将字符转换为灰度并生成二值掩码
                gray = char_img
                if gray.ndim == 3 and gray.shape[2] == 3:
                    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
                elif gray.ndim == 3 and gray.shape[2] == 4:
                    gray = cv2.cvtColor(gray, cv2.COLOR_BGRA2GRAY)
                # 二值化以获得前景掩码（白色像素）
                try:
                    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                except Exception:
                    bw = (gray > 128).astype('uint8') * 255

                h_c, w_c = bw.shape[:2]
                if h_c <= 0 or w_c <= 0:
                    raise ValueError('Empty character image')

                mask = (bw == 255).astype(np.uint8)
                proj_x = np.sum(mask, axis=0).astype(float) / max(1, h_c)  # along columns
                proj_y = np.sum(mask, axis=1).astype(float) / max(1, w_c)  # along rows

                # 对投影进行平滑
                # 使用每字符平滑参数（缺省则退回全局参数）
                k = int(CFG.get('CHAR_PROJ_GAUSSIAN_KERNEL', CFG.get('GAUSSIAN_KERNEL', 11)))
                sigma = float(CFG.get('CHAR_PROJ_GAUSSIAN_SIGMA', CFG.get('GAUSSIAN_SIGMA', 2.0)))
                sm_x = smooth_projection(proj_x, gaussian_k=k, gaussian_sigma=sigma)
                sm_y = smooth_projection(proj_y, gaussian_k=k, gaussian_sigma=sigma)

                # Apply per-character zero threshold if configured
                char_zero = CFG.get('CHAR_PROJ_SMOOTH_ZERO_THRESHOLD', None)
                if char_zero is not None:
                    try:
                        zt = float(char_zero)
                        sm_x[sm_x < zt] = 0.0
                        sm_y[sm_y < zt] = 0.0
                    except Exception:
                        pass

                # 在平滑投影上查找峰值
                px = find_peaks_from_smoothed(sm_x)
                py = find_peaks_from_smoothed(sm_y)

                # 使用 OpenCV 创建小尺寸投影可视化图像
                thumb_h = 120
                thumb_w = int(max(40, min(160, (w_c * thumb_h) // max(1, h_c))))
                # 字符缩略图
                try:
                    thumb = cv2.resize(char_img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
                except Exception:
                    thumb = np.full((thumb_h, thumb_w, 3), 255, dtype=np.uint8)

                # 横向投影图（高度 thumb_h，宽度 plot_w）
                plot_w = max(160, w_c)
                plot_h = thumb_h
                plot_x = np.full((plot_h, plot_w, 3), 255, dtype=np.uint8)
                if len(sm_x) > 1:
                    # normalize sm_x to [0, plot_h-20]
                    sx = np.interp(np.linspace(0, len(sm_x)-1, plot_w), np.arange(len(sm_x)), sm_x)
                    max_sx = float(np.max(sx)) if np.max(sx) > 0 else 1.0
                    pts = []
                    for ix in range(plot_w):
                        val = sx[ix] / max_sx
                        y = plot_h - 10 - int(val * (plot_h - 20))
                        pts.append((ix, y))
                    for i in range(1, len(pts)):
                        cv2.line(plot_x, pts[i-1], pts[i], (0, 0, 0), 1)
                    # 标记峰值（映射索引位置）
                    for p in px:
                        if len(sm_x) > 1:
                            px_pos = int((p / max(1, len(sm_x)-1)) * (plot_w-1))
                            cv2.circle(plot_x, (px_pos, pts[max(0, min(len(pts)-1, px_pos))][1]), 3, (0, 0, 255), -1)

                # 纵向投影图（高度 thumb_h，宽度 120）
                plot2_w = 120
                plot2_h = thumb_h
                plot_y_img = np.full((plot2_h, plot2_w, 3), 255, dtype=np.uint8)
                if len(sm_y) > 1:
                    sy = np.interp(np.linspace(0, len(sm_y)-1, plot2_h), np.arange(len(sm_y)), sm_y)
                    max_sy = float(np.max(sy)) if np.max(sy) > 0 else 1.0
                    pts2 = []
                    for iy in range(plot2_h):
                        val = sy[iy] / max_sy
                        x = 10 + int(val * (plot2_w - 20))
                        pts2.append((x, iy))
                    for i in range(1, len(pts2)):
                        cv2.line(plot_y_img, pts2[i-1], pts2[i], (0, 0, 0), 1)
                    # 标记峰值
                    for p in py:
                        py_pos = int((p / max(1, len(sm_y)-1)) * (plot2_h-1))
                        cv2.circle(plot_y_img, (pts2[py_pos][0], py_pos), 3, (0, 0, 255), -1)

                # 将最终图像水平合成：[缩略图 | 横向投影图 | 纵向投影图]
                spacer = np.full((thumb_h, 8, 3), 255, dtype=np.uint8)
                try:
                    composed = np.hstack([thumb, spacer, plot_x, spacer, plot_y_img])
                except Exception:
                    # 回退：将缩略图与投影图垂直排列
                    composed = np.vstack([thumb, plot_x, plot_y_img])

                proj_name = os.path.splitext(out_name)[0] + '_proj.png'
                proj_path = os.path.join(proj_out, proj_name)
                cv2.imwrite(proj_path, composed)
            except Exception as _e:
                print(f"Failed to generate projection debug for {out_name}: {_e}")
        summary.append({
            'index': char_count,
            'type': char_type,
            'x_start': x_start,
            'x_end': x_end,
            'width': x_end - x_start,
            'num_peaks': len(seg['peaks'])
        })

        print(f"  [{char_count:03d}] type={char_type:12s}, x=[{x_start}..{x_end}], w={x_end-x_start}, peaks={len(seg['peaks'])}")
        char_count += 1
    
    # 可选：从 numpy 数组绘制分段调试图像并保存
    if CFG.get('DEBUG_DRAW_SEGMENTS', False):
        try:
            # 确保输出目录存在
            dbg_out_dir = out_dir if out_dir else os.path.dirname(CFG.get('OUTPUT_TXT', '')) or '.'
            os.makedirs(dbg_out_dir, exist_ok=True)

            dbg_img = img.copy()
            # 确保为 BGR 颜色空间以便绘图
            if dbg_img.ndim == 2:
                dbg_img = cv2.cvtColor(dbg_img, cv2.COLOR_GRAY2BGR)

            # 为每个分段绘制矩形和标签
            for si, s in enumerate(segments):
                xs = max(0, int(s['x_start']))
                xe = min(img_width, int(s['x_end']))
                # 矩形：覆盖整高的切片区域
                if s['type'] == 'space':
                    cv2.rectangle(dbg_img, (xs, 0), (xe, img_height - 1), (255, 0, 0), 2)  # 蓝色表示空格
                else:
                    cv2.rectangle(dbg_img, (xs, 0), (xe, img_height - 1), (0, 255, 0), 2)  # 绿色表示字符

                label = f"{si:02d}:{s['type']}"
                # 在分段左上方写标签
                tx = xs + 4
                ty = 20
                cv2.putText(dbg_img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            dbg_path = os.path.join(dbg_out_dir, 'segmentation_debug.png')
            cv2.imwrite(dbg_path, dbg_img)
            print(f"分段调试图像已保存: {dbg_path}")
        except Exception as _e:
            print(f"写入分段调试图像失败: {_e}")
    
    # 生成垂直投影图，仅在debug模式下启用
    if CFG.get('DEBUG_DRAW_SEGMENTS', False):
        try:
            import matplotlib
            matplotlib.use('Agg')  # 使用非GUI后端
            import matplotlib.pyplot as plt
            # 计算当前图像的垂直投影
            gray_for_proj = img
            if gray_for_proj.ndim == 3:
                gray_for_proj = cv2.cvtColor(gray_for_proj, cv2.COLOR_BGR2GRAY)
            # 计算垂直投影
            white_mask = (gray_for_proj == 255).astype(np.uint8)
            proj_vals = np.sum(white_mask, axis=0).astype(float) / float(max(1, gray_for_proj.shape[0]))
            
            # 创建投影图
            plt.figure(figsize=(12, 4))
            plt.plot(proj_vals, 'b-', linewidth=1, label='Vertical Projection')
            # 标记波峰
            for peak_x, peak_val in peaks:
                plt.plot(int(peak_x), peak_val, 'ro', markersize=4)
            # 标记字符分段边界
            for si, s in enumerate(segments):
                plt.axvline(x=s['x_start'], color='g', linestyle='--', alpha=0.7)
                plt.axvline(x=s['x_end'], color='g', linestyle='--', alpha=0.7)
            plt.title('Character Segmentation - Vertical Projection with Peaks and Segments')
            plt.xlabel('X Coordinate')
            plt.ylabel('Projection Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 保存投影图
            proj_plot_path = os.path.join(out_dir if out_dir else '.', 'vertical_projection_plot.png')
            plt.savefig(proj_plot_path, dpi=150, bbox_inches='tight')
            print(f"垂直投影图已保存: {proj_plot_path}")
            plt.close()
        except ImportError:
            print("matplotlib未安装，跳过垂直投影图生成")
        except Exception as _e:
            print(f"生成垂直投影图失败: {_e}")

    if write_files:
        print(f"\ncut_characters: {char_count} segments saved to '{out_dir}'")
    else:
        print(f"\ncut_characters: {char_count} segments returned in-memory (no files written)")
    return summary, char_images


def main():
    # 默认同时处理两个文件并将结果合并覆盖保存到同一文件
    srcs = ['output/morphology_result.png', 'output/3_enhanced.png']
    dst = 'output/vertical_projection.png'

    # 支持命令行参数：列出要处理的图像路径，最后一个参数为输出路径
    args = sys.argv[1:]
    if len(args) == 1:
        # 只给出输出路径，则使用默认源
        dst = args[0]
    elif len(args) >= 2:
        # 多个参数：前面为输入图像，最后一个为输出
        srcs = args[:-1]
        dst = args[-1]

    # 为了生产环境性能，已移除合并绘图；此处仅执行字符分段并在下方（如果生成）
    # 对 `output/characters` 做轻量级概述。

    # 注意：KNN 调用已移到字符切割之后，以确保 output/characters 已生成并包含样本。
    
    # ---- NEW: 字符切割 ----
    morphology_path = 'output/morphology_result.png'
    enhanced_path = 'output/3_enhanced.png'
    cut_dir = 'output/characters'

    if os.path.exists(morphology_path) and os.path.exists(enhanced_path):
        # 将图像读入内存（numpy 数组）并将数组传递给函数
        morph_img = cv2.imread(morphology_path, cv2.IMREAD_GRAYSCALE)
        enhanced_img = cv2.imread(enhanced_path, cv2.IMREAD_GRAYSCALE)

        if morph_img is None or enhanced_img is None:
            print(f"[main] Skipping character cut: failed to read required images.")
        else:
            _, proj_morph = vertical_projection(morph_img, None)
            peaks_for_cut = detect_and_cluster_peaks(
                proj_morph,
                binary_thresh_factor=CFG["BINARY_THRESH_FACTOR"],
                merge_distance=CFG["MERGE_DISTANCE"],
                min_width=CFG["MIN_WIDTH"],
                gaussian_k=CFG.get("GAUSSIAN_KERNEL", 11),
                gaussian_sigma=CFG.get("GAUSSIAN_SIGMA", 2.0),
            )

            print(f"\nSegmenting characters using {len(peaks_for_cut)} peaks...")
            # 将 numpy 数组（enhanced_img）传入 cut_characters 以在内存中处理
            summary, char_images = cut_characters(peaks_for_cut, enhanced_img, cut_dir, write_files=False)

            # （为生产性能移除了概览绘图）
    else:
        print(f"[main] Skipping character cut: required images not found.")

    # ---- Run KNN-based recognition (optional) ----
    # Use in-memory character images if available; fall back to directory scanning otherwise
    try:
        if 'char_images' in locals() and char_images:
            print("Running in-file KNN recognition on in-memory images...")
            try:
                run_knn_recognition(char_images=char_images)
            except Exception as _knn_err:
                print(f"run_knn_recognition() failed: {_knn_err}")
        else:
            # 回退到经典行为：扫描磁盘目录 output/characters 来查找字符图像
            if os.path.exists(cut_dir) and any(f.lower().endswith(('.png', '.bmp')) for f in os.listdir(cut_dir)):
                print("Running in-file KNN recognition (from disk)...")
                try:
                    run_knn_recognition()
                except Exception as _knn_err:
                    print(f"run_knn_recognition() failed: {_knn_err}")
            else:
                print("KNN skipped: no in-memory images and 'output/characters' not found or empty.")
    except Exception as _e:
        print(f"KNN execution check failed: {_e}")


if __name__ == '__main__':
    main()
