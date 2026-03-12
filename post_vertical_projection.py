import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


# --------- 配置（可调超参数） ---------
# 所有重要的绘图 / 峰值检测 参数在此处定义。
CFG = {
    "FIGSIZE_PER_COL": 6,
    "FIG_HEIGHT": 6,
    # 子图行高度比例（上行原图，下行投影）
    "HEIGHT_RATIOS": [2, 1],
    # 保存时的输出 DPI
    "PLOT_DPI": 100,
    # 二值掩码阈值因子：阈值 = BINARY_THRESH_FACTOR * max(projection)
    "BINARY_THRESH_FACTOR": 0.01,
    # 合并相近区间的距离（像素列），若两个区间间距小于该值则合并
    "MERGE_DISTANCE": 3,
    # 过滤噪声的最小宽度（像素列），小于该宽度的连通区间会被丢弃
    "MIN_WIDTH": 5,
    # 高斯平滑核大小（奇数），建议 5,7,11
    "GAUSSIAN_KERNEL": 5,
    # 高斯平滑的标准差
    "GAUSSIAN_SIGMA": 1.0,
    # 高斯平滑后小值直接置零的绝对阈值（投影值范围0..1），若设置为 None 则使用 BINARY_THRESH_FACTOR * max
    "SMOOTH_ZERO_THRESHOLD": 0.09,
    # 非线性增强参数：gamma<1 可抬升低幅波峰、压缩高峰动态范围
    "NONLINEAR_GAMMA": 0.15,
    # 局部对比度窗口（像素列，建议奇数）
    "LOCAL_CONTRAST_WINDOW": 31,
    # 字符切割相关参数
    "CHAR_SEP_GAP": 50,      # 峰间距超过此值视为字符边界（px）—用于普通字符检测
    "SPECIAL_GAP": 80,       # 用于"1"/"-"检测的峰间距阈值（px）
    "SPACE_GAP": 400,        # 峰间距超过此值插入空格段（px）
    "RATIO_271_TOL": 0.12,   # "1" 的高度比容差（每个分量绝对偏差）
}

# --------------------------------------------------------------


def vertical_projection(image_path, out_path):
    """读取图像，按列计算白色像素比例并绘图

    参数:
        image_path: 输入图像路径（灰度或二值图）
        out_path: 输出绘图保存路径
    """
    # 以灰度读取图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # 采用 Otsu 自动阈值将图像二值化，白色像素为255
    _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 计算每列白色像素数量并转换为比例（0..1）
    height = bw.shape[0]
    white_mask = (bw == 255).astype(np.uint8)
    projection_counts = np.sum(white_mask, axis=0)
    projection = projection_counts.astype(float) / float(height)

    # 返回灰度图和每列白色比例
    return img, projection


def detect_and_cluster_peaks(projection, binary_thresh_factor=None, merge_distance=None, min_width=None,
                             gaussian_k=None, gaussian_sigma=None):
    """基于导数过零点的波峰统计（高斯平滑 + 阈值化 + 导数符号切换）。

    算法要点：
    - 对投影做 1D 高斯平滑以去噪。
    - 使用全局低阈值（binary_thresh_factor * max）把较小值设为 0（区分背景）。
    - 先做非线性重缩放（gamma）压缩大峰、抬升小峰，再做局部对比度增强。
    - 在增强信号上计算一阶导数，在导数从正变负的位置判定局部峰值（zero-crossing of derivative）。
    - 仅保留阈值化后非零区域内的峰；并按 min_width 过滤过窄区域噪声。

    参数（都可从 CFG 读取默认值）：
    - binary_thresh_factor: 二值阈值因子，阈值 = factor * max(projection)
    - merge_distance: （保留用于兼容，当前实现未显式使用）
    - min_width: 最小保留区间宽度
    - gaussian_k: 高斯核大小（奇数）
    - gaussian_sigma: 高斯核标准差
    返回：[(x_peak, y_value), ...]
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
        w = 3
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
    smoothed = enhanced
    d = np.diff(smoothed)
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
            if d[j - 1] > 0.0 and d[j] < 0.0 and smoothed[j] > 0.0:
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
        for j in range(1, n - 1):
            if d[j - 1] > 0.0 and d[j] < 0.0 and smoothed[j] >= float(binary_thresh_factor):
                peaks.append((float(j), float(projection[j])))

    return peaks


def smooth_projection(projection, gaussian_k=None, gaussian_sigma=None):
    """Return 1D Gaussian-smoothed projection (float array)."""
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


def merge_and_plot(image_paths, out_path):
    """对多个图像计算垂直白色像素比例并合并绘制

    参数:
        image_paths: 图像路径列表
        out_path: 合并图像保存路径
    """
    results = []
    for p in image_paths:
        img, proj = vertical_projection(p, None)
        results.append((p, img, proj))

    n = len(results)
    # 创建 2 行 n 列的子图：上行为原图，下行为投影
    fig, axes = plt.subplots(
        2,
        n,
        figsize=(CFG["FIGSIZE_PER_COL"] * n, CFG["FIG_HEIGHT"]),
        gridspec_kw={"height_ratios": CFG["HEIGHT_RATIOS"]},
    )
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    def _title_for_path(p):
        name = os.path.basename(p).lower()
        if 'morphology' in name:
            return 'Morphology Result'
        return 'Original Result'

    # 如果正好有两张图，分别绘制（不使用通用循环），以避免标题错配
    if n == 2:
        # 左图（index 0）
        p0, img0, proj0 = results[0]
        ax_img = axes[0, 0]
        ax_proj = axes[1, 0]
        ax_img.imshow(img0, cmap='gray')
        ax_img.set_title(_title_for_path(p0), fontsize=14)
        ax_img.axis('off')
        cols0 = np.arange(len(proj0))
        # 对 morphology 图使用平滑后的波形以便观察处理效果，否则显示原始比例柱状图
        if 'morphology' in os.path.basename(p0).lower():
            sm0 = smooth_projection(proj0, gaussian_k=CFG.get("GAUSSIAN_KERNEL", 11), gaussian_sigma=CFG.get("GAUSSIAN_SIGMA", 2.0))
            ax_proj.plot(cols0, sm0, color='black', label='Smoothed Proportion')
        else:
            ax_proj.bar(cols0, proj0, color='black', label='White Proportion')
        ax_proj.set_title('Vertical Projection', fontsize=12)
        ax_proj.set_xlabel('Column')
        ax_proj.set_ylabel('Proportion (white)')
        ax_proj.set_ylim(0, 1)
        ax_proj.legend()
        if 'morphology' in os.path.basename(p0).lower():
            peaks0 = detect_and_cluster_peaks(
                proj0,
                binary_thresh_factor=CFG["BINARY_THRESH_FACTOR"],
                merge_distance=CFG["MERGE_DISTANCE"],
                min_width=CFG["MIN_WIDTH"],
                gaussian_k=CFG.get("GAUSSIAN_KERNEL", 11),
                gaussian_sigma=CFG.get("GAUSSIAN_SIGMA", 2.0),
            )
            if peaks0:
                xs = [p[0] for p in peaks0]
                ys = [p[1] for p in peaks0]
                ax_proj.scatter(xs, ys, color='red', zorder=3, s=5, label='Peaks')
                printable = [(round(float(x), 2), round(float(y), 4)) for x, y in peaks0]
                print(f"Peaks for {p0}: {printable}")

        # 右图（index 1）
        p1, img1, proj1 = results[1]
        ax_img = axes[0, 1]
        ax_proj = axes[1, 1]
        ax_img.imshow(img1, cmap='gray')
        ax_img.set_title(_title_for_path(p1), fontsize=14)
        ax_img.axis('off')
        cols1 = np.arange(len(proj1))
        if 'morphology' in os.path.basename(p1).lower():
            sm1 = smooth_projection(proj1, gaussian_k=CFG.get("GAUSSIAN_KERNEL", 11), gaussian_sigma=CFG.get("GAUSSIAN_SIGMA", 2.0))
            ax_proj.plot(cols1, sm1, color='black', label='Smoothed Proportion')
        else:
            ax_proj.bar(cols1, proj1, color='black', label='White Proportion')
        ax_proj.set_title('Vertical Projection', fontsize=12)
        ax_proj.set_xlabel('Column')
        ax_proj.set_ylabel('Proportion (white)')
        ax_proj.set_ylim(0, 1)
        ax_proj.legend()
        if 'morphology' in os.path.basename(p1).lower():
            peaks1 = detect_and_cluster_peaks(
                proj1,
                binary_thresh_factor=CFG["BINARY_THRESH_FACTOR"],
                merge_distance=CFG["MERGE_DISTANCE"],
                min_width=CFG["MIN_WIDTH"],
                gaussian_k=CFG.get("GAUSSIAN_KERNEL", 11),
                gaussian_sigma=CFG.get("GAUSSIAN_SIGMA", 2.0),
            )
            if peaks1:
                xs = [p[0] for p in peaks1]
                ys = [p[1] for p in peaks1]
                ax_proj.scatter(xs, ys, color='red', zorder=3, s=20, label='Peaks')
                printable = [(round(float(x), 2), round(float(y), 4)) for x, y in peaks1]
                print(f"Peaks for {p1}: {printable}")
    else:
        # 其他情况下按文件名决定标题（保留循环以支持任意数量）
        for i, (p, img, proj) in enumerate(results):
            ax_img = axes[0, i]
            ax_proj = axes[1, i]
            ax_img.imshow(img, cmap='gray')
            ax_img.set_title(_title_for_path(p), fontsize=14)
            ax_img.axis('off')
            cols = np.arange(len(proj))
            if 'morphology' in os.path.basename(p).lower():
                sm = smooth_projection(proj, gaussian_k=CFG.get("GAUSSIAN_KERNEL", 11), gaussian_sigma=CFG.get("GAUSSIAN_SIGMA", 2.0))
                ax_proj.plot(cols, sm, color='black', label='Smoothed Proportion')
            else:
                ax_proj.bar(cols, proj, color='black', label='White Proportion')
            ax_proj.set_title('Vertical Projection', fontsize=12)
            ax_proj.set_xlabel('Column')
            ax_proj.set_ylabel('Proportion (white)')
            ax_proj.set_ylim(0, 1)
            ax_proj.legend()
            if 'morphology' in os.path.basename(p).lower():
                peaks = detect_and_cluster_peaks(
                    proj,
                    binary_thresh_factor=CFG["BINARY_THRESH_FACTOR"],
                    merge_distance=CFG["MERGE_DISTANCE"],
                    min_width=CFG["MIN_WIDTH"],
                    gaussian_k=CFG.get("GAUSSIAN_KERNEL", 11),
                    gaussian_sigma=CFG.get("GAUSSIAN_SIGMA", 2.0),
                )
                if peaks:
                    xs = [pp[0] for pp in peaks]
                    ys = [pp[1] for pp in peaks]
                    ax_proj.scatter(xs, ys, color='red', zorder=3, s=20, label='Peaks')
                    printable = [(round(float(x), 2), round(float(y), 4)) for x, y in peaks]
                    print(f"Peaks for {p}: {printable}")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=CFG["PLOT_DPI"])
    plt.close(fig)


def classify_peaks_to_chars(peaks, img_width):
    """
    根据 peaks 的横坐标和高度，使用状态机将 peaks 分组为字符。
    
    字符规则（根据用户要求）：
    - "1"：gap ≥ CHAR_SEP_GAP（50px）前无波峰，3个波峰高度比 2:7:1，gap ≥ CHAR_SEP_GAP 后无波峰
    - "-"：gap ≥ CHAR_SEP_GAP（50px）前无波峰，3个波峰高度比 1:1:1，gap ≥ CHAR_SEP_GAP 后无波峰
    - 正常字符：5个波峰（采用滑动窗口扫描，连续5峰且峰间距<80px）
    - 空格：gap ≥ SPACE_GAP（150px）
    
    返回 list of dict:
    {
      'type': '1' | '-' | 'char' | 'space' | 'unknown',
      'x_start': int,
      'x_end': int,
      'peaks': [(x, h), ...]  # peaks belonging to this segment (empty for space)
    }
    """
    if not peaks:
        return []
    
    CHAR_SEP_GAP = float(CFG.get("CHAR_SEP_GAP", 50))
    NORMAL_CHAR_GAP = 80.0  # 普通字符检测阈值：峰间距<80px
    SPACE_GAP = float(CFG.get("SPACE_GAP", 150))
    RATIO_271_TOL = float(CFG.get("RATIO_271_TOL", 0.12))
    
    # Sort peaks by x
    peaks = sorted(peaks, key=lambda p: p[0])
    n = len(peaks)
    xs = [float(p[0]) for p in peaks]
    hs = [float(p[1]) for p in peaks]
    
    def is_ratio_271(h0, h1, h2):
        total = h0 + h1 + h2
        if total <= 0:
            return False
        r0, r1, r2 = h0/total, h1/total, h2/total
        # 2:7:1 -> 2/10=0.2, 7/10=0.7, 1/10=0.1
        return (abs(r0 - 0.2) <= RATIO_271_TOL and
                abs(r1 - 0.7) <= RATIO_271_TOL and
                abs(r2 - 0.1) <= RATIO_271_TOL)
    
    def is_ratio_111(h0, h1, h2):
        if min(h0, h1, h2) <= 0:
            return False
        # 1:1:1 means max/min ratio close to 1.0
        max_h = max(h0, h1, h2)
        min_h = min(h0, h1, h2)
        # allow up to 1.5x difference (empirical)
        # ADDITIONAL: each peak must be < 0.1 (height threshold)
        if max_h >= 0.1:
            return False
        return max_h / min_h <= 1.5
    
    segments = []
    i = 0
    prev_x_end = 0  # right boundary of previous segment
    
    while i < n:
        # Gap from previous segment end to current peak
        gap_to_here = xs[i] - prev_x_end
        
        # 1. Space detection (highest priority)
        if gap_to_here >= SPACE_GAP:
            space_start = int(prev_x_end)
            space_end = int(xs[i])
            if space_end > space_start:
                segments.append({
                    'type': 'space',
                    'x_start': space_start,
                    'x_end': space_end,
                    'peaks': []
                })
            # Move prev_x_end to the end of this space
            prev_x_end = xs[i]
            gap_to_here = 0.0
        
        # 2. Try to detect "1" or "-" (保持原来的算法：只检查比例，不检查gap)
        # 原代码对于3峰组只检查比例，不检查前后gap
        # 我们在这里尝试检测连续的3个峰，如果比例符合就识别
        if i + 2 < n:
            # 检查内部间距是否小于80px（普通字符阈值），如果太大则不是同一字符
            inner_gap1 = xs[i+1] - xs[i]
            inner_gap2 = xs[i+2] - xs[i+1]
            if inner_gap1 < NORMAL_CHAR_GAP and inner_gap2 < NORMAL_CHAR_GAP:
                h0, h1, h2 = hs[i], hs[i+1], hs[i+2]
                if is_ratio_271(h0, h1, h2):
                    # "1" character found
                    # Determine boundaries: start from midpoint between previous and current, end at midpoint between current+2 and next
                    if i == 0:
                        x_start = 0
                    else:
                        x_start = int((xs[i-1] + xs[i]) / 2)
                    x_end = int((xs[i+2] + (xs[i+3] if i+3 < n else img_width)) / 2)
                    if x_end > x_start:
                        segments.append({
                            'type': '1',
                            'x_start': x_start,
                            'x_end': x_end,
                            'peaks': peaks[i:i+3]
                        })
                    prev_x_end = x_end
                    i += 3
                    continue
                elif is_ratio_111(h0, h1, h2):
                    # "-" character found
                    if i == 0:
                        x_start = 0
                    else:
                        x_start = int((xs[i-1] + xs[i]) / 2)
                    x_end = int((xs[i+2] + (xs[i+3] if i+3 < n else img_width)) / 2)
                    if x_end > x_start:
                        segments.append({
                            'type': '-',
                            'x_start': x_start,
                            'x_end': x_end,
                            'peaks': peaks[i:i+3]
                        })
                    prev_x_end = x_end
                    i += 3
                    continue
        
        # 3. 滑动窗口检测普通字符（5个波峰，每个波峰之间间隔<80px）
        # 按照用户描述：左边界L停在第一个波峰，右边界R继续收集波峰
        # 收集连续5个波峰，每个相邻波峰间距<80px
        collected = [i]  # start with current peak
        j = i
        while j + 1 < n and len(collected) < 5:
            if xs[j+1] - xs[j] < NORMAL_CHAR_GAP:
                collected.append(j+1)
                j += 1
            else:
                break
        
        count = len(collected)
        
        # Determine boundaries for this segment
        # 左边界：如果是第一个字符，从0开始；否则取前一个字符的右边界
        if i == 0:
            x_start = 0
        else:
            # 取前一个波峰和当前第一个波峰的中点
            x_start = int((xs[i-1] + xs[i]) / 2)
        
        # 右边界：如果是最后一个字符，到图像末尾；否则取当前最后一个波峰和下一个波峰的中点
        if j == n - 1:
            x_end = img_width
        else:
            x_end = int((xs[j] + xs[j+1]) / 2)
        
        # Ensure x_start is not before prev_x_end (use prev_x_end as floor)
        x_start = max(int(prev_x_end), x_start)
        if x_end <= x_start:
            x_end = min(img_width, int(prev_x_end) + 10)  # fallback
        
        # Determine type based on count
        if count == 5:
            char_type = 'char'
        elif count == 3:
            # 可能是 "1" 或 "-"，但之前检查已经排除了（gap条件不满足）
            # 这里再检查一次比例，但标记为 unknown3
            h0, h1, h2 = hs[i], hs[i+1], hs[i+2] if i+2 < n else (0, 0, 0)
            if is_ratio_271(h0, h1, h2) or is_ratio_111(h0, h1, h2):
                # 即使gap条件不满足，也尝试标记为特殊字符
                if is_ratio_271(h0, h1, h2):
                    char_type = '1'
                else:
                    char_type = '-'
            else:
                char_type = 'unknown3'
        else:
            char_type = f'unknown{count}'
        
        # Create segment
        segment_peaks = [peaks[idx] for idx in collected]
        segments.append({
            'type': char_type,
            'x_start': x_start,
            'x_end': x_end,
            'peaks': segment_peaks
        })
        
        # Update state
        prev_x_end = x_end
        i += count
    
    return segments


def cut_characters(peaks, src_image_path, out_dir):
    """
    根据 peaks 信息对 src_image_path 进行字符切割，
    保存切割结果到 out_dir。
    
    参数：
        peaks: list of (x, height) from detect_and_cluster_peaks()
        src_image_path: 要切割的图像路径（如 output/3_enhanced.png）
        out_dir: 输出目录（如 output/characters/）
    """
    img = cv2.imread(src_image_path)
    if img is None:
        print(f"[cut_characters] Cannot load image: {src_image_path}")
        return
    
    img_height, img_width = img.shape[:2]
    
    segments = classify_peaks_to_chars(peaks, img_width)
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Clear previous results (PNG files only)
    for f in os.listdir(out_dir):
        if f.lower().endswith('.png'):
            os.remove(os.path.join(out_dir, f))
    
    char_count = 0
    summary = []
    
    for seg_idx, seg in enumerate(segments):
        x_start = max(0, int(seg['x_start']))
        x_end = min(img_width, int(seg['x_end']))
        
        if x_start >= x_end:
            continue
        
        char_type = seg['type']
        char_img = img[:, x_start:x_end]
        
        label = char_type.replace('unknown', 'unk')
        out_name = f"{char_count:03d}_{label}.png"
        out_full = os.path.join(out_dir, out_name)
        cv2.imwrite(out_full, char_img)
        
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
    
    # Also create a debug visualization showing the cuts on the original image
    if char_count > 0:
        debug_img = img.copy()
        for seg in segments:
            x1 = max(0, int(seg['x_start']))
            x2 = min(img_width, int(seg['x_end']))
            color = (255, 0, 0) if seg['type'] == 'space' else (0, 255, 0)
            cv2.rectangle(debug_img, (x1, 0), (x2, img_height-1), color, 1)
        debug_path = "output/character_cuts_debug.png"
        cv2.imwrite(debug_path, debug_img)
        print(f"\nDebug visualization saved to: {debug_path}")
    
    print(f"\ncut_characters: {char_count} segments saved to '{out_dir}'")
    return summary


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

    merge_and_plot(srcs, dst)
    print(f"Saved merged vertical projection image to: {dst}")

    # NOTE: KNN invocation moved to after character cutting to ensure
    # output/characters exists and is populated. See bottom of this main().
    
    # ---- NEW: 字符切割 ----
    morphology_path = 'output/morphology_result.png'
    enhanced_path = 'output/3_enhanced.png'
    cut_dir = 'output/characters'
    
    if os.path.exists(morphology_path) and os.path.exists(enhanced_path):
        _, proj_morph = vertical_projection(morphology_path, None)
        peaks_for_cut = detect_and_cluster_peaks(
            proj_morph,
            binary_thresh_factor=CFG["BINARY_THRESH_FACTOR"],
            merge_distance=CFG["MERGE_DISTANCE"],
            min_width=CFG["MIN_WIDTH"],
            gaussian_k=CFG.get("GAUSSIAN_KERNEL", 11),
            gaussian_sigma=CFG.get("GAUSSIAN_SIGMA", 2.0),
        )
        print(f"\nSegmenting characters using {len(peaks_for_cut)} peaks...")
        cut_characters(peaks_for_cut, enhanced_path, cut_dir)
    else:
        print(f"[main] Skipping character cut: required images not found.")

    # ---- Run KNN-based recognition (optional) ----
    # Only run KNN after characters have been cut and the folder exists and is non-empty.
    try:
        if os.path.exists(cut_dir) and any(f.lower().endswith('.png') for f in os.listdir(cut_dir)):
            try:
                import KNN
                if hasattr(KNN, 'main'):
                    print("Running KNN recognition (KNN.py)...")
                    try:
                        KNN.main()
                    except Exception as _knn_err:
                        print(f"KNN.main() failed: {_knn_err}")
            except Exception as _imp_err:
                print(f"Failed to import/run KNN.py: {_imp_err}")
        else:
            print("KNN skipped: 'output/characters' not found or empty.")
    except Exception as _e:
        print(f"KNN execution check failed: {_e}")


if __name__ == '__main__':
    main()
