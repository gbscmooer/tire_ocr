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
    "MERGE_DISTANCE": 5,
    # 过滤噪声的最小宽度（像素列），小于该宽度的连通区间会被丢弃
    "MIN_WIDTH": 10,
    # 高斯平滑核大小（奇数），建议 5,7,11
    "GAUSSIAN_KERNEL": 5,
    # 高斯平滑的标准差
    "GAUSSIAN_SIGMA": 1.0,
    # 高斯平滑后小值直接置零的绝对阈值（投影值范围0..1），若设置为 None 则使用 BINARY_THRESH_FACTOR * max
    "SMOOTH_ZERO_THRESHOLD": 0.09,
    # 非线性增强参数：gamma<1 可抬升低幅波峰、压缩高峰动态范围
    "NONLINEAR_GAMMA": 0.15,
    # 局部对比度窗口（像素列，建议奇数）
    "LOCAL_CONTRAST_WINDOW": 41,
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
                ax_proj.scatter(xs, ys, color='red', zorder=3, s=20, label='Peaks')
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


if __name__ == '__main__':
    main()
