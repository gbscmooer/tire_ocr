import cv2
import os
import numpy as np

# --------- 全局配置 (CFG) ---------
CFG = {
    "TEMPLATE_DIR": 'patch',
    "INPUT_CHARS_DIR": 'output/characters',
    "OUTPUT_REPORT": 'output/recognition_report.png',
    "OUTPUT_TXT": 'output/characters/recognition_result.txt',
    
    "STD_SIZE": (64, 64),           
    
    # --- 连通域过滤 ---
    "MIN_AREA": 10,                 
    
    # --- 形态学参数 (用于连点成线，供HOG提取特征) ---
    "DILATE_KERNEL_SIZE": (3, 3),
    "DILATE_ITER": 2,               
    "ERODE_ITER": 1,

    # --- HOG 算子参数 ---
    "HOG_WIN_SIZE": (64, 64),
    "HOG_BLOCK_SIZE": (16, 16),
    "HOG_BLOCK_STRIDE": (8, 8),
    "HOG_CELL_SIZE": (8, 8),
    "HOG_NBINS": 9,

    # --- 报告排版 ---
    "REPORT_COLS": 6,               # 只有两张图对比，一行可以放更多个结果（建议6-8个）
    "DISP_BLOCK_SIZE": (90, 120),   # 单个小图的显示尺寸
    "FONT_SCALE": 0.5,
}
# ----------------------------------

# 初始化 HOG 算子
hog = cv2.HOGDescriptor(
    CFG["HOG_WIN_SIZE"],
    CFG["HOG_BLOCK_SIZE"],
    CFG["HOG_BLOCK_STRIDE"],
    CFG["HOG_CELL_SIZE"],
    CFG["HOG_NBINS"]
)

def preprocess_flow(img, size):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    clean_th = np.zeros_like(th)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= CFG["MIN_AREA"]:
            clean_th[labels == i] = 255
            
    coords = cv2.findNonZero(clean_th)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        clean_th = clean_th[y:y+h, x:x+w]
    else:
        return np.zeros(size, dtype=np.uint8)
    
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, CFG["DILATE_KERNEL_SIZE"])
    morphed = cv2.dilate(clean_th, k, iterations=CFG["DILATE_ITER"])
    morphed = cv2.erode(morphed, k, iterations=CFG["ERODE_ITER"])
    
    return cv2.resize(morphed, size)

def get_hog_features(img):
    return hog.compute(img).flatten()

def load_all_templates(base_dir):
    templates = {}
    print(f"正在从 {base_dir} 构建 HOG 模板库...")
    if not os.path.exists(base_dir): return None
    
    for label in sorted(os.listdir(base_dir)):
        label_path = os.path.join(base_dir, label)
        if not os.path.isdir(label_path): continue
        templates[label] = []
        for img_name in os.listdir(label_path):
            img = cv2.imread(os.path.join(label_path, img_name))
            if img is not None:
                proc = preprocess_flow(img, CFG["STD_SIZE"])
                feat = get_hog_features(proc)
                templates[label].append((feat, proc))
    return templates

def recognize_hog(proc_char, templates):
    curr_feat = get_hog_features(proc_char)
    best_label, max_sim, best_t_img = "?", -1.0, None
    
    for label, items in templates.items():
        for t_feat, t_img in items:
            sim = np.dot(curr_feat, t_feat) / (np.linalg.norm(curr_feat) * np.linalg.norm(t_feat) + 1e-9)
            if sim > max_sim:
                max_sim, best_label, best_t_img = sim, label, t_img
    return best_label, max_sim, best_t_img

def create_visual_block(orig_img, match_img, label, score):
    """
    两路对比展示块：[原点阵图 | 匹配到的模板图]
    """
    dw, dh = CFG["DISP_BLOCK_SIZE"]
    
    def format_img(im):
        if im is None: im = np.zeros(CFG["STD_SIZE"], dtype=np.uint8)
        if len(im.shape) == 2: im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        return cv2.resize(im, (dw, dh))
    
    img_left = format_img(orig_img)
    img_right = format_img(match_img)
    
    # 横向拼接两张图
    combined = np.hstack([img_left, img_right])
    
    # 底部信息文字区
    footer_h = 40
    footer = np.zeros((footer_h, combined.shape[1], 3), dtype=np.uint8) + 245
    color = (0, 150, 0) if score > 0.75 else (0, 0, 200)
    
    info_text = f"{label} (S:{score:.2f})"
    cv2.putText(footer, info_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 
                CFG["FONT_SCALE"], color, 1, cv2.LINE_AA)
    
    return np.vstack([combined, footer])

def main():
    templates = load_all_templates(CFG["TEMPLATE_DIR"])
    if not templates: return
    
    char_files = sorted([f for f in os.listdir(CFG["INPUT_CHARS_DIR"]) if f.endswith('.png')])
    if not char_files: return

    visual_blocks, results_str = [], ""
    print(f"正在识别并生成精简版报告...")

    for filename in char_files:
        path = os.path.join(CFG["INPUT_CHARS_DIR"], filename)
        orig = cv2.imread(path)
        if orig is None: continue
        
        proc = preprocess_flow(orig, CFG["STD_SIZE"])
        label, score, t_img = recognize_hog(proc, templates)
        
        results_str += label
        # 此时只传入原图和匹配到的模板图
        visual_blocks.append(create_visual_block(orig, t_img, label, score))

    # 拼接报告
    cols = CFG["REPORT_COLS"]
    rows = (len(visual_blocks) + cols - 1) // cols
    report_rows = []
    
    for r in range(rows):
        row_data = visual_blocks[r*cols : (r+1)*cols]
        while len(row_data) < cols:
            row_data.append(np.zeros_like(visual_blocks[0]) + 255)
        report_rows.append(np.hstack(row_data))
    
    if report_rows:
        cv2.imwrite(CFG["OUTPUT_REPORT"], np.vstack(report_rows))
    
    with open(CFG["OUTPUT_TXT"], 'w', encoding='utf-8') as f:
        f.write(results_str)
    
    print("-" * 30)
    print(f"识别结果: {results_str}")
    print(f"简洁报告已保存: {CFG['OUTPUT_REPORT']}")

if __name__ == "__main__":
    main()