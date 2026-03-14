import cv2
import numpy as np
from post_vertical_projection import find_black_col_segments, classify_peaks_to_chars

def test_segmentation_with_black_cols():
    """测试包含连续黑色列的分割功能"""
    
    # 创建一个测试图像，包含连续黑色列
    test_img = np.ones((50, 100), dtype=np.uint8) * 255  # 白色背景
    
    # 添加一些"字符"（垂直条）
    test_img[:, 10] = 0  # 峰1
    test_img[:, 15] = 0  # 峰2
    test_img[:, 25] = 0  # 峰3 (与前一个峰距离10)
    
    # 在第30-38列创建连续黑色列（9个像素的连续黑色列，超过阈值8）
    test_img[:, 30:39] = 0
    
    # 添加更多"字符"
    test_img[:, 45] = 0  # 峰4
    test_img[:, 50] = 0  # 峰5 (与前一个峰距离5)
    
    # 在第55-65列创建连续黑色列（11个像素的连续黑色列，超过阈值8）
    test_img[:, 55:66] = 0
    
    # 添加最后的字符
    test_img[:, 70] = 0  # 峰6
    test_img[:, 75] = 0  # 峰7 (与前一个峰距离5)
    
    print("测试图像尺寸:", test_img.shape)
    
    # 模拟检测到的峰
    peaks = [(10, 0.8), (15, 0.7), (25, 0.9), (45, 0.6), (50, 0.8), (70, 0.7), (75, 0.9)]
    
    print("检测到的峰:", peaks)
    
    # 使用原始方法（不考虑图像数据）
    segments_original = classify_peaks_to_chars(peaks, test_img.shape[1])
    print(f"\n原始分割结果 ({len(segments_original)} 段):")
    for i, seg in enumerate(segments_original):
        print(f"  段 {i}: [{seg['x_start']}, {seg['x_end']}] 峰数: {len(seg['peaks'])}")
    
    # 使用新方法（考虑图像数据和连续黑色列）
    segments_with_black_cols = classify_peaks_to_chars(peaks, test_img.shape[1], img=test_img)
    print(f"\n使用连续黑色列检测的分割结果 ({len(segments_with_black_cols)} 段):")
    for i, seg in enumerate(segments_with_black_cols):
        print(f"  段 {i}: [{seg['x_start']}, {seg['x_end']}] 峰数: {len(seg['peaks'])}")
    
    # 检测连续黑色列
    black_segments = find_black_col_segments(test_img)
    print(f"\n检测到的连续黑色列段: {black_segments}")
    
    print("\n验证: 新的分割方法应该在连续黑色列的位置进行额外分割")
    print("- 原始方法: 不考虑黑色列，按峰间距分割")
    print("- 新方法: 在检测到的连续黑色列处进行额外分割")

if __name__ == "__main__":
    test_segmentation_with_black_cols()