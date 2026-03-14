import cv2
import numpy as np
from post_vertical_projection import find_black_col_segments, CFG

def test_black_col_detection():
    """测试连续黑色列检测功能"""
    
    # 创建一个测试图像，包含连续黑色列
    test_img = np.ones((50, 100), dtype=np.uint8) * 255  # 白色背景
    
    # 在第20-25列创建连续黑色列（6个像素的连续黑色列）
    test_img[:, 20:26] = 0  # 6个连续黑色列
    
    # 在第40-48列创建连续黑色列（9个像素的连续黑色列，超过默认阈值8）
    test_img[:, 40:49] = 0  # 9个连续黑色列
    
    # 在第60-62列创建连续黑色列（3个像素，少于阈值8）
    test_img[:, 60:63] = 0  # 3个连续黑色列
    
    print("测试图像尺寸:", test_img.shape)
    
    # 使用默认阈值（8）检测连续黑色列
    black_segments_default = find_black_col_segments(test_img)
    print(f"使用默认阈值({CFG['BLACK_COL_THRESHOLD']})检测到的黑色列段: {black_segments_default}")
    
    # 使用阈值6检测连续黑色列
    black_segments_6 = find_black_col_segments(test_img, black_col_threshold=6)
    print(f"使用阈值(6)检测到的黑色列段: {black_segments_6}")
    
    # 使用阈值10检测连续黑色列
    black_segments_10 = find_black_col_segments(test_img, black_col_threshold=10)
    print(f"使用阈值(10)检测到的黑色列段: {black_segments_10}")
    
    # 验证结果
    print("\n验证:")
    print("- 阈值8时，应检测到[40:49]段（9个连续列）")
    print("- 阈值6时，应检测到[20:25]和[40:48]段（6个和9个连续列）")
    print("- 阈值10时，应检测到[]（没有超过10个连续列的段）")
    
    # 保存测试图像
    cv2.imwrite("test_black_col_image.png", test_img)
    print("\n测试图像已保存为 test_black_col_image.png")

if __name__ == "__main__":
    test_black_col_detection()