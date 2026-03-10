#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tire_ocr import TireOCR

def test_stage3(image_path="data/20250115083551210.png"):
    """测试第三阶段处理"""
    ocr = TireOCR()
    
    print("=== 测试第一阶段 ===")
    enhanced = ocr.stage1_extract_roi(image_path)
    print(f"第一阶段完成: 增强图像尺寸 {enhanced.shape}")
    
    print("\n=== 测试第二阶段 ===")
    binary_chars = ocr.stage2_morphological_reconstruction()
    print(f"第二阶段完成: 二值化字符图像尺寸 {binary_chars.shape}")
    
    print("\n=== 测试第三阶段 ===")
    try:
        char_images = ocr.stage3_character_segmentation()
        print(f"第三阶段完成: 分割出 {len(char_images)} 个字符")
        
        # 显示结果
        ocr.stage3_visualize()
        
        print(f"\n第三阶段中间结果已保存到 {ocr.output_dir}/ 目录:")
        print("  - char_00.png, char_01.png, ...: 分割后的单个字符图像")
        print("  - char_positions.txt: 字符位置信息")
        print("  - stage3_result.png: 可视化对比图")
        
        # 显示字符统计信息
        if char_images:
            print(f"\n字符统计:")
            print(f"  总字符数: {len(char_images)}")
            for i, char_img in enumerate(char_images):
                print(f"  字符 {i}: 尺寸 {char_img.shape}, 前景像素 {(char_img > 0).sum()}")
            
            # 检查字符是否合理
            print(f"\n字符预览:")
            for i, char_img in enumerate(char_images[:5]):  # 只显示前5个
                foreground_ratio = (char_img > 0).sum() / char_img.size
                print(f"  字符 {i}: 前景比例 {foreground_ratio:.2%}")
        
    except Exception as e:
        print(f"第三阶段处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "data/20250115083551210.png"
    
    success = test_stage3(image_path)
    sys.exit(0 if success else 1)