#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tire_ocr import TireOCR

def test_stage2(image_path="data/20250115083551210.png"):
    """测试第二阶段处理"""
    ocr = TireOCR()
    
    print("=== 测试第一阶段 ===")
    enhanced = ocr.stage1_extract_roi(image_path)
    print(f"第一阶段完成: 增强图像尺寸 {enhanced.shape}")
    
    print("\n=== 测试第二阶段 ===")
    try:
        binary_chars = ocr.stage2_morphological_reconstruction()
        print(f"第二阶段完成: 二值化字符图像尺寸 {binary_chars.shape}")
        
        # 显示结果
        ocr.stage2_visualize()
        
        print(f"\n第二阶段中间结果已保存到 {ocr.output_dir}/ 目录:")
        print("  - 4_tophat.png: Top-Hat 变换结果")
        print("  - 5_morph.png: 形态学闭运算结果")
        print("  - 6_binary_chars.png: 自适应阈值分割结果")
        print("  - stage2_result.png: 可视化对比图")
        
        # 检查像素统计
        print(f"\n二值化图像统计:")
        print(f"  总像素数: {binary_chars.size}")
        print(f"  前景像素 (白色): {(binary_chars > 0).sum()}")
        print(f"  背景像素 (黑色): {(binary_chars == 0).sum()}")
        print(f"  前景比例: {(binary_chars > 0).sum() / binary_chars.size:.2%}")
        
    except Exception as e:
        print(f"第二阶段处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "data/20250115083551210.png"
    
    success = test_stage2(image_path)
    sys.exit(0 if success else 1)