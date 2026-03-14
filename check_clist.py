#!/usr/bin/env python3
"""
检查 Clist.txt 文件的格式是否合法，并检测重复的字符定义（包括约分后）
格式: 字符:((x方向峰值序列),(y方向峰值序列))
例如: A:((3:1:1:1:3),(3:2:2:5:2:2:2))
"""

import re
import math
from fractions import Fraction
from typing import List, Tuple, Dict


def gcd_list(numbers: List[int]) -> int:
    """计算列表中所有数字的最大公约数"""
    if not numbers:
        return 1
    result = numbers[0]
    for num in numbers[1:]:
        result = math.gcd(result, num)
    return result


def normalize_sequence(seq: List[int]) -> Tuple[int, ...]:
    """规范化序列：去除首尾零值，然后约分"""
    if not seq:
        return tuple()
    
    # 去除首尾的零
    start = 0
    while start < len(seq) and seq[start] == 0:
        start += 1
    end = len(seq) - 1
    while end >= start and seq[end] == 0:
        end -= 1
    
    if start > end:
        return tuple()
    
    trimmed_seq = seq[start:end+1]
    
    # 计算最大公约数并约分
    gcd = gcd_list(trimmed_seq)
    if gcd > 1:
        normalized = tuple(x // gcd for x in trimmed_seq)
    else:
        normalized = tuple(trimmed_seq)
    
    return normalized


def parse_clist_line(line: str) -> Tuple[str, List[int], List[int], bool]:
    """
    解析Clist.txt中的一行
    返回: (字符, x序列, y序列, 是否成功解析)
    """
    line = line.strip()
    if not line:
        return "", [], [], True  # 空行视为有效
    
    # 匹配格式: 字符:((数字序列),(数字序列)) - 支持字母、数字和连字符
    match = re.match(r'^([A-Z0-9\-]):\(\(([^)]+)\),\(([^)]+)\)\)$', line)
    if not match:
        return "", [], [], False
    
    char = match.group(1)
    x_seq_str = match.group(2)
    y_seq_str = match.group(3)
    
    try:
        x_seq = [int(x) for x in x_seq_str.split(':') if x]
        y_seq = [int(y) for y in y_seq_str.split(':') if y]
        
        # 验证序列不为空
        if not x_seq or not y_seq:
            return "", [], [], False
            
        return char, x_seq, y_seq, True
    except ValueError:
        return "", [], [], False


def check_clist_file(file_path: str) -> bool:
    """检查Clist.txt文件并报告问题"""
    print(f"正在检查文件: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 不存在")
        return False
    
    # 存储已见的字符和对应的规范化序列
    char_to_normal = {}  # 字符 -> (规范化x序列, 规范化y序列)
    seen_normal = {}     # (规范化x序列, 规范化y序列) -> 字符名列表
    
    all_valid = True
    
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:  # 跳过空行
            continue
            
        char, x_seq, y_seq, success = parse_clist_line(line)
        
        if not success:
            print(f"第 {line_num} 行格式错误: {line}")
            all_valid = False
            continue
        
        print(f"第 {line_num} 行解析成功: {char}:(({':'.join(map(str, x_seq))}),({':'.join(map(str, y_seq))}))")
        
        # 规范化序列
        norm_x = normalize_sequence(x_seq)
        norm_y = normalize_sequence(y_seq)
        
        if char in char_to_normal:
            print(f"第 {line_num} 行: 字符 '{char}' 重复定义")
            all_valid = False
        
        # 检查规范化后的序列是否已存在
        norm_key = (norm_x, norm_y)
        if norm_key in seen_normal:
            existing_chars = seen_normal[norm_key]
            print(f"第 {line_num} 行: 字符 '{char}' 的序列约分后与字符 {existing_chars} 重复")
            print(f"  规范化序列: X={norm_x}, Y={norm_y}")
            all_valid = False
        else:
            seen_normal[norm_key] = []
        
        char_to_normal[char] = (norm_x, norm_y)
        seen_normal[norm_key].append(char)
    
    # 检查是否有重复字符定义
    char_counts = {}
    for line in lines:
        line = line.strip()
        if line:
            match = re.match(r'^([A-Z0-9\-]):', line)
            if match:
                char = match.group(1)
                if char in char_counts:
                    char_counts[char].append(line)
                else:
                    char_counts[char] = [line]
    
    for char, lines_with_char in char_counts.items():
        if len(lines_with_char) > 1:
            print(f"字符 '{char}' 在多行中重复定义:")
            for line in lines_with_char:
                print(f"  - {line}")
    
    if all_valid:
        print("\n✓ 所有行格式正确")
        print(f"✓ 检查了 {len([l for l in lines if l.strip()])} 个字符定义")
        print("✓ 没有发现重复定义（包括约分后）")
        return True
    else:
        print("\n✗ 发现错误，请修正后重新检查")
        return False


def test_with_sample():
    """使用示例数据测试函数"""
    print("测试示例数据...")
    sample_lines = [
        "A:((3:1:1:1:3),(3:2:2:5:2:2:2))",
        "B:((7:3:3:3:4),(4:2:2:4:2:2:4))", 
        "C:((5:2:2:2:2),(3:2:1:1:1:2:3))"
    ]
    
    print("解析示例数据:")
    for i, line in enumerate(sample_lines):
        char, x_seq, y_seq, success = parse_clist_line(line)
        if success:
            norm_x = normalize_sequence(x_seq)
            norm_y = normalize_sequence(y_seq)
            print(f"  {line} -> {char}: X{norm_x}, Y{norm_y}")
        else:
            print(f"  {line} -> 解析失败")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "Clist.txt"
    
    print("Clist.txt 格式检查工具")
    print("=" * 40)
    
    # 先测试示例
    test_with_sample()
    print("=" * 40)
    
    # 检查文件
    success = check_clist_file(file_path)
    
    if success:
        print("\n✓ 检查通过!")
    else:
        print("\n✗ 检查失败!")
        sys.exit(1)