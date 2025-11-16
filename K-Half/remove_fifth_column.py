#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
脚本：移除train.txt文件的第五列
用法: python remove_fifth_column.py [文件路径]
"""

import sys
import os


def remove_fifth_column(input_file, output_file=None):
    """
    移除文件中的第五列，保留前四列
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径，如果为None则覆盖原文件
    """
    if not os.path.exists(input_file):
        print(f"错误：文件 {input_file} 不存在")
        return False
    
    # 如果未指定输出文件，使用临时文件然后替换
    if output_file is None:
        output_file = input_file + '.tmp'
        overwrite = True
    else:
        overwrite = False
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f_in:
            with open(output_file, 'w', encoding='utf-8') as f_out:
                line_count = 0
                for line in f_in:
                    line = line.strip()
                    if not line:  # 处理空行
                        f_out.write('\n')
                        continue
                    
                    # 按制表符分割列
                    columns = line.split('\t')
                    
                    # 如果列数少于5，保留原行（可能已有问题或格式不同）
                    if len(columns) < 5:
                        f_out.write(line + '\n')
                        continue
                    
                    # 只保留前四列
                    new_line = '\t'.join(columns[:4])
                    f_out.write(new_line + '\n')
                    line_count += 1
                
                print(f"成功处理 {line_count} 行")
        
        # 如果覆盖原文件，替换文件
        if overwrite:
            if os.path.exists(input_file):
                os.remove(input_file)
            os.rename(output_file, input_file)
            print(f"文件已更新：{input_file}")
        else:
            print(f"输出文件：{output_file}")
        
        return True
    
    except Exception as e:
        print(f"处理文件时出错：{e}")
        if overwrite and os.path.exists(output_file):
            os.remove(output_file)
        return False


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python remove_fifth_column.py <文件路径> [输出文件路径]")
        print("示例: python remove_fifth_column.py data/ICEWS14/train.txt")
        print("      python remove_fifth_column.py data/ICEWS14/train.txt data/ICEWS14/train_new.txt")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    # 确认操作
    if output_file is None:
        response = input(f"这将覆盖原文件 {input_file}，确认继续？(y/n): ")
        if response.lower() != 'y':
            print("操作已取消")
            return
    
    remove_fifth_column(input_file, output_file)


if __name__ == '__main__':
    main()

