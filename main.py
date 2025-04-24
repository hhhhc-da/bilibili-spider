from utils.platforms.bilibili.bilibili_oid import get_oid_from_bilibili_url
from utils.platforms.bilibili.bilibili_spider import spider_from_oid
from utils.cookie import get_cookie
import pandas as pd
import os
import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default=os.path.join("xlsx", 'input'), help='输入路径')
    parser.add_argument('--output_path', type=str, default=os.path.join("xlsx", 'output'), help='输出路径')
    parser.add_argument('--input_excel_name', type=str, default='bilibili_input.xlsx', help='输入 Excel 名称')
    parser.add_argument('--output_excel_name', type=str, default="bilibili_output.xlsx", help='输出 Excel 名称')
    parser.add_argument('--cookie', type=str, default=os.path.join('cookie', 'cookie.txt'), help='Cookie 文件路径')
    parser.add_argument('--encode', type=str, default='utf-8', help='Cookie 使用的编码格式')
    return parser.parse_args()

def bilibili_spider(input_path=os.path.join("xlsx", 'input'), output_path=os.path.join("xlsx", 'output'), input_excel_name='bilibili_input.xlsx', output_excel_name="bilibili_output.xlsx", cookie=os.path.join('cookie', 'cookie.txt'), encode='utf-8'):
    # 输入输出文件
    input_handler = os.path.join(input_path, input_excel_name)
    output_handler = os.path.join(output_path, output_excel_name)
    
    if os.path.exists(output_handler):
        os.remove(output_handler)
        
    # 从 Excel 中读取数据
    pf = pd.read_excel(input_handler)
    
    # 读取 Cookie
    cookie = get_cookie(path=cookie, encoding=encode)
    
    for url in pf['url']:
        print("开始解析网址:", url)
        bvid, oid = get_oid_from_bilibili_url(url)
        spider_from_oid(bvid=bvid, oid=oid, cookie=cookie, base_dir=output_path, file_name=output_excel_name)
    
# 程序入口
if __name__ == "__main__":
    # 开始爬取 Bilibili 内容
    opt = parse_opt()
    bilibili_spider(**vars(opt))