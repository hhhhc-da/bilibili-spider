#coding=utf-8
import torch
import os
from torch import nn
from torch.utils.data import DataLoader
from dataset.dataset import DeployDataset
from models.model import BertClassifierModule
from tqdm import tqdm
import pandas as pd
import argparse
import numpy as np

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_len', type=int, default=50, help='句子的最大长度')
    parser.add_argument('--num_labels', type=int, default=6, help='BERT 的参数，表示最后输出特征数')
    parser.add_argument('--pretraind_path', type=str, default='bert-chinese', help='预训练参数路径')
    parser.add_argument('--batch_size', type=int, default=50, help='每个 batch 的大小')
    parser.add_argument('--save_path', type=str, default=os.path.join('models', 'bert_model.pth'), help='预训练参数路径')
    parser.add_argument('--csv_path', type=str, default=os.path.join('..', 'MediaCrawler', 'data', 'douyin'), help='计算出来的文件路径')
    parser.add_argument('--save_xlsx', type=str, default=os.path.join('runs', 'bert-report.xlsx'), help='识别报告输出路径')
    parser.add_argument('--save_id', type=str, default=os.path.join('runs', 'bert-user-score.xlsx'), help='所有值得的用户 user_id 输出路径')
    return parser.parse_args()

# 训练 Bert 模型
def run_bert_model(
    max_len = 50,
    num_labels=6,
    batch_size=50,
    pretraind_path='bert-chinese',
    save_path=os.path.join('models', 'bert.pth'),
    csv_path=os.path.join('..', 'MediaCrawler', 'data', 'douyin'),
    save_xlsx=os.path.join('runs', 'bert-report.txt'),
    save_id=os.path.join('runs', 'bert-user-score.xlsx')
    ):

    # 首先确认我们的模型运行在 GPU 上
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"模型将运行在 {device} 上\n")

    # 创建模型
    bert_model = BertClassifierModule(num_labels=num_labels, device=device, pretraind_path=pretraind_path)
    model = bert_model.get_model()
    model.load_state_dict(torch.load(save_path, map_location=device))
    tokenizer = bert_model.get_tokenizer()

    label_dict = {
        "冗余信息": 0,
        "推销广告": 1,
        "学习人员": 2,
        "从业人员": 3,
        "实际用户": 4,
        "需求用户": 5
    }
    # 反转字典
    type_dict = {}
    for k, v in label_dict.items():
        type_dict[v] = k
            
    # 读取数据
    filenames = os.listdir(csv_path)
    lst = {}
    for filename in filenames:
        infomation = filename.split('_')
        if infomation[1] == 'search' and infomation[2] == 'comments':
            lst[infomation[0]] = os.path.join(csv_path, filename)
            
    # 要处理最后一个文件
    filename = lst[max(list(lst.keys()))]
    print("\n处理文件名:", filename, '\n')
    pf = pd.read_csv(filename)
    
    # 开始将数据转换为测试输出集
    deploy_dataset = DeployDataset(pf_data=pf, tokenizer=tokenizer, max_len=max_len)
    deploy_loader = DataLoader(deploy_dataset, batch_size=batch_size, shuffle=False)

    last_layer = nn.Softmax(dim=1)
    model.eval()
    # 存储真实标签和预测标签
    all_content = []
    all_predictions = []
    all_user_id = []
    all_nickname = []
    
    with torch.no_grad():
        with tqdm(deploy_loader, desc='Test') as tbar:
            for batch in tbar:
                all_content.extend(batch['input_text'])
                all_user_id.extend(batch['user_id'].cpu().numpy())
                all_nickname.extend(batch['nickname'])
                
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)

                outputs = model(input_ids, attention_mask, token_type_ids)
                out = last_layer(outputs.logits)

                all_predictions.extend(torch.argmax(out, dim=1).cpu().numpy())
    
    # 输出测试案例
    pf = pd.DataFrame({
        'pred': [type_dict[i] for i in all_predictions],
        'user_id': all_user_id,
        'nickname': all_nickname,
        'text': all_content
    })
    
    print(pf)
    with pd.ExcelWriter(save_xlsx, 'openpyxl') as writer:
        sheet_name = 'Version 1.0'
        pf.to_excel(writer, index=False, sheet_name=sheet_name)
        
        # 获取工作簿和工作表对象
        workbook  = writer.book
        worksheet = workbook[sheet_name]
        # 设置列宽
        worksheet.column_dimensions['B'].width = 30
        worksheet.column_dimensions['C'].width = 30
        worksheet.column_dimensions['D'].width = 100
        
    print("分析 Excel 文件保存到了:", save_xlsx)
    
    # 开始筛选值得的用户, 我们首先挑选出所有 user_id 和 nickname 的组合
    user_pf = pf[['user_id', 'nickname']]
    user_pf = user_pf.drop_duplicates()
    user_pf = user_pf.dropna()
    
    # 统计每个 user_id 和 nickname 对应的所有 pred, 给每一个用户一个分数
    user_score = []
    for index, row in user_pf.iterrows():
        preds = pf[pf['user_id'] == row['user_id']]['pred'].values.tolist()
        score = np.average([1 if i == '需求用户' else 0 for i in preds])
    
        user_score.append([row['user_id'], row['nickname'], score])
        
    # 保存所有值得的用户
    user_score = pd.DataFrame(user_score, columns=['user_id', 'nickname', 'score'])
    user_score = user_score.sort_values(by='score', ascending=False)
    
    print(user_score)
    with pd.ExcelWriter(save_id, 'openpyxl') as writer:
        sheet_name = 'V1'
        user_score.to_excel(writer, index=False, sheet_name=sheet_name)
        
        # 获取工作簿和工作表对象
        workbook  = writer.book
        worksheet = workbook[sheet_name]
        # 设置列宽
        worksheet.column_dimensions['A'].width = 30
        worksheet.column_dimensions['B'].width = 30
        worksheet.column_dimensions['C'].width = 20
        
    print("用户 Excel 文件保存到了:", save_id, '\n')
        
if __name__ == '__main__':
    opt = parse_opt()
    run_bert_model(**vars(opt))