#coding=utf-8
import torch
import os
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from dataset.dataset import TextDataset
from models.model import BertClassifierModule
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import numpy as np

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode', '--epoch', type=int, default=40, help='训练次数')
    parser.add_argument('--batch_size', type=int, default=32, help='每次训练送入的批量数')
    parser.add_argument('--max_len', type=int, default=50, help='句子的最大长度')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='学习步长（不是越大越好！！）')
    parser.add_argument('--num_labels', type=int, default=6, help='BERT 的参数，表示最后输出特征数')
    parser.add_argument('--dataset', type=str, default=os.path.join('dataset', 'aim', 'sampled_label.txt'), help='读取的文本位置')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='衰减率（不是越小越好！！）')
    parser.add_argument('--dropout_prop', '--dropout', '--drop', nargs='+', type=int, default=0.3, help='丢弃率（不是越小越好！！）')
    parser.add_argument('--pretraind_path', type=str, default='bert-chinese', help='预训练参数路径')
    parser.add_argument('--save_path', type=str, default=os.path.join('models', 'bert_model.pth'), help='预训练参数路径')
    parser.add_argument('--save_txt', type=str, default=os.path.join('runs', 'bert-report.txt'), help='训练报告输出路径')
    parser.add_argument('--human_test', type=bool, default=True, help='是否启用人工测试')
    return parser.parse_args()

# 训练 Bert 模型
def train_bert_model(
    episode = 28,
    batch_size = 32,
    max_len = 50,
    learning_rate = 2e-6,
    weight_decay = 1e-3,
    num_labels=6,
    dropout_prop=0.3,
    pretraind_path='bert-chinese',
    save_path=os.path.join('models', 'bert.pth'),
    save_txt=os.path.join('runs', 'bert-report.txt'),
    dataset=os.path.join('dataset', 'ad', 'train.txt'),
    human_test=False
    ):

    # 首先确认我们的模型运行在 GPU 上
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"模型将运行在 {device} 上")

    # 创建模型
    bert_model = BertClassifierModule(num_labels=num_labels, dropout_prop=dropout_prop, device=device, pretraind_path=pretraind_path)
    model = bert_model.get_model()
    tokenizer = bert_model.get_tokenizer()

    # 创建 Dataset 和 DataLoader
    ad_data = []
    ad_labels = []

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
            
    with open(dataset, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            part = line.strip().split('\t')
            if len(part) != 2:
                continue
            
            ad_labels.append(label_dict[part[0]])
            ad_data.append(part[1])
            
    print(pd.DataFrame({
        "label": ad_labels,
        "data": ad_data
    }))

    # 数据切割
    X_train, X_test, y_train, y_test = train_test_split(ad_data, ad_labels, test_size=0.2, random_state=42)

    train_dataset = TextDataset(X_train, y_train, tokenizer, max_len=max_len)
    test_dataset = TextDataset(X_test, y_test, tokenizer, max_len=max_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 创建损失函数和迭代器
    # 设置 bias 和 LayerNorm.weight 不使用 weight_decay
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    last_layer = nn.Softmax(dim=1)

    for epoch in range(episode):
        model.train()
        optimizer.zero_grad()
        
        losses = []
        with tqdm(train_loader, desc=f'Epoch {epoch + 1}') as tbar:
            for step, batch in enumerate(tbar, start=0):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids, attention_mask, token_type_ids)
                
                # 最后补一层 Softmax
                out = last_layer(outputs.logits)
                loss = criterion(out, labels)

                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
                tbar.set_postfix(loss=np.mean(losses))

    model.eval()
    # 存储真实标签和预测标签
    all_content = []
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        with tqdm(test_loader, desc='Test') as tbar:
            for batch in tbar:
                all_content.extend(batch['input_text'])
                
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids, attention_mask, token_type_ids)
                out = last_layer(outputs.logits)
                # print("out:", out)

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(torch.argmax(out, dim=1).cpu().numpy())
    
    with open(save_txt, 'w+', encoding='utf-8') as f:
        # 计算混淆矩阵
        cm = confusion_matrix(all_labels, all_predictions)
        f.write("混淆矩阵:\n" + str(cm) + "\n\n")
        print(cm)
        
        # 输出模型报告
        rp = classification_report(all_labels, all_predictions)
        f.write("分类报告:\n" + str(rp) + "\n")
        print(rp)
        
        # 输出测试案例
        pf = pd.DataFrame({
            'True': [type_dict[i] for i in all_labels],
            'Pred': [type_dict[i] for i in all_predictions],
            'Text': all_content,
        })
        f.write("测试案例:\n" + str(pf) + "\n")
        print(pf)
        pf.to_csv(os.path.join("dataset", 'valid.txt'), index=False, encoding='utf-8')
        
    # 创建子图
    fig, axes = plt.subplots(1, 1, figsize=(6, 5))

    # 绘制混淆矩阵
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes)
    axes.set_xlabel('Predicted')
    axes.set_ylabel('Actual')
    axes.set_title('Confusion Matrix')
    plt.tight_layout()
    plt.show()
    
    if human_test:
        while True:
            text = input("输入要判断的内容: ")
            if text == '' or text == 'q':
                print('退出最终人工测试')
                break
            
            inputs = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=max_len,
                pad_to_max_length = True,
                return_token_type_ids=True,
                truncation=True,
            )

            # 将编码后的输入转换为张量，并移动到 GPU 上
            input_ids = torch.tensor(inputs['input_ids']).unsqueeze(0).to(device)
            attention_mask = torch.tensor(inputs['attention_mask']).unsqueeze(0).to(device)
            token_type_ids = torch.tensor(inputs['token_type_ids']).unsqueeze(0).to(device)

            # 将输入传递给模型，并获取预测结果
            with torch.no_grad():
                outputs = model(input_ids, attention_mask, token_type_ids)
                predicted = torch.argmax(outputs.logits, 1).cpu().numpy()[0]

            # 打印预测结果
            print("预测结果:", type_dict[predicted])
            
    torch.save(model.state_dict(), save_path)
    print("Bert 模型保存到了:", save_path)
    
if __name__ == '__main__':
    opt = parse_opt()
    train_bert_model(**vars(opt))