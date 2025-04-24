import torch, os
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification

class BertClassifierModule():
    def __init__(self, num_labels=2, dropout_prop=0.3, device='cuda:0' if torch.cuda.is_available() else 'cpu', pretraind_path=os.path.join('models', 'bert-chinese')):
        # 获取 Bert 的 Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(pretraind_path)
        # 获取 Bert 的模型结构
        self.config = BertConfig.from_pretrained("bert-chinese", num_labels=num_labels, hidden_dropout_prob=dropout_prop)
        self.model = BertForSequenceClassification.from_pretrained("bert-chinese", config=self.config).to(device)
        print('模型结构:', self.model, '\n')
        
    def get_model(self):
        return self.model
    
    def get_config(self):
        return self.config
    
    def get_tokenizer(self):
        return self.tokenizer