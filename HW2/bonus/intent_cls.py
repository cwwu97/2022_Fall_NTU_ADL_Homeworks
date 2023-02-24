import sys
# sys.path.append('C:\\Users\\joung_msi_1\\R10725051\\r10725051\\lib\\site-packages') # delete before submission
import os
import argparse

import pandas as pd
import numpy as np
from transformers import BertModel, BertForSequenceClassification, BertTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss, Dropout, ReLU, Linear, Module
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tqdm import tqdm

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, default='./data/intent/train.json', help='Train file directory.')
    parser.add_argument('--validation_file', type=str, default='./data/intent/eval.json', help='Validation file directory.')
    parser.add_argument('--max_seq_len', type=int, default=32, help='Padding and truncation length.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_epoch', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--num_label', type=int, default=150, help='Number of clssification labels')
    args = parser.parse_args()
    return args

class BERT_Arch(Module):
    def __init__(self, model):
        super(BERT_Arch, self).__init__()
        self.model = model
        self.dropout = Dropout(0.2)
        self.relu = ReLU()
        self.fc1 = Linear(768,512)
        self.fc2 = Linear(512,args.num_label)

    def forward(self, sent_id, mask):
      _, cls_hs = self.model(sent_id, attention_mask=mask, return_dict=False)
      x = self.fc1(cls_hs)
      x = self.relu(x)
      x = self.dropout(x)
      x = self.fc2(x)
      return x

if __name__ == '__main__':

    args = parse_args()

    # 1. Load File As DataFrame
    train = pd.read_json(args.train_file)
    val = pd.read_json(args.validation_file)

    # 2. Encode Intent Label
    le = LabelEncoder()
    le.fit(train['intent'])
    train['intent_label'] = le.transform(train['intent'])
    val['intent_label'] = le.transform(val['intent'])

    # 3. Tokenize
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    tokens_train = tokenizer.batch_encode_plus(
        train['text'].tolist(),
        max_length = args.max_seq_len,
        padding='max_length',
        truncation=True,
        return_token_type_ids=False,
    )
    tokens_val = tokenizer.batch_encode_plus(
        val['text'].tolist(),
        max_length = args.max_seq_len,
        padding='max_length',
        truncation=True,
        return_token_type_ids=False,
    )

    # 4. Data to Tensor
    train_seq = torch.tensor(tokens_train['input_ids'], dtype=torch.long)
    train_mask = torch.tensor(tokens_train['attention_mask'], dtype=torch.long)
    train_y = torch.tensor(train['intent_label'].tolist(), dtype=torch.long)

    val_seq = torch.tensor(tokens_val['input_ids'], dtype=torch.long)
    val_mask = torch.tensor(tokens_val['attention_mask'], dtype=torch.long)
    val_y = torch.tensor(val['intent_label'].tolist(), dtype=torch.long)

    # 5. Create DataLoader
    train_data = TensorDataset(train_seq, train_mask, train_y) 
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
 
    val_data = TensorDataset(val_seq, val_mask, val_y)
    val_dataloader = DataLoader(val_data, shuffle=False, batch_size=args.batch_size)

    # 6. Load Model
    # model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels = args.num_label, output_attentions = False, output_hidden_states = False)
    model = BertModel.from_pretrained('bert-base-uncased')
    for param in model.parameters():
        param.requires_grad = False
    model = BERT_Arch(model)    
    model = model.to(DEVICE)

    # 7. Optimizer
    optimizer = AdamW(model.parameters(), lr = 1e-3)

    # 8. Loss function
    cross_entropy  = CrossEntropyLoss()

    # 9. Training and Validation
    total_train_loss, total_valid_loss = [], []
    total_train_acc, total_valid_acc = [], []
    for epoch in range(args.num_epoch):
        print(f'Epoch {epoch+1}/{args.num_epoch}')
        # Training...
        model.train()
        train_loss, train_acc = 0, 0
        train_preds=[]
        for step, batch in enumerate(tqdm(train_dataloader)):
            batch = [r.to(DEVICE) for r in batch]
            sent_id, mask, label = batch  
            model.zero_grad()
            preds = model(sent_id, mask)
            loss = cross_entropy(preds, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            preds=preds.detach().cpu().numpy()
            train_preds.append(preds)
        train_loss = train_loss/len(train_dataloader)
        train_preds = np.concatenate(train_preds, axis=0)

        # Validating
        model.eval()
        val_loss, val_acc = 0, 0
        val_preds = []
        for step, batch in enumerate(tqdm(val_dataloader)):
            batch = [r.to(DEVICE) for r in batch]
            sent_id, mask, label = batch
            with torch.no_grad():
                preds = model(sent_id, mask)
                loss = cross_entropy(preds, label)
                val_loss += loss.item()
                preds=preds.detach().cpu().numpy()
                val_preds.append(preds)
        val_loss = val_loss/len(val_dataloader)
        val_preds = np.concatenate(val_preds, axis=0)
        val_acc = accuracy_score(val_y , np.argmax(val_preds, axis=-1))

        print(f'Training Loss: {train_loss:.3f} | Validation Loss: {val_loss:.3f} | Validation Acc: {val_acc:.3f}')