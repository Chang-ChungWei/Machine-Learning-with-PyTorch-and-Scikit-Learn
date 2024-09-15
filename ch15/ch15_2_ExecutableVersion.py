# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 12:11:51 2024

@author: user
"""

# coding: utf-8

# coding: utf-8

# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import os
import re
from collections import Counter, OrderedDict
from tqdm import tqdm
import tarfile
import urllib.request

# 如果需要手動下載 IMDB 資料集，可以使用以下方式下載
url = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
target_path = './aclImdb_v1.tar.gz'
urllib.request.urlretrieve(url, target_path)
with tarfile.open(target_path, 'r:gz') as tar:
    tar.extractall()

# 設置隨機種子以便重複結果
torch.manual_seed(1)

# 路徑設置
train_dir = './aclImdb/train'
test_dir = './aclImdb/test'

# 定義分詞函數
def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)  # 刪除HTML標籤
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    tokenized = text.split()
    return tokenized

# 加載數據集
class IMDbDataset(Dataset):
    def __init__(self, data_dir):
        self.data = self.load_dataset(data_dir)
        self.vocab = self.build_vocab()

    def load_dataset(self, data_dir):
        dataset = []
        for label in ['pos', 'neg']:
            label_dir = os.path.join(data_dir, label)
            for filename in tqdm(os.listdir(label_dir), desc=f'Loading {label} reviews'):
                with open(os.path.join(label_dir, filename), 'r', encoding='utf-8') as file:
                    text = file.read()
                    dataset.append((label, text))
        return dataset

    def build_vocab(self):
        token_counts = Counter()
        for label, line in tqdm(self.data, desc="Building Vocabulary"):
            tokens = tokenizer(line)
            token_counts.update(tokens)
        
        sorted_by_freq_tuples = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
        ordered_dict = OrderedDict(sorted_by_freq_tuples)

        # Create vocab object
        vocab = Vocab(ordered_dict)
        vocab.token2idx['<pad>'] = 0
        vocab.token2idx['<unk>'] = 1

        return vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        label, text = self.data[index]
        return text, 1 if label == 'pos' else 0

# 定義詞彙表類
class Vocab:
    def __init__(self, vocab_dict):
        self.vocab_dict = vocab_dict
        self.token2idx = {token: idx for idx, (token, _) in enumerate(self.vocab_dict.items())}
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}

    def __len__(self):
        return len(self.vocab_dict)

    def __getitem__(self, token):
        return self.token2idx.get(token, self.token2idx['<unk>'])

# 定義文本轉換函數
def text_pipeline(text, vocab):
    return [vocab[token] for token in tokenizer(text)]

# 定義數據加載函數
def collate_batch(batch, vocab):
    texts, labels = zip(*batch)
    text_list = [torch.tensor(text_pipeline(text, vocab), dtype=torch.int64) for text in texts]
    lengths = torch.tensor([len(text) for text in text_list])
    padded_text_list = nn.utils.rnn.pad_sequence(text_list, batch_first=True, padding_value=0)
    labels = torch.tensor(labels)
    return padded_text_list, labels, lengths

# 創建數據集實例
dataset = IMDbDataset(train_dir)
train_size = int(0.8 * len(dataset))
valid_size = len(dataset) - train_size
train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

# 轉換為 DataLoader
batch_size = 32
train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda b: collate_batch(b, dataset.vocab))
valid_dl = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda b: collate_batch(b, dataset.vocab))

# 定義RNN模型
class RNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, rnn_hidden_size, fc_hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.LSTM(embed_dim, rnn_hidden_size, batch_first=True)
        self.fc1 = nn.Linear(rnn_hidden_size, fc_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text, lengths):
        embedded = self.embedding(text)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (hidden, cell) = self.rnn(packed_embedded)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        out = self.fc1(hidden[-1])
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

# 初始化模型和定義損失函數與優化器
vocab_size = len(dataset.vocab)
embed_dim = 100  # 調整嵌入維度
rnn_hidden_size = 128  # 調整RNN隱藏層大小
fc_hidden_size = 64  # 調整全連接層大小
model = RNN(vocab_size, embed_dim, rnn_hidden_size, fc_hidden_size)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 訓練和評估函數
def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_acc = 0
    for text, labels, lengths in iterator:
        optimizer.zero_grad()
        predictions = model(text.to(device), lengths.to(device)).squeeze(1)
        rounded_preds = torch.round(predictions)
        correct = (rounded_preds == labels.to(device).float()).sum().item()
        acc = correct / labels.size(0)
        epoch_acc += acc
        loss = criterion(predictions, labels.to(device).float())
        loss.backward()
        optimizer.step()
    return epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_acc = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for text, labels, lengths in iterator:
            predictions = model(text.to(device), lengths.to(device)).squeeze(1)
            rounded_preds = torch.round(predictions)
            correct += (rounded_preds == labels.to(device).float()).sum().item()
            total += labels.size(0)
        epoch_acc = correct / total
    return epoch_acc

# 訓練模型
N_EPOCHS = 10
best_valid_acc = 0
for epoch in range(N_EPOCHS):
    train_acc = train(model, train_dl, optimizer, criterion)
    valid_acc = evaluate(model, valid_dl, criterion)
    print(f'Epoch: {epoch+1:02}, Train Acc: {train_acc:.3f}, Val. Acc: {valid_acc:.3f}')
    if valid_acc > best_valid_acc:
        best_valid_acc = valid_acc
        torch.save(model.state_dict(), 'rnn_model.pt')

# 評估模型
model.load_state_dict(torch.load('rnn_model.pt'))
test_dl = DataLoader(IMDbDataset(test_dir), batch_size=batch_size, shuffle=False, collate_fn=lambda b: collate_batch(b, dataset.vocab))
test_acc = evaluate(model, test_dl, criterion)
print(f'Test Acc: {test_acc:.3f}')
