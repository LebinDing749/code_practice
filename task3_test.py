import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from collections import Counter
import statistics


def load_imdb_data(data_dir):
    """
    加载 IMDB 数据集并返回训练集和测试集的文本和标签。
    :param data_dir: 数据集所在的目录路径
    :return: 训练集文本、训练集标签、测试集文本、测试集标签
    """
    train_texts, train_labels = [], []
    for category in ['pos', 'neg']:
        train_dir = os.path.join(data_dir, 'train', category)
        for filename in os.listdir(train_dir):
            with open(os.path.join(train_dir, filename), 'r', encoding='utf-8') as file:
                text = file.read().strip()
                train_texts.append(text)
                train_labels.append(1 if category == 'pos' else 0)

    test_texts, test_labels = [], []
    for category in ['pos', 'neg']:
        test_dir = os.path.join(data_dir, 'test', category)
        for filename in os.listdir(test_dir):
            with open(os.path.join(test_dir, filename), 'r', encoding='utf-8') as file:
                text = file.read().strip()
                test_texts.append(text)
                test_labels.append(1 if category == 'pos' else 0)

    return train_texts, train_labels, test_texts, test_labels


def build_vocab(texts, max_vocab_size=None):
    """
    构建词汇表并返回词汇表对象和词汇表大小。
    :param texts: 文本列表
    :param max_vocab_size: 最大词汇表大小
    :return: 词汇表对象、词汇表大小
    """
    counter = Counter()
    for text in texts:
        counter.update(text.split())
    if max_vocab_size is not None:
        counter = dict(counter.most_common(max_vocab_size))
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for i, (word, freq) in enumerate(counter.items(), 2):
        vocab[word] = i
    return vocab, len(vocab)


def convert_texts_to_ids(texts, vocab):
    """
    将文本列表转换为 ID 列表。
    :param texts: 文本列表
    :param vocab: 词汇表对象
    :return: ID 列表
    """
    unk_id = vocab['<UNK>']
    return [[vocab.get(word, unk_id) for word in text.split()] for text in texts]


class IMDBDataset(Dataset):
    """
    IMDB 数据集类，用于加载和处理 IMDB 数据集。
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text, label = self.data[index]
        text = torch.tensor(text)
        label = torch.tensor(label)
        return text, label
class LSTMClassifier(nn.Module):
    """
    LSTM 分类器，用于对 IMDB 数据集进行二分类。
    """
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, num_classes, dropout_prob):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        h_n = h_n[-1, :, :]
        out = self.fc(h_n)
        return out

class GRUClassifier(nn.Module):
    """
    GRU 分类器，用于对 IMDB 数据集进行二分类。
    """
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, num_classes, dropout_prob):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, h_n = self.gru(x)
        h_n = h_n[-1, :, :]
        out = self.fc(h_n)
        return out

def same_len(text,size):
    text_1=[]
    for i in text:
        if(len(i)>size):
            i=i[:size]
        while(len(i)<size):
            i.append(0)
        text_1.append((i))
    return text_1

if __name__ =="__main__":
    # 设置超参数
    max_vocab_size = 10000
    embedding_size = 128
    hidden_size = 64
    num_layers = 2
    num_classes = 2
    dropout_prob = 0.5
    batch_size = 1
    learning_rate = 0.001
    num_epochs = 6

    # 加载 IMDB 数据集并构建词汇表
    data_dir = 'C:/Users/lenovo/Downloads/aclImdb_v1/aclImdb'
    train_texts, train_labels, test_texts, test_labels = load_imdb_data(data_dir)
    vocab, vocab_size = build_vocab(train_texts, max_vocab_size)
    train_ids = convert_texts_to_ids(train_texts, vocab)
    test_ids = convert_texts_to_ids(test_texts, vocab)
    text_len=[len(i) for i in train_ids]
    cut_len=int(statistics.mean(text_len))
    train_ids=same_len(train_ids,cut_len)
    test_ids=same_len(test_ids,cut_len)

    # 将 ID 列表转换为张量并分批
    train_data = [(ids, label) for ids,label in zip(train_ids, train_labels)]
    test_data = [(ids, label) for ids, label in zip(test_ids, test_labels)]
    train_dataset = IMDBDataset(train_data)
    test_dataset = IMDBDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 创建模型
    #model = LSTMClassifier(vocab_size, embedding_size, hidden_size, num_layers, num_classes, dropout_prob)
    #model.load_state_dict(torch.load('LSTM.pth'))
    model = GRUClassifier(vocab_size, embedding_size, hidden_size, num_layers, num_classes, dropout_prob)
    model.load_state_dict(torch.load('GRU.pth'))

    # 在测试集上评估模型
    model.eval()
    with torch.no_grad():
        correct = [0, 0, 0]
        total = [0, 0, 0]
        for inputs, labels in test_loader:
            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)

            if(labels == 0):
                total[0] = total[0]+1
            else:
                total[1] = total[1]+1
            total[2] = total[2]+1

            if(predicted == labels):
                correct[labels] = correct[labels]+1
                correct[2] = correct[2]+1
        print(float(correct[0]/total[0]))
        print(float(correct[1] / total[1]))
        print(float(correct[2] / total[2]))

