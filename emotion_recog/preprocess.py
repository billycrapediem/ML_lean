import torch
import os
import numpy as np
from gensim.models import Word2Vec
from torch.utils import data
from torch import nn
import pandas as pd


def load_training_data(path = 'training_label.txt'):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip('\n').split(' ') for line in lines]
    x = [line[2:] for line in lines]
    y = [line[0] for line in lines]
    return x,y 

def load_testing_data(path = 'testing_data'):
    with open(path, 'r') as f:
        lines = f.readlines()
    x = ["".join(line.strip('\n').split(",")[1:]).strip() for line in lines[1:]]
    x = [sen.split(' ') for sen in x]
    return x

def evaluation(outputs, labels):
    outputs[outputs >= 0.5] = 1
    outputs[outputs < 0.5] = 0
    loss = torch.sum(torch.eq(outputs,labels)).item()
    return loss

class Preprocess():
    def __init__(self,sentences, sen_len, w2v_path = "./w2vall.model"):
        self.w2v_path = w2v_path
        self.sentences = sentences 
        self.sen_len = sen_len # mark the longest sentence length
        self.idx2word =[]
        self.word2idx = {}
        self.embedding_matrix = []

    def get_w2v_model(self):
        ## 加载模型
        self.embedding = Word2Vec.load(self.w2v_path)
        self.embedding_dim = self.embedding.vector_size
    
    def add_embedding(self,word):
        vector =torch.empty(1,self.embedding_dim)
        torch.nn.init.uniform_(vector) # 随机创建值
        self.word2idx[word] = len(self.word2idx) ##
        self.embedding_matrix = torch.cat([self.embedding_matrix,vector],0) ## 竖着平接
    
    def make_embedding(self, load=True):
        print("Get embedding ...")
        # 取得訓練好的 Word2vec word embedding
        if load:
            print("loading word to vec model ...")
            self.get_w2v_model()
        else:
            raise NotImplementedError
        cnt = 0
        for word in self.embedding.wv.index_to_key:
            print('get words #{}'.format(cnt+1), end='\r')
            #e.g. self.word2index['he'] = 1 
            #e.g. self.index2word[1] = 'he'
            #e.g. self.vectors[1] = 'he' vector
            self.word2idx[word] = len(self.word2idx)
            self.idx2word.append(word)
            self.embedding_matrix.append(self.embedding.wv[word])
            cnt = cnt + 1
        print('')
        self.embedding_matrix = torch.tensor(self.embedding_matrix)
        # 將 "<PAD>" 跟 "<UNK>" 加進 embedding 裡面
        self.add_embedding("<PAD>")
        self.add_embedding("<UNK>")
        print("total words: {}".format(len(self.embedding_matrix)))
        return self.embedding_matrix
    def pad_sequence(self, sentence):
        if len(sentence) > self.sen_len:
            sentence = sentence[:self.sen_len]
        else:
            pad_len = self.sen_len - len(sentence)
            for _ in range(pad_len):
                sentence.append(self.word2idx["<PAD>"])
        assert len(sentence) == self.sen_len
        return sentence

    def sentence_word2idx(self):
        # 把句子裡面的字轉成相對應的 index
        sentence_list = []
        for i, sen in enumerate(self.sentences):
            print('sentence count #{}'.format(i+1), end='\r')
            sentence_idx = []
            for word in sen:
                if (word in self.word2idx.keys()):
                    sentence_idx.append(self.word2idx[word])
                else:
                    sentence_idx.append(self.word2idx["<UNK>"])
            # 將每個句子變成一樣的長度
            sentence_idx = self.pad_sequence(sentence_idx)
            sentence_list.append(sentence_idx)
        return  torch.LongTensor(sentence_list)
    def labels_to_tensor(self, y):
        # 把 labels 轉成 tensor
        y = [int(label) for label in y]
        return torch.LongTensor(y)
    
class WordDataset(data.Dataset):
    def __init__(self,x,y):
        self.data = x
        self.label = y
    def __getitem__(self,idx):
        if self.label is None:
            return self.data[idx]
        return self.data[idx], self.label[idx]
    def __len__(self):
        return len(self.data)

class LSTM_Net(nn.Module):
    def __init__(self, embedding, embedding_dim, hidden_dim, num_layers, dropout=0.5, fix_embedding=True):
        super(LSTM_Net, self).__init__()
        # 製作 embedding layer
        self.embedding = torch.nn.Embedding(embedding.size(0),embedding.size(1))
        self.embedding.weight = torch.nn.Parameter(embedding)
        # 是否將 embedding fix 住，如果 fix_embedding 為 False，在訓練過程中，embedding 也會跟著被訓練
        self.embedding.weight.requires_grad = False if fix_embedding else True
        self.embedding_dim = embedding.size(1)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.classifier = nn.Sequential( nn.Dropout(dropout),
                                         nn.Linear(hidden_dim, 1),
                                         nn.Sigmoid() )
    def forward(self, inputs):
        inputs = self.embedding(inputs)
        x, _ = self.lstm(inputs, None)
        # x 的 dimension (batch, seq_len, hidden_size)
        # 取用 LSTM 最後一層的 hidden state
        x = x[:, -1, :] 
        x = self.classifier(x)
        return x

import torch.optim as optim
import torch.nn.functional as F

def training(batch_size, n_epoch, lr, model_dir, train, valid, model, device):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\nstart training, parameter total:{}, trainable:{}\n'.format(total, trainable))
    model.train() # 將 model 的模式設為 train，這樣 optimizer 就可以更新 model 的參數
    criterion = nn.BCELoss() # 定義損失函數，這裡我們使用 binary cross entropy loss
    t_batch = len(train) 
    v_batch = len(valid) 
    optimizer = optim.Adam(model.parameters(), lr=lr) # 將模型的參數給 optimizer，並給予適當的 learning rate
    total_loss, total_acc, best_acc = 0, 0, 0
    for epoch in range(n_epoch):
        total_loss, total_acc = 0, 0
        # 這段做 training
        for i, (inputs, labels) in enumerate(train):
            inputs = inputs.to(device, dtype=torch.long) # device 為 "cuda"，將 inputs 轉成 torch.cuda.LongTensor
            labels = labels.to(device, dtype=torch.float) # device為 "cuda"，將 labels 轉成 torch.cuda.FloatTensor，因為等等要餵進 criterion，所以型態要是 float
            optimizer.zero_grad() # 由於 loss.backward() 的 gradient 會累加，所以每次餵完一個 batch 後需要歸零
            outputs = model(inputs) # 將 input 餵給模型
            labels = labels.view(-1,1)
            loss = criterion(outputs, labels)
            loss.backward() # 算 loss 的 gradient
            optimizer.step() # 更新訓練模型的參數
            correct = evaluation(outputs, labels) # 計算此時模型的 training accuracy
            total_acc += (correct / batch_size)
            total_loss += loss.item()
            print('[ Epoch{}: {}/{} ] loss:{:.3f} acc:{:.3f} '.format(
            	epoch+1, i+1, t_batch, loss.item(), correct*100/batch_size), end='\r')
        print('\nTrain | Loss:{:.5f} Acc: {:.3f}'.format(total_loss/t_batch, total_acc/t_batch*100))

        # 這段做 validation
        model.eval() # 將 model 的模式設為 eval，這樣 model 的參數就會固定住
        with torch.no_grad():
            total_loss, total_acc = 0, 0
            for i, (inputs, labels) in enumerate(valid):
                inputs = inputs.to(device, dtype=torch.long) # device 為 "cuda"，將 inputs 轉成 torch.cuda.LongTensor
                labels = labels.to(device, dtype=torch.float) # device 為 "cuda"，將 labels 轉成 torch.cuda.FloatTensor，因為等等要餵進 criterion，所以型態要是 float
                outputs = model(inputs) # 將 input 餵給模型
                labels = labels.view(-1,1)
                loss = criterion(outputs, labels) # 計算此時模型的 validation loss
                correct = evaluation(outputs, labels) # 計算此時模型的 validation accuracy
                total_acc += (correct / batch_size)
                total_loss += loss.item()

            print("Valid | Loss:{:.5f} Acc: {:.3f} ".format(total_loss/v_batch, total_acc/v_batch*100))
            if total_acc > best_acc:
                # 如果 validation 的結果優於之前所有的結果，就把當下的模型存下來以備之後做預測時使用
                best_acc = total_acc
                #torch.save(model, "{}/val_acc_{:.3f}.model".format(model_dir,total_acc/v_batch*100))
                torch.save(model, "{}/ckpt.model".format(model_dir))
                print('saving model with acc {:.3f}'.format(total_acc/v_batch*100))
        print('-----------------------------------------------')
        model.train() # 將 model 的模式設為 train，這樣 optimizer 就可以更新 model 的參數（因為剛剛轉成 eval 模式）
def testing(batch_size, test_loader, model, device):
    model.eval()
    ret_output = []
    with torch.no_grad():
        for i, inputs in enumerate(test_loader):
            inputs = inputs.to(device, dtype = torch.long)
            outputs = model(inputs)
            outputs = outputs.squeeze()
            outputs[outputs>=0.5] = 1 # 大於等於 0.5 為正面
            outputs[outputs<0.5] = 0 # 小於 0.5 為負面
            ret_output += outputs.int().tolist()
    return ret_output

if __name__ == '__main__':
    
    path_prefix = './'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_with_label = os.path.join(path_prefix, 'training_label.txt')
    testing_data = os.path.join(path_prefix, 'testing_data.txt')

    w2v_path = os.path.join(path_prefix, 'w2v_all.model')

    sen_len = 20
    fix_embedding = False
    batch_size = 512
    epoch = 5
    lr = (1e-2)
    
    print("loading data...")
    train_x, y = load_training_data(train_with_label)
    preprocess = Preprocess(train_x,sen_len,w2v_path)
    embedding = preprocess.make_embedding(True)
    train_x = preprocess.sentence_word2idx()
    y = preprocess.labels_to_tensor(y)
    X_train, X_val, y_train, y_val = train_x[:180000], train_x[180000:], y[:180000], y[180000:]
    model = LSTM_Net(embedding=embedding,embedding_dim=250,hidden_dim=150,num_layers=1,dropout=0.5, fix_embedding=fix_embedding)
    model = model.to(device=device)

    train_dataset = WordDataset(X_train,y_train)
    train_loader = data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=8)
    valid_dataset = WordDataset(X_val,y_val)
    valid_loader = data.DataLoader(valid_dataset, batch_size, shuffle=True, num_workers=8)
    training(batch_size,epoch,lr,path_prefix,train_loader,valid_loader,model,device)


    
    print("loading testing data ...")
    test_x = load_testing_data(testing_data)
    preprocess = Preprocess(test_x, sen_len, w2v_path=w2v_path)
    embedding = preprocess.make_embedding(load=True)
    test_x = preprocess.sentence_word2idx()
    test_dataset = WordDataset(test_x, y=None)
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                            batch_size = batch_size,
                                            shuffle = False,
                                            num_workers = 8)
    print('\nload model ...')
    model = torch.load(os.path.join(path_prefix, 'ckpt.model'))
    outputs = testing(batch_size, test_loader, model, device)
    tmp = pd.DataFrame({"id":[str(i) for i in range(len(test_x))],"label":outputs})
    print("save csv ...")
    tmp.to_csv(os.path.join(path_prefix, 'predict.csv'), index=False)