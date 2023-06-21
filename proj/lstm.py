import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import matplotlib.pyplot as plt

class lSTM_net(nn.Module):
    def __init__(self,input_dim:int, output_dim:int, hidden_layers = 2,mid_dim = 128) -> None:
        super(lSTM_net,self).__init__()
        self.lstm = nn.LSTM(input_dim,mid_dim,hidden_layers)
        self.linear = nn.Sequential(
            nn.Linear(mid_dim,mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim,output_dim)
        )
    def forward(self,input):
        x = self.lstm(input)[0]
        x = self.linear(x)
        return x

def data_preprocess():
    data_state = np.load('state.npy')
    data_vel = np.load('vel.npy')
    data_Ft = np.load('Ft.npy')
    #process the euler angle to normalize into [0,1]
    min_value = -np.pi
    angle_range = 2 * np.pi
    for i in range (3,6):
        data_state[i,:] = (data_state[3,:] - min_value) / angle_range
    data = np.concatenate((data_state,data_vel),axis=0)
    data = np.transpose(data)
    data_Ft = np.transpose(data_Ft)
    return data,data_Ft




def train():
    ## hyper-parameter
    learning_rate = 1e-3
    batch_size = 128
    n_episode = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = 16
    output_dim = 6
    # get data

    data, label_value = data_preprocess()
    train_size = int(len(data) * 0.75)
    train_data = data[:train_size,:]
    train_lable = label_value[:train_size,:]
    test_data = data[:train_size,:]
    test_lable = label_value[:train_size,:]
    
    test_data = torch.tensor(test_data,dtype=torch.float32,device=device)
    test_lable = torch.tensor(test_lable,dtype=torch.float32,device=device)
    train_data = torch.tensor(train_data,dtype=torch.float32,device=device)
    train_lable = torch.tensor(train_lable,dtype=torch.float32,device=device)
    train_tensor_set = Data.TensorDataset(train_data,train_lable)
    train_loader = Data.DataLoader(dataset=train_tensor_set,batch_size=batch_size,shuffle=True,num_workers=4)

    ## build network
    net_work = lSTM_net(input_dim,output_dim)
    optimizer = optim.Adam(net_work.parameters(),lr=learning_rate)
    loss_func = nn.MSELoss()
    loss_value = []
    #start training
    
    
    for episode in range(n_episode):
        loss = 0
        for batch_x, batch_y in train_loader:
            predict_y = net_work(batch_x)
            loss = loss_func(predict_y,batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_value.append(loss.item())
        if episode % 5 == 0:
            print(f'episode : {episode}, loss:{loss}')
    plt.plot(loss_value)
    plt.show()
    
    # eval the model
    eval_loss = []
    for (data_x, data_y) in zip(test_data,test_lable):
        data_x = data_x.unsqueeze(0)
        print(data_x.shape)
        predict_y = net_work(data_x)
        loss = loss_func(predict_y,data_y)
        eval_loss.append(loss.item())
    plt.plot(eval_loss)
    plt.show()




if __name__ == '__main__':
    train()


