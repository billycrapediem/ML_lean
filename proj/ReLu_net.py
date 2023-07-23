import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
delta = 2.0
class neuro_net(nn.Module):
    def __init__(self,input_dim:int, output_dim:int, hidden_layers = 2,mid_dim = 512) -> None:
        super(neuro_net,self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim,mid_dim),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Linear(mid_dim,mid_dim),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Linear(mid_dim,output_dim),
        )
        self.net_init()
    def net_init(self):
        array = [0,3,6]
        for i in array:
            nn.init.kaiming_normal_(self.linear[i].weight)
    def forward(self,input):
        x = self.linear(input)
        #x = torch.clamp(x,max= 1)
        return x







def train():
    ## hyper-parameter
    learning_rate = 1e-4
    n_episode = int(8e4)
    decay_rate = 0.95
    input_dim = 16
    output_dim = 6
    # get data

    data = np.load('process_data.npy')
    train_size = int(len(data) * 0.75)
    train_dataset = data[:train_size,:]
    test_dataset = data[train_size:,:]
    train_data = train_dataset[:,:16]
    train_lable = train_dataset[:,16:]
    test_data = test_dataset[:,:16]
    test_lable = test_dataset[:,16:]
    print(train_data.shape)

    
    test_data = torch.tensor(test_data,dtype=torch.float32,device=device)
    test_lable = torch.tensor(test_lable,dtype=torch.float32,device=device)
    train_data = torch.tensor(train_data,dtype=torch.float32,device=device)
    train_lable = torch.tensor(train_lable,dtype=torch.float32,device=device)

    ## build network
    net_work = neuro_net(input_dim,output_dim).to(device=device)
    #net_work.load_state_dict(torch.load('Relu_neuro_net.pt'))
    '''
    net_work = torch.load('model.pt')
    '''
    optimizer = optim.Adam(net_work.parameters(),lr=learning_rate,betas=(0.9,0.99),weight_decay=0.2)
    loss_func = nn.MSELoss()
    #start training
    for episode in range(n_episode):
        net_work.train()
        loss = 0
        predict_y = net_work(train_data)
        loss = loss_func(predict_y,train_lable)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #learning rate decay
        # evaluate
        if episode % 100 == 0:
            print(f'episode : {episode}, L1 smooth loss:{test_eval(test_data,test_lable, net_work)}, train loss{loss.item()}')
            torch.save(net_work.state_dict(),'Relu_neuro_net.pt')
        
    
    
    
def test_eval(test_data,test_lable,net_work):
    with torch.no_grad():
        test_data.to(device)
        predict_y = net_work(test_data)
        test_lable.to(device)
        loss = nn.MSELoss()(predict_y,test_lable)
        return loss


# eval
def eval(test_data, test_lable, net_work, final = False):
    network.eval()
        # eval the model
    with torch.no_grad():
        loss = 0
        eval_loss = []
        comparative_loss = []
        data = []
        #initialize the array 
        for i in range(6):
            array = []
            array_1 = []
            comparative_loss.append(array)
            data.append(array_1)
        for (data_x, data_y) in zip(test_data,test_lable):
            data_x = data_x
            data_x.to(device)
            data_y.to(device)
            predict_y = net_work(data_x)
            comparative = ((  data_y - predict_y) / data_y).numpy()
            for i in range (6):
                comparative_loss[i].append(comparative[i])
                data[i].append(data_y[i])
            loss = nn.MSELoss()(predict_y,data_y)
            eval_loss.append(loss.item())
        print(f'the MSE loss mean is :{np.array(eval_loss).mean()}')    
        if final : 
            fig, axes = plt.subplots(len(comparative_loss), 1, figsize=(6, 6 * len(comparative_loss)))
            for i in range (6):
                y = comparative_loss[i]
                print(f'datatype: {i}, comparative loss: {np.array(y).mean()}')
                x = data[i]
                ax = axes[i]
                ax.scatter(x,y)
                ax.set_xlabel('data')
                ax.set_ylabel('loss')
                ax.set_title(f'数据集{i+1}')
                ax.grid(True)
            plt.tight_layout()
            plt.savefig('Relu_test_loss.png')
        return np.array(eval_loss).mean()



if __name__ == '__main__':
    train()
    network = neuro_net(16,6).to(device=device)
    data = np.load('process_data.npy')
    train_size = int(len(data) * 0.75)
    test_data, test_label = data[:,:16], data[:,16:]

    network.load_state_dict(torch.load('Relu_neuro_net.pt'))
    
    print(test_label.shape)
    test_data = torch.tensor(test_data,dtype=torch.float32,device=device)
    test_label = torch.tensor(test_label,dtype=torch.float32,device=device)
    eval(test_data,test_label,network,True)
    
    
    
