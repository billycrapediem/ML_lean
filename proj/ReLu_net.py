import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import matplotlib.pyplot as plt


device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

class neuro_net(nn.Module):
    def __init__(self,input_dim:int, output_dim:int, hidden_layers = 2,mid_dim = 256) -> None:
        super(neuro_net,self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim,mid_dim),
            nn.LeakyReLU(),
            nn.Linear(mid_dim,mid_dim),
            nn.LeakyReLU(),
            nn.Linear(mid_dim,output_dim),
            nn.ReLU()
        )
    def forward(self,input):
        x = self.linear(input)
        x = torch.clamp(x,max= 1)
        return x

def data_preprocess():
    data_state = np.load('state.npy')
    data_vel = np.load('vel.npy')
    data_Ft = np.load('Ft.npy')
    print(data_state[6,:].max())
    #process the euler angle to normalize into [-1,1]
    min_value = -np.pi
    angle_range = 2 * np.pi
    # normalize the angle 
    for i in range (3,6):
        data_state[i,:] = (data_state[i,:] - min_value) / (angle_range)
    #try to not normalize the output 
    '''
    for i in range (0,3):
        data_Ft[i,:] = (data_Ft[i,:] - (-50)) / (100)
    for i in range (3,6):
        data_Ft[i,:] = (data_Ft[i,:] - (-5)) / (10)
    '''
    data = np.concatenate((data_state,data_vel),axis=0)
    data = np.transpose(data)
    data_Ft = np.transpose(data_Ft)
    #return data,data_Ft
    data = np.concatenate((data,data_Ft),axis=1)
    np.random.shuffle(data)
    np.save('process_data.npy',data) 




def train():
    ## hyper-parameter
    learning_rate = 1e-4
    batch_size = 5096
    n_episode = 5000
    
    input_dim = 16
    output_dim = 6
    # get data

    data = np.load('process_data.npy')
    train_size = int(len(data) * 0.75)
    np.random.shuffle(data)
    train_dataset = data[:train_size,:]
    test_dataset = data[train_size:,:]
    train_data = train_dataset[:,:16]
    train_lable = train_dataset[:,16:]
    test_data = test_dataset[:,:16]
    test_lable = test_dataset[:,16:]
    print(train_data.shape)

    
    test_data = torch.tensor(test_data,dtype=torch.float32)
    test_lable = torch.tensor(test_lable,dtype=torch.float32)
    train_data = torch.tensor(train_data,dtype=torch.float32)
    train_lable = torch.tensor(train_lable,dtype=torch.float32)
    train_tensor_set = Data.TensorDataset(train_data,train_lable)
    train_loader = Data.DataLoader(dataset=train_tensor_set,batch_size=batch_size,shuffle=True,num_workers=4)

    ## build network
    net_work = neuro_net(input_dim,output_dim)
    #net_work.load_state_dict(torch.load('Relu_neuro_net.pt'))
    '''
    net_work = torch.load('model.pt')
    '''
    optimizer = optim.Adam(net_work.parameters(),lr=learning_rate)
    loss_func = nn.MSELoss()
    loss_value = []
    valid_loss = []
    
    #start training
    for episode in range(n_episode):
        loss = 0
        for batch_x, batch_y in train_loader:
            batch_x.to(device)
            batch_y.to(device)
            predict_y = net_work(batch_x)
            loss = loss_func(predict_y,batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_value.append(loss.item())
            valid_loss.append(test_eval(test_data,test_lable, net_work))
        if episode % 5 == 0:
            print(f'episode : {episode}, loss:{loss}')

    
    torch.save(net_work.state_dict(),'Relu_neuro_net.pt')
    eval(test_data,test_lable, net_work, True)
    
def test_eval(test_data,test_lable,net_work):
    with torch.no_grad():
        test_data.to(device)
        predict_y = net_work(test_data)
        test_lable.to(device)
        loss = nn.MSELoss()(predict_y,test_lable)
        print(f'MES loss is {loss}')
        return loss


# eval
def eval(test_data, test_lable, net_work, final = False):
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
            comparative = (( - data_y + predict_y) / data_y).numpy()
            for i in range (6):
                comparative_loss[i].append(comparative[i])
                data[i].append(data_y[i])
            loss = nn.MSELoss()(predict_y,data_y)
            eval_loss.append(loss.item())
            plt.scatter(torch.sum(data_x),loss,c='c')
        plt.xlabel("value")
        plt.ylabel("loss")
        plt.title('loss')
        plt.savefig('Relu_test_mseloss.png')
        #plt.plot(eval_loss,label = "loss")
        print(f'the MSE loss mean is :{np.array(eval_loss).mean()}')    
        if final : 
            fig, axes = plt.subplots(len(comparative_loss), 1, figsize=(6, 6 * len(comparative_loss)))
            for i in range (6):
                y = comparative_loss[i]
                print(f'datatype: {i}, comparative loss: {np.array(x).mean()}')
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
    


    '''
    network = neuro_net(16,6)
    data = np.load('process_data.npy')
    test_data, test_label = data[:,:16], data[:,16:]

    network.load_state_dict(torch.load('Relu_neuro_net.pt'))
    
    print(test_label.shape)
    test_data = torch.tensor(test_data,dtype=torch.float32)
    test_label = torch.tensor(test_label,dtype=torch.float32)
    eval(test_data,test_label,network,True)
    '''
    
    
