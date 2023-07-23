import numpy as np
import matplotlib.pylab as plt
from ReLu_net import neuro_net
import torch
data = np.load('process_data.npy')
train_size = int(len(data) * 0.75)
data_Ft = data[:,16:]
data_Ft = data_Ft[:,:]
data = data[:,:16]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
network = neuro_net(16,6).to(device=device)
network.load_state_dict(torch.load('Relu_neuro_net.pt'))

data = torch.tensor(data,dtype=torch.float32,device=device)
data_Ft = torch.tensor(data_Ft,dtype=torch.float32,device=device)

 

comparative_loss = []
diffs = []
x_array = []
cnt =0
with torch.no_grad():
    network.eval()
    pred = network(data)
    for _ in range(6):
        array = []
        array_2 = []
        array3 = []
        diffs.append(array3)
        x_array.append(array_2)
        comparative_loss.append(array)
    for(real, p) in zip(data_Ft,pred):
        diff = abs(real-p)#(( -real + p) / real).numpy()
        comparative = ((  real - p) / real)
        for i in range(6):
            diffs[i].append(diff[i])
            comparative_loss[i].append(comparative[i])
            x_array[i].append(real[i])
    fig, axes = plt.subplots(len(comparative_loss), 1, figsize=(6, 6 * len(comparative_loss)))
    comparative_loss = np.array(comparative_loss)
    x_array = np.array(x_array)
    print(x_array.shape)
    for i in range (6):
        y1 = comparative_loss[i]
        y2 = diffs[i]
        print(f'variable: {i + 1}, abs average loss: {np.array(y2).mean()}')
        x = x_array[i]
        ax = axes[i]
        ax.scatter(x,y2)
        ax.set_xlabel('data')
        ax.set_ylabel('difference')
        ax.set_title(f'data set{i+1}')
        ax.grid(True)
    plt.tight_layout()
    print(cnt)
    plt.savefig('conver_test.png')
    