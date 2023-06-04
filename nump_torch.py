import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
'''
np_data = np.arange(6).reshape((2,3))
torch_data = torch.from_numpy(np_data)

tensor2array = torch_data.numpy()
print(
    np_data,
    torch_data,
    tensor2array,
)

#abs
data = [-1,-2,1,2]
tensor = torch.FloatTensor(data) # 32bit

print(
    '/abs',torch.abs(tensor)
)


tensor = torch.FloatTensor([[1,2],[3,4]])
variable = Variable(tensor,requires_grad=True)

t_out = torch.mean(tensor * tensor)
v_out = torch.mean(variable * variable)
print(t_out)
print(v_out)

v_out.backward()
## how to calculate grad
## v_out = 1/4 * sum(var* var)
# d(v_out)/d(var) = 1/4 * 2 * var = variable /2
print(variable.grad)
print(variable)
print(variable.data)
'''
'''
why we need activation function?
to deal with problem that we can not explain with linear function 
'''


## regression

x = torch.unsqueeze(torch.linspace(-1,1,100),dim = 1)
y = x.pow(2) + 0.2 * torch.rand(x.size())

x, y  = Variable(x), Variable(y)

#plt.scatter(x.data.numpy(), y.data.numpy())
#plt.show()
class Net(torch.nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature,n_hidden)
        self.predict = torch.nn.Linear(n_hidden,n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        output = self.predict(x)
        return output

net = Net(1, 10, 1)
print(net)

## optimization
plt.ion()
plt.show()


optimizer = torch.optim.SGD(net.parameters(), lr = 0.5)
loss_func = torch.nn.MSELoss()

for t in range (200):
    prediction = net(x)
    ## get the different
    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step() 
    if t % 5 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), y.data.numpy())
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)
plt.ioff()
plt.show()


'''
## method 1
class Net(torch.nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature,n_hidden)
        self.predict = torch.nn.Linear(n_hidden,n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        output = self.predict(x)
        return output

## classification

n_data = torch.ones(100, 2)
x0 = torch.normal(2*n_data, 1)      # class0 x data (tensor), shape=(100, 2)
y0 = torch.zeros(100)               # class0 y data (tensor), shape=(100, 1)
x1 = torch.normal(-2*n_data, 1)     # class1 x data (tensor), shape=(100, 2)
y1 = torch.ones(100)                # class1 y data (tensor), shape=(100, 1)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # shape (200, 2) FloatTensor = 32-bit floating
y = torch.cat((y0, y1), ).type(torch.LongTensor)    # shape (200,) LongTensor = 64-bit integer

net1 = Net(2,40,2)
net2 = torch.nn.Sequential(
    torch.nn.Linear(2,10),
    torch.nn.ReLU(),
    torch.nn.Linear(10,2),
)

print(net2)
print(net1)
optimizer = torch.optim.SGD(net1.parameters(), lr = 0.02)
loss_func = torch.nn.CrossEntropyLoss()



plt.ion()
plt.show()


for t in range (200):
    out = net2(x)
    loss = loss_func(out,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if t % 2 == 0:
        # plot and show learning process
        plt.cla()
        prediction = torch.max(out, 1)[1]
        pred_y = prediction.data.numpy()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
'''
### save the neuro network
x = torch.unsqueeze(torch.linspace(-1,1,100),dim = 1)
y = x.pow(2) + 0.2 * torch.rand(x.size())
def save():
    # save net1
    net1 = torch.nn.Sequential(
        torch.nn.Linear(1,10),
        torch.nn.ReLU(),
        torch.nn.Linear(10,1)
    )
    optimizer = torch.optim.SGD(net1.parameters(),lr = 0.5)
    loss_func = torch.nn.MSELoss()
    for t in range(100):
        prediction = net1(x)
        loss = loss_func(prediction,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    torch.save(net1,'net.pkl')
    torch.save(net1.state_dict(),'net_params.pkl')

def restore_net():
    net2 = torch.load('net.pkl')

def restore_params():
    net3 = torch.nn.Sequential(
        torch.nn.Linear(1,10),
        torch.nn.ReLU(),
        torch.nn.Linear(10,1)
    )
    net3.load_state_dict(torch.load('net_params.pkl'))
    


