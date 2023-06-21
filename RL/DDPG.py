import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
import random
from case3d_env import chase3D
import gym



# actor net
class actor_net(nn.Module):
    def __init__(self, input_dim:int, output_dim:int,max_action, hidden_dim = 128,init_v = 1e-5) -> None:
        super(actor_net,self).__init__()
        self.linear1 = nn.Linear(input_dim,hidden_dim)
        self.linear2 = nn.Linear(hidden_dim,output_dim)
        self.linear2.weight.data.uniform_(-init_v,init_v)
        self.linear2.bias.data.uniform_(-init_v,init_v)
        self.max_action = max_action
    def forward(self,x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.tanh(x)
        return x * self.max_action
        # discrete
        '''
        
        
        probs = F.softmax(x)
        dist = Categorical(probs)
        return dist
        '''
        

# critic net double q net work
class critic_net(nn.Module):
    def __init__(self, input_dim:int, hidden_dim = 128) -> None:
        super(critic_net,self).__init__()
        #q1
        self.linear1 = nn.Linear(input_dim,hidden_dim)
        self.linear2 = nn.Linear(hidden_dim,hidden_dim)
        self.linear3 = nn.Linear(hidden_dim,1)

        self.linear4 = nn.Linear(input_dim,hidden_dim)
        self.linear5 = nn.Linear(hidden_dim,hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)


    def forward(self,state, action):

        input = torch.cat([action,state],1)
        x1 = F.relu(self.linear1(input))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(input))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)
        return x1, x2
    
    def Q1(self,state,action):
        input = torch.cat([action,state],1)
        x1 = F.relu(self.linear1(input))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)
        return x1
    
# buffer
class replay_buffer:
    def __init__(self,maxlen:int) -> None:
        self.memory = deque(maxlen=maxlen)
        self.maxlen = maxlen
    def sample(self,batch_size:int):
        if len(self.memory) <= batch_size:
            return None
        batch = random.sample(self.memory,batch_size)
        return zip(*batch)
    def append(self,state,action,next_state,reward,done):
        if len(self.memory) >= self.maxlen:
            self.memory.popleft()
        self.memory.append((state,action,next_state,reward,done))

class DDPG_net:
    def __init__(self,input_dim:int, output_dim:int,max_action,init_v = 1e-5) -> None:
        # hyper-parameter
        self.gamma = 0.01 ** (1/10)
        self.critic_lr = 0.001
        self.actor_lr = 0.001
        self.batch_size = 2
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.memory = replay_buffer(20000)


        #net work
        self.critic = critic_net(input_dim + 1).to(device=self.device)
        self.critic_target = critic_net(input_dim +  1).to(device=self.device)
        self.actor = actor_net(input_dim,output_dim,max_action).to(device=self.device)
        self.actor_target = actor_net(input_dim,output_dim,max_action).to(device=self.device)

        self.critic_optimizer = optim.Adam(self.critic.parameters(),lr = self.critic_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(),lr= self.actor_lr)

    

    def predict_action(self,state):
        state = torch.tensor(state,dtype=torch.float32,device=self.device)
        action = self.actor_target(state)
        return action.detach().cpu().numpy()
    '''
    def predict_action(self,state):
        state = torch.tensor(state,dtype=torch.float32,device=self.device)
        dist = self.actor(state)
        action = dist.sample()
        return action
    '''
    def update(self):
        if len(self.memory.memory) <= self.batch_size:
            return 
        ## turn batch data into tensor
        states, actions, next_state, rewards ,dones= self.memory.sample(self.batch_size)
        actions = torch.tensor(np.array(actions), dtype=torch.float32,device=self.device)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float32,device=self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32,device=self.device).unsqueeze(1)
        states = torch.tensor(np.array(states), dtype=torch.float32,device=self.device)
        dones = torch.tensor(np.array(dones), device=self.device).unsqueeze(1)

        # actor loss = critic.mean()
        actor_loss = self.critic.Q1(states,self.actor(states))
        actor_loss = -actor_loss.mean()

        # the critic loss
        next_action = self.actor_target(states)
        Q1_target_value,Q2_target_value = self.critic_target(next_state,next_action)
        target_value = torch.min(Q1_target_value,Q2_target_value)
        target_value = rewards + self.gamma* target_value * (~ dones)
        Q1_value, Q2_value = self.critic(states,actions)


        critic_loss = nn.MSELoss()(Q1_value,target_value) + nn.MSELoss()(Q2_value,target_value)

        #calculate gradient
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
    
    def update_model(self):
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

def train():
    env = gym.make('Pendulum-v1')
    n_state = env.observation_space.shape[0]
    n_action = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    agents = DDPG_net(n_state,n_action,max_action)
    max_step = 200
    n_episode = 300
    ep_reward = []
    for episode in range (n_episode):
        state = env.reset()
        step = 0
        done = False
        total_reward = 0
        # start episode
        while not done and step < max_step:
            action = agents.predict_action(state)
            next_state, reward, done,_ = env.step(action)
            agents.memory.append(state,action,next_state,reward,done)
            state = next_state
            total_reward += reward
            agents.update()
            step += 1
        ep_reward.append(total_reward)
        if episode % 20 == 0:
            print(f'episode: {episode},  reward:{total_reward}')
            agents.update_model()
    plt.plot(ep_reward)
    plt.show()
    eval(agents)

def eval(agents: DDPG_net):
    env = gym.make('Pendulum-v1')
    max_step =  10000
    done = False
    total_reward = 0
    step = 0
    state = env.reset()
    while not done and step < max_step:
        action = agents.predict_action(state)
        next_state, reward, done, _ = env.step(action)
        agents.memory.append(state,action,next_state,reward,done)
        state = next_state
        total_reward += reward
        agents.update()
        step += 1
    print(f' reward:{total_reward}')


if __name__ == '__main__':
    train()

        









