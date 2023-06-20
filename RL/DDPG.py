import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
import gym



# actor net
class actor_net(nn.Module):
    def __init__(self, input_dim:int, output_dim:int, hidden_dim = 128,init_v = 1e-5) -> None:
        super(actor_net,self).__init__()
        self.linear1 = nn.Linear(input_dim,hidden_dim)
        self.linear2 = nn.Linear(hidden_dim,output_dim)
    def forward(self,x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = torch.tanh(x)
        return x

# critic net
class critic_net(nn.Module):
    def __init__(self, input_dim:int, hidden_dim = 128) -> None:
        super(critic_net,self).__init__()
        self.linear1 = nn.Linear(input_dim,hidden_dim)
        self.linear2 = nn.Linear(hidden_dim,1)
    def forward(self,action, state):
        x = torch.cat([state,action],1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        return x
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
    def __init__(self,input_dim:int, output_dim:int) -> None:
        #net work
        self.critic = critic_net(input_dim + output_dim)
        self.critic_target = critic_net(input_dim +  output_dim)
        self.actor = actor_net(input_dim,output_dim)
        self.actor_target = actor_net(input_dim,output_dim)
        
        # hyper-parameter
        self.gamma = 0.01 ** (1/10)
        self.critic_lr = 0.001
        self.actor_lr = 0.001
        self.batch_size = 256
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.memory = replay_buffer(10000)
        self.critic_optimizer = optim.Adam(self.critic.parameters(),lr = self.critic_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(),lr= self.actor_lr)

    
    def predict_action(self,state):
        state = torch.tensor(state,dtype=torch.float32,device=self.device)
        action = self.actor_target(state)
        return action.detach().cpu().numpy()
    
    def update(self):
        if len(self.memory.memory) <= self.batch_size:
            return 
        states, actions, next_state, rewards ,dones= self.memory.sample(self.batch_size)
        actions = torch.tensor(np.array(actions), dtype=torch.float32,device=self.device)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float32,device=self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32,device=self.device).unsqueeze(1)
        states = torch.tensor(np.array(states), dtype=torch.float32,device=self.device)
        dones = torch.tensor(np.array(dones), device=self.device).unsqueeze(1)

        # actor loss = critic.mean()
        actor_loss = self.critic(states,self.actor(states))
        actor_loss = -actor_loss.mean()

        # the critic loss
        target_value = self.critic(states,actions)
        next_action = self.actor_target(next_state)
        expected_value = rewards + self.gamma* next_action * (~ dones)
        critic_loss = nn.MSELoss()(target_value,expected_value)

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
    env = gym.make("Pendulum-v1")
    n_state = env.observation_space.shape[0]
    n_action = env.action_space.shape[0]
    agents = DDPG_net(n_state,n_action)
    max_step = 100000
    n_episode = 300
    ep_reward = []
    for episode in range (n_episode):
        state = env.reset()
        step = 0
        done = False
        total_reward = 0
        while not done and step < max_step:
            action = agents.predict_action(state)
            next_state, reward, done, _ = env.step(action)
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
    env = gym.make("Pendulum-v1")
    max_step =  10000
    done = False
    total_reward = 0
    step = 0
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

        









