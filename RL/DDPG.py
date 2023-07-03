import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
import random
import gym



# actor net
class actor_net(nn.Module):
    def __init__(self, input_dim:int, output_dim:int, hidden_dim = 128,init_v = 1e-5) -> None:
        super(actor_net,self).__init__()
        self.linear1 = nn.Linear(input_dim,hidden_dim)
        self.linear2 = nn.Linear(hidden_dim,output_dim)
        self.linear2.weight.data.uniform_(-init_v,init_v)
        self.linear2.bias.data.uniform_(-init_v,init_v)
    def forward(self,x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        '''
        x = F.tanh(x)
        return x * self.max_action
        '''
        # discrete
        probs = F.softmax(x)
        dist = Categorical(probs)
        return dist
        
        

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
        
        action = torch.unsqueeze(action,1)
        input = torch.cat([action,state],1)
        
        x1 = F.relu(self.linear1(input))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(input))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)
        return x1, x2
    
    def Q1(self,state,action):
        action = torch.unsqueeze(action,1)
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
    def __init__(self,input_dim:int, output_dim:int,policy_freq = 2, tau =1e-2) -> None:
        # hyper-parameter
        self.gamma = 0.01 ** (1/30)
        self.critic_lr = 1e-3
        self.actor_lr = 1e-2
        self.batch_size =  512
        self.tau = tau
        self.policy_freq = policy_freq
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cur_step = 1
        self.memory = replay_buffer(20000)


        #net work
        self.critic = critic_net(input_dim + 1).to(device=self.device)
        self.critic_target = critic_net(input_dim +1).to(device=self.device)
        self.actor = actor_net(input_dim,output_dim).to(device=self.device)
        self.actor_target = actor_net(input_dim,output_dim).to(device=self.device)

        self.critic_optimizer = optim.Adam(self.critic.parameters(),lr = self.critic_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(),lr= self.actor_lr)

    

    def predict_action(self,state):
        state = torch.tensor(state,dtype=torch.float32,device=self.device)
        dist = self.actor_target(state)
        action = dist.sample()
        return action.detach().cpu().numpy()

    def update(self):
        if len(self.memory.memory) <= self.batch_size:
            return 
        ## turn batch data into tensor
        
        states, actions, next_state, rewards ,dones= self.memory.sample(self.batch_size)
        actions = torch.tensor(np.array(actions), dtype=torch.float32,device=self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32,device=self.device).unsqueeze(1)
        states = torch.tensor(np.array(states), dtype=torch.float32,device=self.device)
        dones = torch.tensor(np.array(dones), device=self.device).unsqueeze(1)

        with torch.no_grad():
            next_action = self.predict_action(next_state)
            next_action = torch.tensor(np.array(next_action),device=self.device)
            next_state = torch.tensor(np.array(next_state),device=self.device)
            Q1_target_value,Q2_target_value = self.critic_target(next_state,next_action)
            target_value = torch.min(Q1_target_value,Q2_target_value)
            target_value = rewards + self.gamma* target_value * (~ dones)

        Q1_value, Q2_value = self.critic(states,actions)
        critic_loss = nn.MSELoss()(Q1_value,target_value) + nn.MSELoss()(Q2_value,target_value)

        #calculate gradient
       
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # actor loss = critic.mean()
        if self.cur_step % self.policy_freq == 0:
            actor_loss = self.critic.Q1(states,torch.tensor(np.array(self.predict_action(states)),device=self.device))
            actor_loss = -actor_loss.mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        self.cur_step += 1

class OUNoise(object):
    '''Ornstein–Uhlenbeck噪声
    '''
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu           = mu # OU噪声的参数
        self.theta        = theta # OU噪声的参数
        self.sigma        = max_sigma # OU噪声的参数
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.n_actions   = action_space.shape[0]
        self.low          = action_space.low
        self.high         = action_space.high
        self.reset()
    def reset(self):
        self.obs = np.ones(self.n_actions) * self.mu
    def evolve_obs(self):
        x  = self.obs
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.n_actions)
        self.obs = x + dx
        return self.obs
    def get_action(self, action, t=0):
        ou_obs = self.evolve_obs()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period) # sigma会逐渐衰减
        return np.clip(action + ou_obs, self.low, self.high) # 动作加上噪声后进行剪切
    


def train():
    env = gym.make('CartPole-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agents = DDPG_net(state_size,action_size)
    max_step = 2000
    n_episode = 200
    
    ep_reward = []
    eval_reward = []
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
            #eval_reward.append(eval(agents))
    plt.plot(ep_reward)
    #plt.plot(eval_reward)
    plt.savefig('train_result.png')
    

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
    print(f'eval: reward:{total_reward}')
    return total_reward


if __name__ == '__main__':
    train()

        









