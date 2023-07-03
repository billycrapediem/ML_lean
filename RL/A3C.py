import torch
#from case3d_env import chase3D
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import torch.nn as nn
from collections import deque
import torch.multiprocessing as mp
import gym
import matplotlib.pyplot as plt
import random
class ActorCritic(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim):
        super(ActorCritic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=1),
        )
        
    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        dist  = Categorical(probs)
        return dist, value
    


class ReplayBuffer:
    def __init__(self,buffer_size) -> None:
        self.memory = deque(maxlen=buffer_size)
        self.max_len = buffer_size
    def sample_batch(self,size):
        if len(self.memory) < size:
           return None
        batch = random.sample(self.memory,size)
        return zip(*batch)
    def remember(self,probs,advantages,entropy):
        if len(self.memory) >= self.max_len:
            self.memory.popleft()
        self.memory.append(probs,advantages,entropy)

class learner:
    def __init__(self, state_size:int, action_size:int) -> None:
        self.state_size = state_size
        self.action_size = action_size
        #hyper-parameters
        self.batch_size = 128
        self.gamma = 0.1 ** (1/10) 
        self.device  = torch.device("cpu")
        #model
        self.model = ActorCritic(state_size,action_size,128).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(),lr = 1e-3)
        #replay buffer
        #self.buffer_size = 10000
        #self.buffer = deque(maxlen=self.buffer_size)

    def update(self,replay_buffer:ReplayBuffer):  
        probs,advantages,entropy = replay_buffer.sample_batch(self.batch_size)      
        actor_loss = -(probs * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()
        loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def compute_advantage(self,next_value,value,reward,dones):
        R = next_value
        Q_target = []
        for step in reversed(range(len(reward))):
           R = reward[step] + self.gamma * R * dones[step]
           Q_target.insert(0,R)
        advantages = torch.cat(Q_target) - value
        return advantages




def eval(env,agent,vis=False):
    state = env.reset()
    if vis: env.render()
    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
        dist, _ = agent.model(state)
        next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
        state = next_state
        total_reward += reward
    return total_reward

def train(agents:learner):
  n_episode = 1000
  # start training
  env = gym.make('CartPole-v0')
  test_rewards = []
  test_ma_rewards = []
  for episode in range (n_episode):
     state = env.reset()
     done = False
     entropy = 0
     probs = []
     values = []
     rewards = []
     dones = []
     while not done:
        state = torch.from_numpy(state).float().unsqueeze(0).to(device=agents.device)
        dist, value = agents.model(state)
        action = dist.sample()
        #choose action
        #interaction
        next_state, reward, done, _ = env.step(action.cpu().numpy()[0])
        log_prob = dist.log_prob(action)
        probs.append(log_prob)
        entropy += dist.entropy().mean()
        values.append(value)
        if not done:
           reward = -10
        reward = torch.tensor([reward]).unsqueeze(1).to(device=agents.device)
        rewards.append(reward)
        dones.append(torch.tensor([~done]).unsqueeze(1).to(device=agents.device))
        state = next_state
     if episode % 50 == 0:
         test_reward = np.mean([eval(env,agents) for _ in range(10)])
         print(f"episode:{episode}, test_reward:{test_reward}")
         test_rewards.append(test_reward)
         if test_ma_rewards:
             test_ma_rewards.append(0.9*test_ma_rewards[-1]+0.1*test_reward)
         else:
             test_ma_rewards.append(test_reward)
     next_state = torch.from_numpy(next_state).float().unsqueeze(0).to(device=agents.device)
     _,next_value = agents.model(next_state)
     probs = torch.cat(probs)
     values = torch.cat(values)
     advantages = agents.compute_advantage(next_value,values,rewards,dones)
     actor_loss = -(probs * advantages.detach()).mean()
     critic_loss = advantages.pow(2).mean()
     loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy
     agents.optimizer.zero_grad()
     loss.backward()
     agents.optimizer.step()

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agents = learner(n_states,n_actions)
    torch.manual_seed(1)
    mp.set_start_method('spawn', force=True)
    agents.model.share_memory()
    processes = []
    for rank in range(8):
        p = mp.Process(target=train, args = (agents,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    print(f"test_reward:{eval(env,agents)}")








            
        






       
     


