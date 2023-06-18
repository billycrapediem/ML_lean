import torch
#from case3d_env import chase3D
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import torch.nn as nn
from collections import deque
import gym
import matplotlib.pyplot as plt
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
    
class A3C:
  def __init__(self, state_size:int, action_size:int) -> None:
      self.state_size = state_size
      self.action_size = action_size
      #hyper-parameters
      self.batch_size = 512
      self.gamma = 0.1 ** (1/10) 
      self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      #model
      self.model = ActorCritic(state_size,action_size,128).to(self.device)
      self.optimizer = optim.Adam(self.model.parameters(),lr = 1e-3)
      #replay buffer
      #self.buffer_size = 10000
      #self.buffer = deque(maxlen=self.buffer_size)
  '''



  def remember(self,state,next_state,action,reward,done):
      if len(self.buffer) >= self.buffer_size:
          self.buffer.popleft()
      self.buffer.append(state,next_state,action,reward,done)
  '''
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
        if vis: env.render()
        total_reward += reward
    return total_reward

def train():
  update_freq = 7
  env = gym.make('CartPole-v0')
  n_states = env.observation_space.shape[0]
  n_actions = env.action_space.n
  agents = A3C(n_states,n_actions,)
  n_episode = 20000
  # start training
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
  print('finish training')
  return test_rewards, test_ma_rewards

test_rewards, test_ma_rewards = train()
plt.plot(test_rewards)
plt.plot(test_ma_rewards)
plt.show()






            
        






       
     


