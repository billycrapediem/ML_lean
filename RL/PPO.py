import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import deque
import copy
import gym


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class actor(nn.Module):
    def __init__(self,input_dim, ouput_dim,mid_dim = 128) -> None:
        super(actor,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim,mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim,ouput_dim),
            nn.ReLU()
        )
    def forward(self,x):
        x = self.net(x)
        probs = F.softmax(x,dim=1)
        return probs

class critic(nn.Module):
    def __init__(self,input_dim, ouput_dim,mid_dim = 128) -> None:
        super(critic,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim,mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim,ouput_dim),
            nn.ReLU()
        )
    def forward(self,x):
        x = self.net(x)
        return x
class replay_buffer:
    def __init__(self,maxlen:int) -> None:
        self.memory = deque(maxlen=maxlen)
        self.maxlen = maxlen
    def sample(self):
            batch = list(self.memory)
            return zip(*batch)
    def append(self,trainsition):
        if len(self.memory) >= self.maxlen:
            self.memory.popleft()
        self.memory.append(trainsition)
    def clear(self):
        self.memory.clear()


class PPO:
    def __init__(self,
                 action_space:int, 
                 state_space:int,  
                 gamma = 0.99,
                 update_freq = 512,
                 epoch_num = 4,
                 learning_rate = 1e-3,
                 eplison_max = 0.2,
                 entropy_coef = 0.01):
        self.actor = actor(state_space,action_space).to(device)
        self.critic = critic(state_space,1).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr=learning_rate)
        self.buffer = replay_buffer(100000)
        self.gamma = gamma
        self.update_freq = update_freq
        self.eplison_max = eplison_max
        self.epoch_num = epoch_num
        self.step = 0
        self.entropy_coef = entropy_coef
    def sample_action(self, state):
        state = torch.tensor(state, dtype=torch.float32,device=device).unsqueeze(dim=0)
        probs = self.actor(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        self.step += 1
        return action.item(), dist.log_prob(action).detach()
    def update(self):
        if not (self.step % self.update_freq == 0):
            return
        # sample the action
        old_states,next_state, old_actions, rewards,old_log_probs, old_next_states,  dones = self.buffer.sample()
        old_states = torch.tensor(np.array(old_states), dtype=torch.float32, device=device)
        old_log_probs = torch.tensor(old_log_probs, device=device, dtype=torch.float32)
        old_actions = torch.tensor(np.array(old_actions),dtype=torch.float32,device=device)
        dones = torch.tensor(np.array(dones),dtype=torch.bool,device=device)
        next_state = torch.tensor(np.array(next_state),dtype=torch.float32,device=device)
        rewards = torch.tensor(np.array(rewards),dtype=torch.float32,device=device)
        # calculate estimate reward
        
        returns = self.critic(next_state).detach() * self.gamma + rewards
        #normalize
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        for _ in range(self.epoch_num):
            # compute advantage
            values = self.critic(old_states) # detach to avoid backprop through the critic
            advantage = returns - values.detach()
            # get action probabilities
            probs = self.actor(old_states)
            dist = torch.distributions.Categorical(probs)
            # get new action probabilities
            new_probs = dist.log_prob(old_actions).detach()
            # compute ratio (pi_theta / pi_theta__old):
            ratio = torch.exp(new_probs - old_log_probs) # old_log_probs must be detached
            # compute surrogate loss
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eplison_max, 1 + self.eplison_max) * advantage
            # compute actor loss
            actor_loss = -torch.min(surr1, surr2).mean() + self.entropy_coef * dist.entropy().mean()
            # compute critic loss
            critic_loss = (returns - values).pow(2).mean()
            # take gradient step
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
        self.buffer.clear()    
def train():
    env = gym.make('CartPole-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = PPO(action_size,state_size)
    max_step = 200
    n_episode = 400
    episode_reward = []
    for episode in range (n_episode):
        state = env.reset()
        done = False
        step = 0
        total_reward = 0
        while step < max_step and not done:
            actions,logprob = agent.sample_action(state)
            next_state, reward, done, _ = env.step(actions)
            if done:
                reward = -10
            agent.buffer.append((state,next_state,actions,reward,logprob,done))
            agent.update()
            state = next_state
            total_reward += reward
            step += 1
        episode_reward.append(total_reward)
        if episode % 20 == 0:
            print(f'episode:{episode}  reward:{total_reward}')
            '''
            
            state = env.reset()
            done = False
            total_reward = 0
            step = 0
            while step < max_step:
                actions = agent.sample_action(state)
                next_state, reward, done, _ = env.step(actions)
                state = next_state
                total_reward += reward
                if done:
                    break
                step += 1
            if total_reward> max_reward:
                print("update")
                output_agnet = copy.deepcopy(agent)
                max_reward = total_reward
            else:
                agent = copy.deepcopy(output_agnet)
            '''
    plt.plot(episode_reward)
    plt.savefig('PPO.png')
train()





