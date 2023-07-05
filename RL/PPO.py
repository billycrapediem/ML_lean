from typing import Any
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import deque
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from muti_chase import chase3D


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def orthogonal_init(layer,gain=1.0):
    nn.init.orthogonal_(layer.weight,gain=gain)
    nn.init.constant_(layer.bias,0)

class actor(nn.Module):
    def __init__(self,input_dim, ouput_dim,mid_dim = 128) -> None:
        super(actor,self).__init__()
        self.linear1 = nn.Linear(input_dim,mid_dim)
        self.linear2 = nn.Linear(mid_dim,mid_dim)
        self.linear3 = nn.Linear(mid_dim,ouput_dim)
        self.active = nn.Tanh()
        orthogonal_init(self.linear1)
        orthogonal_init(self.linear2)
        orthogonal_init(self.linear3)
    def forward(self,x):
        x = self.active(self.linear1(x))
        x = self.active(self.linear2(x))
        x = self.linear3(x)
        probs = torch.softmax(x,dim=1)
        return probs

class critic(nn.Module):
    def __init__(self,input_dim, ouput_dim,mid_dim = 128) -> None:
        super(critic,self).__init__()
        self.linear1 = nn.Linear(input_dim,mid_dim)
        self.linear2 = nn.Linear(mid_dim,mid_dim)
        self.linear3 = nn.Linear(mid_dim,ouput_dim)
        self.active = nn.Tanh()
        orthogonal_init(self.linear1)
        orthogonal_init(self.linear2)
        orthogonal_init(self.linear3)
    def forward(self,x):
        x = self.active(self.linear1(x))
        x = self.active(self.linear2(x))
        x = self.linear3(x)
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
                 max_train_steps,
                 gamma = 0.99,
                 update_freq = 2**12,
                 mini_batch = 1024,
                 epoch_num = 6,
                 learning_rate = 1e-4,
                 eplison_max = 0.1,
                 lamda = 0.96,
                 entropy_coef = 0.01):
        self.actor = actor(state_space,action_space).to(device)
        self.critic = critic(state_space,1).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate,eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr=learning_rate,eps=1e-5)
        self.buffer = replay_buffer(2**13)
        self.gamma = gamma
        self.batch_size = update_freq
        self.eplison_max = eplison_max
        self.epoch_num = epoch_num
        self.step = 0
        self.entropy_coef = entropy_coef
        self.lamda = lamda
        self.lr = learning_rate
        self.max_train_steps = max_train_steps
        self.mini_batch = mini_batch
    def sample_action(self, state):
        state = torch.tensor(np.array(state), dtype=torch.float32,device=device).unsqueeze(dim=0)
        probs = self.actor(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        self.step += 1
        return action.squeeze().cpu().numpy(), dist.log_prob(action).detach()
    def evaluate(self, s):  # When evaluating the policy, we select the action with the highest probability
        s = torch.unsqueeze(torch.tensor(np.array(s), dtype=torch.float,device=device), 0)
        a_prob = self.actor(s).squeeze().detach().cpu().numpy()
        action = []
        for probs in a_prob:
            a = np.argmax(probs)
            action.append(a)
        return action
    def update(self):
        if not (self.step % self.batch_size == 0):
            return
        # init
        old_states,next_state, old_actions, rewards,old_log_probs,  dones = self.buffer.sample()
        old_states = torch.tensor(np.array(old_states), dtype=torch.float32, device=device)
        old_log_probs = torch.tensor(old_log_probs, device=device, dtype=torch.float32)
        old_actions = torch.tensor(np.array(old_actions),dtype=torch.float32,device=device)
        dones = torch.tensor(np.array(dones),dtype=torch.bool,device=device)
        next_state = torch.tensor(np.array(next_state),dtype=torch.float32,device=device)
        rewards = torch.tensor(np.array(rewards),dtype=torch.float32,device=device)
        
        # calculate estimate reward
        returns = []
        gae = 0
        with torch.no_grad():
            values_state = self.critic(old_states)
            values_next_state = self.critic(next_state)
            deltas = rewards + self.gamma * values_next_state - values_state
            for delta, done in zip(reversed(deltas.flatten().cpu().numpy()), reversed(dones.flatten().cpu().numpy())):
                gae = delta + self.gamma * self.lamda * gae * (~done)
                returns.insert(0,gae)
            returns = torch.tensor(np.array(returns),dtype=torch.float32,device=device).view(-1,1)
            value_target = returns + values_state
            #normalize
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        for _ in range(self.epoch_num):
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)),self.mini_batch,False):
                # get action probabilities
                dist = torch.distributions.Categorical(probs=self.actor(old_states[index]))
                dist_entropy = dist.entropy().view(-1, 1)
                # get new action probabilities
                new_probs= dist.log_prob(old_actions[index]).view(-1, 1) 
                # compute ratio (pi_theta / pi_theta__old):
                ratio = torch.exp(new_probs - old_log_probs[index])
                # compute surrogate loss
                surr1 = ratio * returns[index]
                surr2 = torch.clamp(ratio, 1 - self.eplison_max, 1 + self.eplison_max) * returns[index]
                # compute actor loss
                actor_loss = (-torch.min(surr1, surr2) - self.entropy_coef * dist_entropy).mean()
                # compute critic loss
                values = self.critic(old_states[index]) # detach to avoid backprop through the critic
                critic_loss = F.mse_loss(value_target[index],values)
                # take gradient step & gradian clip
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(),0.5)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(),0.5)
                self.critic_optimizer.step()
        lr_now = self.lr * (1-self.step/self.max_train_steps)
        for p in self.actor_optimizer.param_groups:
            p['lr'] = lr_now
        for p in self.critic_optimizer.param_groups:
            p['lr'] = lr_now
        self.buffer.clear()    
class DynamicMeanSTD:
    def __init__(self,shape) -> None:
        self.count = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)
    def update(self,x):
        x = np.array(x)
        self.count += 1
        if self.count == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.count
            self.S = self.S + (x - old_mean) / (x - self.mean)
            self.std = np.sqrt(self.S / self.count)
class RewardScaling:
    def __init__(self,shape,gamma) -> None:
        self.shape = shape
        self.gamma = gamma
        self.running_ms = DynamicMeanSTD(shape)
        self.reward = np.zeros(self.shape)
    def __call__(self, x) -> Any:
        self.reward = self.gamma * self.reward + x
        self.running_ms.update(self.reward)
        x = x / (self.running_ms.std + 1e-8)
        return x
    def reset(self):
        self.reward = np.zeros(self.shape)
    
def evaluate_policy( env, agent):
    times = 5
    evaluate_reward = 0
    cnt = 0
    for _ in range(times):
        s = env.reset()

        done = False
        episode_reward = 0
        while not done:
            a = agent.evaluate(s)  # We use the deterministic policy during the evaluating
            s_, r, dones= env.step(a)
            done = dones[0]
            episode_reward += r[0]
            for i in range(5):
                if r[i] == 20000:
                    cnt += 1
            s = s_
        evaluate_reward += episode_reward

    return evaluate_reward / times, cnt / times

def train():
    env = chase3D()
    state_size = env.state_size
    action_size = env.action_size
    train_steps = int(100 * 2000)
    agent = PPO(action_size,state_size,max_train_steps=train_steps)
    episode_reward = []
    step = 0
    episode = 0
    train_scale = []
    reward_scaling = RewardScaling(1,0.99)
    while step < train_steps:
        state = env.reset()
        done = False
        reward_scaling.reset()
        while not done:
            actions,logprob = agent.sample_action(state)
            next_state, reward, dones= env.step(actions)
            done = dones[0]
            for i in range (5):
                reward[i] = reward_scaling(reward[i])
                agent.buffer.append((state[i],next_state[i],actions[i],reward[i],logprob[0][i],dones[i]))
            agent.update()
            state = next_state
            step += 1
        episode += 1
        if episode % 5 == 0:
            eps_reward,captured = evaluate_policy(env,agent)
            train_scale.append(eps_reward)
            print(f'episode:{episode} step:{step} reward:{eps_reward} captured:{captured}')
            '''

            if eps_reward > 2000 or max_reward <= eps_reward:
                save_agent.actor.load_state_dict(agent.actor.state_dict())
                save_agent.critic.load_state_dict(agent.critic.state_dict())
                max_reward = eps_reward
                print("update")
            else:
                agent.actor.load_state_dict(save_agent.actor.state_dict())
                agent.critic.load_state_dict(save_agent.critic.state_dict())
                '''
    torch.save(agent.actor.state_dict(),"PPO_agent_model.pt")
    torch.save(agent.critic.state_dict(),"PPO_critic_model.pt")
    #captured = []
    for _ in range (40):
        r,_ = evaluate_policy(env,agent)
        episode_reward.append(r)
        #captured.append(c)
    plt.plot(train_scale)
    plt.plot(episode_reward)
    plt.savefig('chase.png')
train()





