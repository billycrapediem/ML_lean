import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
from case3d_env import chase3D
import math

# Define the Q-Network architecture
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size,hidden_dim = 256):
        super(QNetwork, self).__init__()
        
        ### hidden layer
        self.hidden_layer = nn.Sequential(
            nn.Linear(state_size,hidden_dim),
            nn.ReLU()
        )
        ### value layer
        self.value_layer = nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1)
        )
        ## action layer
        self.action_layer = nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,action_size)
        )

    def forward(self, x):
        x = self.hidden_layer(x)
        value_x = self.value_layer(x)
        action_x = self.action_layer(x)
        return value_x + action_x - action_x.mean()


# Define the DQN agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.time = 101
        self.max_score = -math.inf
        self.state_size = state_size
        self.action_size = action_size
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory = deque(maxlen=100000)
        self.batch_size = 2048
        self.gamma = 0.1 ** (1/11)  # discount factor
        self.epsilon = 0.97  # exploration rate
        self.prev_espsilon = 0.95
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.0001
        self.model = QNetwork(state_size, action_size).to(device=self.device)  # train model
        self.target_model = QNetwork(state_size, action_size).to(device=self.device) # act model
        self.ans_model =  QNetwork(state_size, action_size).to(device=self.device) # final model
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
    # replay buffer
    def remember(self, state, action, reward, next_state, done):
        if len(self.memory) >= 100000:
            self.memory.pop()
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, eval = False):
        if not eval and random.random() < self.epsilon :
            return random.randint(0, self.action_size - 1)
        else:
            state = torch.from_numpy(state).float().unsqueeze(0).to(device=self.device)
            with torch.no_grad():
                q_values = self.target_model(state)
            return q_values.max(1)[1].item()

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        # init the factors 
        states = torch.tensor(np.vstack(states),device=self.device).float()
        actions = torch.tensor(actions,device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards,device=self.device).unsqueeze(1)
        next_states = torch.tensor(np.vstack(next_states),device=self.device).float()
        dones = torch.tensor(dones,device=self.device).unsqueeze(1)

        model_action = self.model(next_states).max(1)[1]
        model_action = model_action.view(-1,actions.shape[1])

        # double DQN
        Q_targets_next = self.target_model(next_states).gather(1,model_action).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + self.gamma * Q_targets_next * (~dones)

        Q_expected = self.model(states).gather(1, actions)

        # train
        loss = F.mse_loss(Q_expected, Q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    def test_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            q_values = self.ans_model(state)
        return q_values.max(1)[1].item()


def eval(env: chase3D, agent:DQNAgent):
    # test for the model
    state = env.reset()
    done = False
    caputred = False
    model_reward = 0
    rewards = []
    time = 0
    while  not caputred:
        action = agent.act(state, True)
        next_state, reward, caputred = env.step(action)
        state = next_state
        rewards.append(reward)
        if reward == 200:
            print("capturedddddd")
        model_reward += reward
        time += 1
        

    if model_reward > agent.max_score or (caputred and time < agent.time):
        print(f'eval:   total reward:{model_reward}')
        agent.ans_model.load_state_dict(agent.target_model.state_dict())
        agent.max_score = model_reward
        agent.prev_espsilon = agent.epsilon
        agent.time = time
        
    else:
        agent.epsilon = agent.prev_espsilon
        agent.model.load_state_dict(agent.ans_model.state_dict())
    return model_reward

if __name__ == "__main__":
    ## main function
    env = chase3D()
    state_size = env.state_size
    action_size = env.action_size
    agent = DQNAgent(state_size, action_size)
    update_frequency = 5
    eval_frequency = 20
    
    # Training parameters
    num_episodes = 8000
    scores = []  # List to store the scores
    for episode in range (num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        while  not done :
            # visualize the interaction
            #env.render()
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            # Modify the reward to encourage longer pole balancing
            reward = float(reward)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
                
        agent.train()
        if episode % update_frequency == 0:
            print(f'episode : {episode}  epslion:{agent.epsilon}  total score: {total_reward}')
            agent.update_target_model()
        if episode % eval_frequency ==0 and episode != 0:
            r = eval(env,agent)
            scores.append(r)
    plt.plot(scores)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Training Scores')
    plt.savefig('D3QN.png')
    

### start testing
    rewards = []
    for i in range (50):
        rewards.append(eval(env,agent))
    plt.plot(rewards)
    plt.xlabel('episode')
    plt.ylabel('Score')
    plt.title('testing Scores')
    plt.savefig('test_D3QN.png')
    
    
    
    


