import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import gym
import matplotlib.pyplot as plt
from collections import deque

# Define the Q-Network architecture
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size,hidden_dim = 128):
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
        self.state_size = state_size
        self.action_size = action_size
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory = deque(maxlen=10000)
        self.batch_size = 512
        self.gamma = 0.99  # discount factor
        self.epsilon = 0.99  # exploration rate
        self.epsilon_decay = 0.95
        self.epsilon_min = 0.01
        self.model = QNetwork(state_size, action_size).to(device=self.device)
        self.target_model = QNetwork(state_size, action_size).to(device=self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def remember(self, state, action, reward, next_state, done):
        if len(self.memory) == self.memory.maxlen:
            self.memory.pop()
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            state = torch.from_numpy(state).float().unsqueeze(0).to(device=self.device)
            with torch.no_grad():
                q_values = self.target_model(state)
            return q_values.max(1)[1].item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.vstack(states),device=self.device).float()
        actions = torch.tensor(actions,device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards,device=self.device).unsqueeze(1)
        next_states = torch.tensor(np.vstack(next_states),device=self.device).float()
        dones = torch.tensor(dones,device=self.device).unsqueeze(1)

        model_action = self.model(next_states).max(1)[1]
        model_action = model_action.view(-1,actions.shape[1])
        Q_targets_next = self.target_model(next_states).gather(1,model_action).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + self.gamma * Q_targets_next * (~dones)

        Q_expected = self.model(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

# Create the CartPole environment
env = gym.make('CartPole-v0')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
# Create the DQN agent
agent = DQNAgent(state_size, action_size)

# Training parameters
num_episodes = 200
total_reward = 0

scores = []  # List to store the scores
for episode in range (num_episodes):
    for epcho in range( 10 ):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            # visualize the interaction
            #env.render()

            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            # Modify the reward to encourage longer pole balancing
            reward = reward if not done else -10

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        
        agent.replay()
        scores.append(total_reward)
    agent.update_target_model()


    # Print the episode score every 10 episodes
    if episode % 10 == 0:
        print(f"Episode: {episode}, Score: {total_reward}")

# Close the environment
env.close()

# Plot the scores
plt.plot(scores)
plt.xlabel('Episode')
plt.ylabel('Score')
plt.title('Training Scores')
plt.show()