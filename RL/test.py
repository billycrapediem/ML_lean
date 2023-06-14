from D3QN import DQNAgent
from case3d_env import chase3D
import torch
import matplotlib.pyplot as plt
env = chase3D()
agent = DQNAgent(env.state_size,env.action_size)
state_dict = torch.load('net_params.pt')
agent.target_model.load_state_dict(state_dict)


### start testing
rewards = []
for i in range (100):
    state = env.reset()
    done = False
    cap = False
    num = 0
    total = 0

    while not done and not cap:
        action = agent.act(state,True)
        next_state, reward, caputred,done = env.step(action,True)
        state = next_state
        if cap:
            reward = 2000
        total += reward
        num += 1
    rewards.append[total]

plt.plot(rewards)
plt.xlabel('episode')
plt.ylabel('Score')
plt.title('testing Scores')
plt.show()