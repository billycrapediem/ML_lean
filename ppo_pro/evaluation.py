from ppo_discrete import PPO_discrete, SharedActor,SharedCritic, MlpLstmExtractor
from muti_chase import chase3D
import numpy as np
import torch
from arguments import ppo_args
import matplotlib.pyplot as plt
args = ppo_args
env = chase3D()
agent = torch.load('model/actor.pth')
update_time = [0.1,0.3,0.5, 1, 3, 5,7,9]
success_rate = []
for time in update_time:
    probability = 0
    for _ in range(30):
        state = env.reset(time)
        actor_hidden_state = torch.zeros(size=(8,args.num_layers, 1, args.rnn_hidden_dim), dtype=torch.float32)
        done = False
        while not done:
            actions = []
            state = torch.as_tensor(np.array(state), dtype=torch.float32)
            for id in range(8): 
                a, actor_hidden_state[id] = agent.choose_action(state[id].unsqueeze(0).unsqueeze(0), actor_hidden_state[id], True)
                actions.append(a)
            next_state, r, dones = env.step(actions)  # Take a step
            state = next_state
            for i in range(8):
                if dones[i]:
                    done  = True
                if r[i] == 700:
                    probability += 1
                    break
    success_rate.append( probability / 30)

plt.plot(update_time,success_rate)
plt.grid(True)
plt.xlabel('speed ratio')
plt.ylabel('capture rate')
plt.show()

                
