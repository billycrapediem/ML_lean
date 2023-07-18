
import torch

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np


from muti_chase import chase3D
from ppo_discrete import PPO_discrete, SharedActor,SharedCritic, MlpLstmExtractor
from ppo_discrete import SharedActor
from arguments import ppo_args

args = ppo_args
actor_hidden_state = torch.zeros(size=(3,args.num_layers, 1, args.rnn_hidden_dim), dtype=torch.float32)

def update_map(frame, agent_scatters,target_scatter, env, agent,state):
    episode_reward = 0
    actions = []
    state = torch.as_tensor(np.array(state), dtype=torch.float32)
    for id in range(16): 
        a, a_logprob, actor_hidden_state[id] = agent.choose_action(state[id].unsqueeze(0).unsqueeze(0), actor_hidden_state[id], False)
        actions.append(a)
    next_state, r, dones = env.step(actions)  # Take a step
    episode_reward += np.sum(np.array(r))
    state = next_state
    cnt = 0
    t_pos = env.target.cur_point
    target_scatter._offsets3d = ([t_pos[0]],[t_pos[1]],[t_pos[2]])
    for pursuer in env.agent:
        pos = np.array(pursuer.cur_point)
        print(pos.shape)
        agent_scatters[cnt]._offsets3d = ([pos[0]],[pos[1]],[pos[2]])
        cnt += 1
    print(f'target_pos:{t_pos}')
    return target_scatter, *agent_scatters

env = chase3D()
args.episode_limit = env.time_limits
args.state_dim = env.state_size
args.action_dim = 1
args.discrete_action_dim = env.action_size



agent = torch.load('actor.pth')



fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')

state = env.reset()
map_ob_x = np.array(env.map.obstacles_x)
map_ob_y = np.array(env.map.obstacles_y)
map_ob_z = np.array(env.map.obstacles_z)

scatter = ax.scatter(map_ob_x,map_ob_y,map_ob_z, c='r', marker='o')
target_point = env.target.cur_point
target_scatter = ax.scatter([target_point[0]],[target_point[1]],[target_point[2]],c='m',marker='s')
num_agent = 15
agent_scatters = []
for chaser in env.agent:
    pos = chaser.cur_point
    agent_scatter = ax.scatter([pos[0]],[pos[1]],[pos[2]],c='b',marker='D')
    agent_scatters.append(agent_scatter)



# An episode is over, store obs_n, s and avail_a_n in the last step

animation = FuncAnimation(fig, update_map, frames=150, fargs=(agent_scatters,target_scatter,env,agent,state),interval=2000)
plt.show()
