
import torch

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np
from D3QN import DQNAgent


from muti_chase import chase3D

def update_map(frame, agent_scatters,target_scatter, env, agent,state):
    episode_reward = 0
    state = torch.as_tensor(np.array(state), dtype=torch.float32)
    action = agent.act(state)
    next_state, r, dones = env.step(action)  # Take a step
    episode_reward += np.sum(np.array(r))
    state = next_state
    t_pos = env.target.cur_point
    target_scatter._offsets3d = ([t_pos[0]],[t_pos[1]],[t_pos[2]])
    pos = np.array(env.agent.cur_point)
    agent_scatter._offsets3d = ([pos[0]],[pos[1]],[pos[2]])
    print(f'target_pos:{t_pos}')
    return target_scatter, *agent_scatters


env = chase3D()
agent = DQNAgent(env.state_size,env.action_size)
agent.target_model = torch.load('D3QN.pth')



fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')

state = env.reset()
map_ob_x = np.array(env.map.obstacles_x)
map_ob_y = np.array(env.map.obstacles_y)
map_ob_z = np.array(env.map.obstacles_z)

scatter = ax.scatter(map_ob_x,map_ob_y,map_ob_z, c='r', marker='o')
target_point = env.target.cur_point
target_scatter = ax.scatter([target_point[0]],[target_point[1]],[target_point[2]],c='m',marker='s')
num_agent = 3
agent_scatters = []
pos = env.agent.cur_point
agent_scatter = ax.scatter([pos[0]],[pos[1]],[pos[2]],c='b',marker='D')


# An episode is over, store obs_n, s and avail_a_n in the last step

animation = FuncAnimation(fig, update_map, frames=150, fargs=(agent_scatter,target_scatter,env,agent,state),interval=2000)
