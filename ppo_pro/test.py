
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



def update(frame, scatter, map):
    scatter._offsets3d = map
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Map Animation')
    
    # 返回更新后的图形对象
    return scatter,
args = ppo_args
env = chase3D()
args.episode_limit = env.time_limits
args.state_dim = env.state_size
args.action_dim = 1
args.discrete_action_dim = env.action_size



agent = torch.load('model/actor.pth',map_location='cpu')



fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')

state = env.reset()
map_ob_x = np.array(env.map.obstacles_x)
map_ob_y = np.array(env.map.obstacles_y)
map_ob_z = np.array(env.map.obstacles_z)

scatter = ax.scatter(map_ob_x,map_ob_y,map_ob_z, c='r', marker='o')
num_agent = 3
agent_scatters = []
for chaser in env.agent:
    pos = chaser.cur_point
    agent_scatter = ax.scatter(pos[0],pos[1],pos[2],c='b',marker='D')
    agent_scatters.append(agent_scatter)
actor_hidden_state = torch.zeros(size=(3,args.num_layers, 1, args.rnn_hidden_dim), dtype=torch.float32)
def update_map(frame,scatter, agent_scatters, env, agent):
    episode_reward = 0
    actions = []
    state = torch.as_tensor(np.array(state), dtype=torch.float32)
    for id in range(agent.agent_num): 
        a, a_logprob, actor_hidden_state[id] = agent.actor.choose_action(state[id].unsqueeze(0).unsqueeze(0), actor_hidden_state[id], False)
        actions.append(a)
    next_state, r, dones = env.step(actions)  # Take a step
    episode_reward += np.sum(np.array(r))
    state = next_state
    cnt = 0
    for pursuer in env.agnet:
        pos = pursuer.cur_point
        agent_scatters[cnt]._offsets3d = ([pos[0]],pos[1],pos[2])
        cnt += 1
    return scatter, *agent_scatters

# An episode is over, store obs_n, s and avail_a_n in the last step

animation = FuncAnimation(fig, update, frames=150, fargs=(scatter,agent_scatters,env,agent),interval=2000)

animation.save('map_animation.gif', writer='matplotlib')
