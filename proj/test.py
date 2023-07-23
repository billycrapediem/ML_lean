import numpy as np
import matplotlib.pyplot as plt

data_state = np.load('state.npy')
data_vel = np.load('vel.npy')
data_Ft = np.load('Ft.npy')

for i in range(6):
    print(f'variable type:{i+1} mean:{data_Ft[i,:].mean()} std:{data_Ft[i,:].std()}')

'''

for i in range(6):
    ax = axes[i]
    print(data_Ft[i,:].shape)
    count, bins, ignored = ax.hist(data_Ft[i,:], bins=30)
    ax.grid(True)
    ax.set_title(f'data set{i+1}')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
plt.tight_layout()
plt.savefig('analysis.png')
'''