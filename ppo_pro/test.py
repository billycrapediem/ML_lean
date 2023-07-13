import numpy as np
obstacles = []
for _ in range(int(100)):
    ob = np.random.normal(100,40,3)
    print(ob[0],ob[1],ob[2])
    obstacles.append(ob)
x = [[1,2,3],[1,2,3]]
x = np.array(x)
obstacles = np.array(obstacles)
x = np.concatenate((x,obstacles))
print(x)