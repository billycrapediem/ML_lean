import math
import heapq
import random
import numpy as np

OBSTACLE = 1
class ThreeDMap:
    def __init__(self,X,Y,Z):
        self.X = X
        self.Y = Y
        self.Z = Z
        self.map = [[[0 for k in range (Z)]for i in range(X)] for j in range (Y)]
    def add_obstacles(self):
        #create obstacles
        obstacles = []
        self.obstacles_size = self.X * self.Y * self.Z * 0.05
        for _ in range(int(self.obstacles_size)):
            ob = np.random.normal(50,15,3)
            obstacles.append(ob)
        cnt = 0
        #add obstacles into the map
        for ob in obstacles:
            x = round(ob[0])
            y = round(ob[1])
            z = round(ob[2])
            for i in range(3):
                cnt += 1
            if x >= 0 and x < self.X and y >=0 and y < self.Y and z >= 0 and z < self.Z:
                self.map[x][y][z] = OBSTACLE
        
def isCollision(map:ThreeDMap,cur_point):
    if map.X <= cur_point[0] or cur_point[0] < 0:
            return True
    elif map.Y <= cur_point[1] or cur_point[1] < 0:
            return True
    elif map.Z <= cur_point[2]or cur_point[2] < 0:
            return True
    elif map.map[cur_point[0]][cur_point[1]][cur_point[2]] != 0 :
        return True
    return False
class AStar:
    def __init__(self,start_point,end_point,Map:ThreeDMap):
        self.start_point = start_point
        self.end_point = end_point
        self.map = Map
        self.open_list = []
        self.closed_list = []
        self.parent = dict()
        self.g_value = dict()
    
    def heuristics(self, cur_point):
        #Manhattan Distance
        value_f = abs(cur_point[0] - self.end_point[0]) + abs(cur_point[1] - self.end_point[1]) + abs(cur_point[2] - self.end_point[2])
        return value_f
    

    
    def real_distance(self,cur_point, parents):
        cnt = 0
        for i in range(3): 
            cnt = cnt + (abs(cur_point[i] - parents[i])) ** 2
        return math.sqrt( cnt )
    
    def f_value(self,cur_point):
        ans = self.heuristics(cur_point) + self.g_value[cur_point]
        return ans
    
    def get_successor(self, cur_point):
        successors = []
        for i in range (-1,2):
            for j in range (-1,2):
                for k in range(-1,2):
                    if i != 0 or j != 0 or k != 0:
                        if not isCollision(self.map,(cur_point[0]+i,cur_point[1]+j,cur_point[2]+k)): 
                            successors.append((cur_point[0]+i,cur_point[1]+j,cur_point[2]+k))
        
        return successors
    
    def check_end_point(self, cur_point):
        for i in range(3):
            if cur_point[i] != self.end_point[i]:
                return False
        return True
    
    def extract_path(self,cur_point):
        path = []
        while self.parent[cur_point] != cur_point:
            path.append(cur_point)
            cur_point = self.parent[cur_point]
        return path
    
    def search(self):

        if self.start_point == self.end_point:

            return None
        self.parent[self.start_point] = self.start_point
        self.g_value[self.start_point] = 0
        self.g_value[self.end_point] = math.inf

        time = 0
        heapq.heappush(self.open_list,(0,self.start_point))
        while self.open_list and time <= 1000:
            time += 1

            _,cur_point = heapq.heappop(self.open_list)
            self.closed_list.append(cur_point)
            ## sus is the end point
            if self.check_end_point(cur_point):
                break
            successors = self.get_successor(cur_point = cur_point)
            for sus_point in successors:
                ## compute g and h value for each successor
                new_cost = self.g_value[cur_point] + self.real_distance(cur_point,sus_point)
                if sus_point not in self.g_value:
                    self.g_value[sus_point] = math.inf
                if new_cost < self.g_value[sus_point]:
                    self.g_value[sus_point] = new_cost
                    self.parent[sus_point] = cur_point
                    heapq.heappush(self.open_list,(self.f_value(sus_point),sus_point))
        return None
class Agent:
    def __init__(self,start_point,map:ThreeDMap) -> None:
        self.cur_point = start_point
        self.prev_point = []
        for i in range (-1,2):
            for j in range (-1,2):
                for k in range(-1,2):
                    if not isCollision(map,(self.cur_point[0]+i,self.cur_point[1]+j,self.cur_point[2]+k)): 
                        map.map[self.cur_point[0]+i][self.cur_point[1]+j][self.cur_point[2]+k] = 2
                        self.prev_point.append((self.cur_point[0]+i,self.cur_point[1]+j,self.cur_point[2]+k))
                        
    def update(self,action,map: ThreeDMap):
        while self.prev_point:
            point = self.prev_point.pop()
            map.map[point[0]][point[1]][point[2]] = 0
        self.cur_point = (self.cur_point[0] + action[0], self.cur_point[1] + action[1], self.cur_point[2] + action[2])
        for i in range (-1,2):
            for j in range (-1,2):
                for k in range(-1,2):
                    if not isCollision(map,(self.cur_point[0]+i,self.cur_point[1]+j,self.cur_point[2]+k)): 
                        map.map[self.cur_point[0]+i][self.cur_point[1]+j][self.cur_point[2]+k] = 2
                        self.prev_point.append((self.cur_point[0]+i,self.cur_point[1]+j,self.cur_point[2]+k))
        return map
class target:
    def __init__(self,start_point,end_point) -> None:
        self.cur_point = start_point
        self.end_point = end_point
        self.path = None
        self.AStar = None
        self.next_move = None
    def check_obstacles_path(self,map:ThreeDMap): # check obstacles on the path
        if self.path == None:
            return True
        for points in self.path:
            if map.map[points[0]][points[1]][points[2]] != 0:
                return True
        return False
    def get_path(self,map:ThreeDMap):
        if self.check_obstacles_path(map):
            self.AStar = AStar(self.cur_point,self.end_point,map)
            self.path = self.AStar.search()
            if self.path:
                self.next_move = self.path.pop()
            else:
                self.next_move = self.cur_point
    def update(self,map:ThreeDMap):
        self.get_path(map)
        self.cur_point = self.next_move
        if not self.path :
                self.next_move = self.cur_point
        else:
            self.next_move = self.path.pop()


class chase3D():
    def __init__(self) -> None:
        
        self.action_size = 27
        self.agent_num = 3
        self.time_limits = 150
        self.time = 0
        self.action_space  = [(-1, -1, -1), (-1, -1, 0), (-1, -1, 1), (-1, 0, -1), 
                      (-1, 0, 0), (-1, 0, 1), (-1, 1, -1), (-1, 1, 0),
                      (-1, 1, 1), (0, -1, -1), (0, -1, 0), (0, -1, 1),
                      (0, 0, -1), (0, 0, 1), (0, 1, -1), (0, 1, 0),
                      (0, 1, 1), (1, -1, -1), (1, -1, 0), (1, -1, 1),
                      (1, 0, -1), (1, 0, 1), (1, 1, -1), (1, 1, 0),
                      (1, 1, 1), (0,0,0),(1,0,0)
                    ]
        self.limits = 100
        self.map = ThreeDMap(self.limits,self.limits,self.limits)
        self.map.add_obstacles()
        self.state_size = 6 + int(self.map.obstacles_size * 3)
        #self.map.add_blocker()
    def reset(self) -> None:
        self.map = ThreeDMap(self.limits,self.limits,self.limits)
        self.map.add_obstacles()
        x_start = random.randint(0,self.limits - 1)
        y_start = random.randint(0,self.limits - 1)
        z_start = random.randint(0,self.limits - 1)
        x_end = random.randint(0,self.limits - 1)
        y_end = random.randint(0,self.limits - 1)
        z_end = random.randint(0,self.limits - 1)
        self.target = target((x_start,y_start,z_start),(x_end,y_end,z_end))
        states = []
        self.agent = []
        for _ in range (self.agent_num):
            agent_x = random.randint(0,self.limits - 1)
            agent_y = random.randint(0,self.limits - 1)
            agent_z = random.randint(0,self.limits - 1)
            self.agent.append(Agent((agent_x,agent_y ,agent_z),self.map))
            state = [agent_x,agent_y ,agent_z,x_start,y_start,z_start]
            state = np.concatenate((np.array(state), self.map.obstacles))
            states.append(state)
        self.time = 0
        return np.array(states, dtype= np.int32)
    def is_captured(self):
        return self.map.map[self.target.cur_point[0]][self.target.cur_point[1]][self.target.cur_point[2]]== 2
    def step(self,action):
        states = []
        rewards = []
        dones = []
        if self.time % 3 == 0:
            self.target.update(self.map)
        cnt = 0
        terminated = False
        for agent in self.agent:
            move = self.action_space[action[cnt]] # update the point
            pre_agent_point = agent.cur_point
            agent.update(move,self.map)
            done = self.time >= self.time_limits
            reward = 0
            if self.is_captured() and not done : 
                done = True
                reward = 20000
            else:
                pre_distance = np.linalg.norm(np.array(pre_agent_point) - np.array(self.target.cur_point)) 
                time_penalty = -0.5   # calculate the reward
                #approach_reward = 150
                new_distance = np.linalg.norm(np.array(agent.cur_point) - np.array(self.target.cur_point))
                if pre_distance - new_distance > 0:
                    reward = 100
                else:
                    reward = -100
                reward += time_penalty * self.time
            state = np.array((agent.cur_point[0],agent.cur_point[1],agent.cur_point[2],self.target.cur_point[0],self.target.cur_point[1],self.target.cur_point[2]),dtype=np.int32)
            state = np.concatenate((state, self.map.obstacles))
            states.append(state)
            rewards.append(reward)
            dones.append(done)
            cnt += 1
        self.time = self.time + 1
        return states ,rewards, dones
        
