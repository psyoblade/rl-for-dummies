# -*- coding:utf-8 -*-

import gym
import numpy as np
import matplotlib.pyplot as plt
import random as pr

env = gym.make('FrozenLake-v0')

# deterministic q function
Q = np.zeros([env.observation_space.n, env.action_space.n])
num_episodes = 1000
rList = []
sList = []
def rargmax(vector):
    m = np.amax(vector) # extract max value
    indices = np.nonzero(vector == m)[0] # extract value-indices equals to maxval
    return pr.choice(indices) # return random value of same values

for i in range(num_episodes):
    s = env.reset()
    rAll = 0 # total reward
    d = False # end of learning
    j = 0 # value large enough
    sList = []

    while not d and j < 250: # loop until goal or 250 iterations
        j += 1
        a = rargmax(Q[s,:]) # input: actions at current-position s
        s1, r, d, _ = env.step(a) # input: best action
        if r == 1:
            print(sList)
        Q[s,a] = r + np.max(Q[s1,:]) # update previous q values with current q value
        s = s1 # update current position
        rAll = rAll + r # sum reward
        sList.append(s)
    
    rList.append(rAll) # append reward for debugging learning rate

print("final qtable values")
print(" left down right up")
print(Q)
print("성공한 확률 : ", len(rList)/num_episodes)

# plt.bar(range(len(rList)),rList, color="Blue")
# plt.show()

    # print(s) # s means current position (0 ~ 15)
    # print(Q) # 16 x 4 matrix
    # print(Q[0,:]) # first tuple