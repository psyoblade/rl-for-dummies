from osim.env import ProstheticsEnv

env = ProstheticsEnv(visualize=True)
observation = env.reset()

def exit(default=0):
    import sys
    sys.exit(default)

actions = env.action_space.sample()
print(actions)
exit()


total_reward = 0.0
for i in range(200):
    observation, reward, done, info = env.step(env.action_space.sample())
    total_reward += reward
    # print(observation, reward, done, info)
    # observation 158개 실수값
    # reward 하나의 실수값
    # done boolean
    # info dict

print("Total reward is '%f'" % total_reward)