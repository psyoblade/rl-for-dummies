import opensim as osim
from osim.env import ProstheticsEnv

env = ProstheticsEnv(visualize=True)

def exit(default=0):
    import sys
    sys.exit(default)

def get_default():
    return [0.5488135, 0.71518934, 0.60276335, 0.5448832, 0.4236548, 0.6458941, 0.4375872, 0.891773, 0.96366274, 0.3834415, 0.79172504, 0.5288949, 0.56804454, 0.92559665, 0.07103606, 0.0871293, 0.0202184, 0.83261985, 0.77815676]

def my_controller(observation):
    return get_default()

observation = env.reset()
total_reward = 0.0

for i in range(200):
    [observation, reward, done, info] = env.step(my_controller(observation), True)
    total_reward += reward
    print("reward/total_reward : %0.2f/%0.2f" % (reward, total_reward))
    if done:
        print("done")
        break
print("Total reward %f" % total_reward)

