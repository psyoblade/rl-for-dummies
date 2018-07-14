import opensim as osim
from osim.http.client import Client
from osim.env import ProstheticsEnv

remote_base = "http://grader.crowdai.org:1729"
crowdai_token = "a6d6c970d3883bee5730708739550518"

client = Client(remote_base)
observation = client.env_create(crowdai_token, env_id="ProstheticsEnv")
# env = ProstheticsEnv(visualize=True)
# i = 0

def exit(default=0):
    import sys
    sys.exit(default)

def get_default():
    return [0.5488135, 0.71518934, 0.60276335, 0.5448832, 0.4236548, 0.6458941, 0.4375872, 0.891773, 0.96366274, 0.3834415, 0.79172504, 0.5288949, 0.56804454, 0.92559665, 0.07103606, 0.0871293, 0.0202184, 0.83261985, 0.77815676]

def my_controller(observation):
    return get_default()

while True:
    # [observation, reward, done, info] = client.env_step(env.action_space.sample().tolist())
    [observation, reward, done, info] = client.env_step(my_controller(observation), True)
    if done:
        print("done")
        observation = client.env_reset()
        if not observation:
            print("break")
            break

client.submit()