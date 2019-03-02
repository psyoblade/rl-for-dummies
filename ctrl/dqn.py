import gym
from com.modulabs.ctrl.utils.FileUtils import FileUtils
from com.modulabs.ctrl.utils.SeedNumbers import RandomSeeder
from com.modulabs.ctrl.agents.DQNAgent import DQNAgent

RandomSeeder.set_seed()
GYM_ENV_NAME = 'CartPole-v1'
EPISODES = 300

model_path = './models/dqn.h5'
cache_name = 'cartpole'
cache_dir = './' + cache_name


def train():
    env = gym.make(GYM_ENV_NAME)
    agent = DQNAgent(env)
    completed = False

    if FileUtils.exists_file(model_path):
        agent.load_weights(model_path)

    for e in range(EPISODES):
        done = False
        scores, episodes = [], []
        state = agent.reset()
        score = 0

        while not done:
            action = agent.get_action(state)
            next_state, reward, done, info = agent.step(action)

            agent.append_sample(state, action, reward, next_state, done)
            if len(agent.replay_memory) >= agent.train_start:
                agent.train_model()

            scores.append(reward)
            score += reward
            state = next_state

            if done:
                agent.update_model()
                print("episode:{}, score:{} ".format(e, score))

        if score >= 490 and e > 200:
            agent.save_weights(model_path)
            completed = True
            print("learning completed")
            break

    return completed


def test():
    env = gym.make(GYM_ENV_NAME)
    agent = DQNAgent(env, epsilon=0.0)
    if FileUtils.exists_file(model_path):
        agent.load_weights(model_path)
    FileUtils.remove_dir(cache_dir)
    wenv = gym.wrappers.Monitor(env, cache_name)

    done = False
    states = wenv.reset()
    while not done:
        wenv.render()
        action = agent.get_greedy_action(states)
        states, reward, done, _ = wenv.step(action)


if __name__ == "__main__":
    if train():
        test()
