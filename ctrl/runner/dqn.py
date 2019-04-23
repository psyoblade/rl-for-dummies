import gym
import numpy as np
from datetime import datetime
from com.modulabs.ctrl.utils.FileUtils import FileUtils
from com.modulabs.ctrl.utils.SeedNumbers import RandomSeeder
from com.modulabs.ctrl.agents.DQNAgent import DQNAgent

RandomSeeder.set_seed()
RandomSeeder.reset()
GYM_ENV_NAME = 'CartPole-v1'
EPISODES = 300

timestamp = datetime.now().strftime("%Y%m%d_%H%M")
model_name = 'cartpole'
model_path = 'models/{}/dqn-{}.h5'.format(model_name, timestamp)
cache_dir = 'caches/' + model_name


def train():
    global model_name, timestamp

    env = gym.make(GYM_ENV_NAME)
    agent = DQNAgent(env, name=model_name, timestamp=timestamp)
    completed = False
    global_step = 0

    if FileUtils.exists_file(model_path):
        agent.load_weights(model_path)

    for e in range(EPISODES):
        done = False
        scores, episodes = [], []
        state = agent.reset()
        score, step = 0, 0

        while not done:
            if agent.rend:
                agent.render()

            global_step += 1
            step += 1

            action = agent.get_action(state)
            next_state, reward, done, info = agent.step(action)
            # reward = reward if not done or score == 499 else -100

            agent.append_sample(state, action, reward, next_state, done)
            if len(agent.replay_memory) >= agent.train_start:
                agent.train_model()

            score += reward
            state = next_state

            if done:
                if global_step > agent.train_start:
                    stats = [score, agent.avg_q_max / float(step), step, agent.avg_loss / float(step)]
                    for i in range(len(stats)):
                        agent.session.run(agent.update_ops[i], feed_dict={agent.summary_placeholders[i]: float(stats[i])})
                    summary_str = agent.session.run(agent.summary_op)
                    agent.summary_writer.add_summary(summary_str, e + 1)
                scores.append(score)
                agent.update_model()
                print("global_step[{}] episode:{}, score:{} ".format(global_step, e, score))

                agent.avg_q_max, agent.avg_loss = 0, 0

            mean_score = np.mean(scores[-10:])
            if len(scores) > 10 and mean_score > 490:
                agent.save_weights(model_path)
                completed = True
                print("last 10 mean of scores is {}, learning completed = {}".format(mean_score, scores[-10:]))
                break

    return completed


def test():
    global model_name, timestamp

    FileUtils.remove_dir(cache_dir)
    env = gym.make(GYM_ENV_NAME)
    wenv = gym.wrappers.Monitor(env, model_name)
    agent = DQNAgent(wenv, name=model_name, timestamp=timestamp, epsilon=0.0)

    if FileUtils.exists_file(model_path):
        agent.load_weights(model_path)

    done = False
    states = agent.reset()
    while not done:
        agent.render()
        action = agent.get_action(states)
        states, reward, done, _ = wenv.step(action)


if __name__ == "__main__":
    if train():
        test()
