import gym
import numpy as np
from datetime import datetime
from com.modulabs.ctrl.utils.FileUtils import FileUtils
from com.modulabs.ctrl.utils.SeedNumbers import RandomSeeder
from com.modulabs.ctrl.agents.DQNAgent import DQNAgent

RandomSeeder.set_seed()
# RandomSeeder.reset()
GYM_ENV_NAME = 'CartPole-v1'
EPISODES = 1000

timestamp = datetime.now().strftime("%Y%m%d_%H%M")
model_name = 'cartpole'
model_path = 'models/{}/dqn-{}.h5'.format(model_name, timestamp)
cache_dir = 'caches/' + model_name


def train():
    global model_name, timestamp

    env = gym.make(GYM_ENV_NAME)
    agent = DQNAgent(env, name=model_name, timestamp=timestamp, use_swa=True)
    global_step = 0
    size_of_window = 15
    scores = []

    if FileUtils.exists_file(model_path):
        agent.load_weights(model_path)

    for e in range(EPISODES):
        done = False
        state = agent.reset()
        score, step = 0, 0

        while not done:
            global_step += 1
            step += 1

            action = agent.get_action(state)
            next_state, reward, done, info = agent.step(action)

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
                agent.update_model(e)

                if len(scores) == size_of_window:
                    scores.pop(0)
                scores.append(score)
                mean_score = np.mean(scores)
                print("global_step[{}] episode:{}, score:{}, mean_score:{} of scores:{}".format(global_step, e, score, mean_score, scores))
                agent.avg_q_max, agent.avg_loss = 0, 0

    agent.save_weights(model_path)
    print("last {} mean of scores is {}, learning completed = {}".format(size_of_window, mean_score, scores[-size_of_window:]))
    return True


def test(timestamp):
    global model_name

    FileUtils.remove_dir(cache_dir)
    env = gym.make(GYM_ENV_NAME)
    wenv = gym.wrappers.Monitor(env, timestamp)
    agent = DQNAgent(wenv, name=model_name, timestamp=timestamp, epsilon=0.0)

    if FileUtils.exists_file(model_path):
        agent.load_weights(model_path)

    done = False
    state = agent.reset()
    while not done:
        agent.render()
        action = agent.get_action(state)
        next_state, reward, done, info = wenv.step(action)
        state = next_state


if __name__ == "__main__":
    train()
    # test('20190424_2146')
