import gym
import numpy as np
from datetime import datetime
from com.modulabs.ctrl.utils.FileUtils import FileUtils
from com.modulabs.ctrl.utils.SeedNumbers import RandomSeeder
from com.modulabs.ctrl.agents.CNNAgent import CNNAgent

RandomSeeder.set_seed()
RandomSeeder.reset()
GYM_ENV_NAME = 'Pong-v0'
EPISODES = 1000

model_name = 'pong'
cache_dir = 'caches/' + model_name
timestamp = datetime.now().strftime("%Y%m%d_%H%M")


def get_model_dir(_model_name, _timestamp):
    _model_dir = 'models/{}/dqn-{}.h5'.format(_model_name, _timestamp)
    print("get_model_dir({})".format(_model_dir))
    return _model_dir


def train(_model_name, _timestamp, _model_dir, _render):

    env = gym.make(GYM_ENV_NAME)
    agent = CNNAgent(env, name=_model_name, timestamp=_timestamp, train_start=50000)
    global_step = 0
    size_of_window = 15
    scores = []
    mean_score = 0.0

    if FileUtils.exists_file(_model_dir):
        agent.load_weights(_model_dir)

    for e in range(EPISODES):
        done = False
        history = agent.reset()
        score, step = 0, 0

        while not done:
            if (global_step % 1000) == 0:
                import datetime
                now = datetime.datetime.now()
                current_time = now.strftime("%Y-%m-%d %H:%M")
                print("[{}] global_step [{}] has passed.".format(current_time, global_step))
            global_step += 1
            step += 1

            if _render:
                agent.render()
            action = agent.get_action(history)
            next_history, reward, done, info = agent.step(action, history)

            agent.append_sample(history, action, reward, next_history, done)

            score += reward
            history = next_history

            # CNN 학습 시에 매 step 마다 학습하는 경우 너무 오랜 시간과 랙 발생
            size_of_replay_memory = len(agent.replay_memory)

            # 100 스텝 당 한 번씩 만 학습
            if size_of_replay_memory >= agent.train_start and (step % 100) == 0:
                agent.train_model()

            if reward > 0:
                # 텐서보드 로깅을 하기 위한 장치
                if global_step > agent.train_start:
                    stats = [score, agent.avg_q_max / float(step), step, agent.avg_loss / float(step)]
                    for i in range(len(stats)):
                        agent.session.run(agent.update_ops[i], feed_dict={agent.summary_placeholders[i]: float(stats[i])})
                    summary_str = agent.session.run(agent.summary_op)
                    agent.summary_writer.add_summary(summary_str, e + 1)
                # 보상이 있는 경우에만 모델을 업데이트
                agent.update_model()

                if len(scores) == size_of_window:
                    scores.pop(0)
                scores.append(score)
                mean_score = np.mean(scores)
                print("global_step[{}] episode:{}, score:{}, mean_score:{} of scores:{}".format(global_step, e, score, mean_score, scores))
                agent.avg_q_max, agent.avg_loss = 0, 0

    agent.save_weights(_model_dir)
    print("last {} mean of scores is {}, learning completed = {}".format(size_of_window, mean_score, scores[-size_of_window:]))


def test(_model_name, _timestamp, _model_dir):

    FileUtils.remove_dir(cache_dir)
    env = gym.make(GYM_ENV_NAME)

    monitor_file="monitors/{}".format(_timestamp)
    wenv = gym.wrappers.Monitor(env, monitor_file, force=True)
    agent = CNNAgent(wenv, name=_model_name, timestamp=_timestamp, epsilon=0.0)

    print("model_dir:{}".format(_model_dir))
    if FileUtils.exists_file(_model_dir):
        agent.load_weights(_model_dir)

    done = False
    history = agent.reset()

    while not done:
        agent.render()
        action = agent.get_greedy_action(history)
        # action = agent.get_random_action()
        # action = agent.get_action(history)
        next_history, reward, done, info = agent.env_step(wenv, action, history)
        history = next_history


# python pong.py train render [model_path]
# python pong.py test  render [model_path]
if __name__ == "__main__":
    import sys
    target = sys.argv[1]
    model_dir = sys.argv[3] if len(sys.argv) > 3 else get_model_dir(model_name, timestamp)
    if target == "test":
        render = sys.argv[2]
        test(model_name, timestamp, model_dir)
    else:
        render = True if sys.argv[2] == "render" else False
        train(model_name, timestamp, model_dir, render)
        test(model_name, timestamp, model_dir)
