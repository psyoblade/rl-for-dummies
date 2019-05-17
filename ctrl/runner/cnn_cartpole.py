#!/usr/bin/env python
# -*- coding:utf-8 -*-

import gym
import numpy as np
from com.modulabs.ctrl.utils.FileUtils import FileUtils
from com.modulabs.ctrl.utils.SeedNumbers import RandomSeeder
from com.modulabs.ctrl.agents.CNNAgent import CNNAgent

RandomSeeder.set_seed()
GYM_ENV_NAME = 'BreakoutDeterministic-v4'
EPISODES = 1000

model_path = './models/cnn.h5'
cache_name = 'breakout'
cache_dir = './' + cache_name


def get_recent_modelname():
    import os
    files = os.listdir("models")
    files.sort()
    filename = files[-1]
    return filename


def get_model_path():
    global model_path
    model_path = './models/%s' % get_recent_modelname()
    return model_path


def train():
    global model_path
    env = gym.make(GYM_ENV_NAME)
    agent = CNNAgent(env, name=cache_name, rend=True)
    completed = False

    if FileUtils.exists_file(model_path):
        agent.load_weights(model_path)

    scores, episodes, global_step = [], [], 0

    for e in range(EPISODES):
        done = False
        dead = False

        step, score, start_life = 0, 0, 5
        history = agent.reset()
        score = 0

        while not done:
            if agent.rend:
                agent.render()

            global_step += 1
            step += 1

            action = agent.get_action(history)
            if action == 0:
                real_action = 1
            elif action == 1:
                real_action = 2
            else:
                real_action = 3

            next_history, reward, done, info = agent.step(real_action, history)

            if start_life > info['ale.lives']:
                dead = True
                start_life = info['ale.lives']

            reward = np.clip(reward, -1., 1.)
            agent.append_sample(history, action, reward, next_history, dead)

            if len(agent.replay_memory) >= agent.train_start:
                agent.train_model()

            if global_step % agent.update_target_rate == 0:
                agent.update_model()

            score += reward

            if dead:
                dead = False
            else:
                history = next_history

            if done:
                scores.append(score)
                agent.update_model()
                print("episode:{}, score:{} ".format(e, score))

                mean_score = np.mean(scores[-10:])
                if len(scores) > 10 and mean_score > 10:
                    agent.save_weights(model_path)
                    completed = True
                    print("last 10 mean of scores is {}, learning completed = {}".format(mean_score, scores[-10:]))
                    break

    return completed


def test():
    FileUtils.remove_dir(cache_dir)

    env = gym.make(GYM_ENV_NAME)
    wenv = gym.wrappers.Monitor(env, cache_name)
    agent = CNNAgent(wenv, epsilon=0.0)
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
