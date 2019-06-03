#!/usr/bin/env python
# -*- coding:utf-8 -*-

import gym
import numpy as np
from datetime import datetime
from com.modulabs.ctrl.utils.FileUtils import FileUtils
from com.modulabs.ctrl.utils.SeedNumbers import RandomSeeder
from com.modulabs.ctrl.agents.RNDAgent import RNDAgent

RandomSeeder.set_seed()
GYM_ENV_NAME = 'Pong-v0'
EPISODES = 1000

timestamp = datetime.now().strftime("%Y%m%d_%H%M")
model_name = 'pong'
model_path = 'models/{}/dqn-{}.h5'.format(model_name, timestamp)
cache_dir = 'caches/' + model_name


def train():
    global model_name, model_path, timestamp

    env = gym.make(GYM_ENV_NAME)
    agent = RNDAgent(env)
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

            score += reward
            state = next_state

            if done:
                scores.append(score)
                # agent.update_model()
                print("episode:{}, score:{} ".format(e, score))

        mean_score = np.mean(scores[-10:])
        if len(scores) > 10 and mean_score > 490:
            agent.save_weights(model_path)
            completed = True
            print("last 10 mean of scores is {}, learning completed = {}".format(mean_score, scores[-10:]))
            break

    return completed


def test():
    FileUtils.remove_dir(cache_dir)

    env = gym.make(GYM_ENV_NAME)
    wenv = gym.wrappers.Monitor(env, cache_name)
    agent = RNDAgent(wenv, epsilon=0.0)
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
