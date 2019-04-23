#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import random
import tensorflow as tf
from collections import deque
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dense
from keras import backend as K
from keras.optimizers import Adam
from skimage.transform import resize
from skimage.color import rgb2gray


class CNNAgent:

    def __init__(self, env, name, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.01, batch_size=32,
                 train_start=50000, update_target_rate=10000, learning_rate=0.001, discount_factor=0.99,
                 max_replay=400000, load_model=False, no_op_steps=30, rend=False):
        self.env = env
        self.name = name
        self.load_model = load_model
        self.no_op_steps = no_op_steps
        self.rend = rend
        # Bugfix, 실제 state 값은 env.observation_space.shape 84 x 84 x 3 이지만 history 4장을 쓰기 때문에 수정
        self.state_size = (84, 84, 4)
        self.action_size = env.action_space.n
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.train_start = train_start
        self.update_target_rate = update_target_rate
        self.discount_factor = discount_factor
        # 행동 및 타깃 모델 초기화 리플레이 버퍼 생성
        self.behavior_policy = self.build_model()
        self.target_policy = self.build_model()
        self.replay_memory = deque(maxlen=max_replay)
        self.update_model()
        self.optimizer = self.optimizer()
        # 텐서보드 설정
        self.session = tf.InteractiveSession()
        K.set_session(self.session)
        # 서머리 정보 저장
        self.avg_q_max, self.avg_loss = 0, 0
        self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
        self.summary_writer = tf.summary.FileWriter("summary/{}".format(self.name), self.session.graph)
        self.session.run(tf.global_variables_initializer())

    def __del__(self):
        pass

    def render(self):
        self.env.render()

    @staticmethod
    def rgb_to_gray(obs):
        return np.uint8(resize(rgb2gray(obs), (84, 84), mode='constant') * 255)

    @staticmethod
    def setup_summary():
        episode_total_reward = tf.Variable(0.)
        episode_avg_max_q = tf.Variable(0.)
        episode_duration = tf.Variable(0.)
        episode_avg_loss = tf.Variable(0.)

        tf.summary.scalar('Total Reward/Episode', episode_total_reward)
        tf.summary.scalar('Average Max Q/Episode', episode_avg_max_q)
        tf.summary.scalar('Duration/Episode', episode_duration)
        tf.summary.scalar('Average Loss/Episode', episode_avg_loss)

        summary_vars = [episode_total_reward, episode_avg_max_q, episode_duration, episode_avg_loss]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op

    # state = 84 x 84 x 3 ( 84 * 84 rgb pixel image)
    def reset(self):
        do_nothing = 0
        obs = self.env.reset()
        for _ in range(random.randint(1, self.no_op_steps)):
            obs, _, _, _ = self.env.step(do_nothing)
        state = self.rgb_to_gray(obs)
        history = np.stack((state, state, state, state), axis=2)
        history = np.reshape([history], (1, 84, 84, 4))
        return history

    def step(self, action, history):
        next_obs, reward, done, info = self.env.step(action)
        next_state = self.rgb_to_gray(next_obs)
        next_state = np.reshape([next_state], (1, 84, 84, 1))
        next_history = np.append(next_state, history[:, :, :, :3], axis=3)
        return next_history, reward, done, info

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=self.state_size))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size))
        model.summary()
        return model

    def get_greedy_action(self, history):
        history = np.float32(history / 255.0)
        q_value = self.behavior_policy.predict(history)
        return np.argmax(q_value[0])

    def get_random_action(self):
        action = random.randrange(self.action_size)
        return action

    def get_action(self, history):
        history = np.float32(history / 255.0)
        if np.random.rand() <= self.epsilon:
            return self.get_random_action()
        q_value = self.behavior_policy.predict(history)
        return np.argmax(q_value[0])

    def append_sample(self, history, action, reward, next_history, done):
        x = (history, action, reward, next_history, done)
        self.replay_memory.append(x)

    def update_model(self):
        self.target_policy.set_weights(self.behavior_policy.get_weights())

    def optimizer(self):
        a = K.placeholder(shape=(None,), dtype='int32')
        y = K.placeholder(shape=(None,), dtype='float32')
        prediction = self.behavior_policy.output
        a_one_hot = K.one_hot(a, self.action_size)
        q_value = K.sum(prediction * a_one_hot, axis=1)
        error = K.abs(y - q_value)

        quadratic_part = K.clip(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)

        optimizer = Adam(lr=0.00025, epsilon=0.01)
        updates = optimizer.get_updates(self.behavior_policy.trainable_weights, [], loss)
        train = K.function([self.behavior_policy.input, a, y], [loss], updates=updates)
        return train

    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        mini_batch = random.sample(self.replay_memory, self.batch_size)
        history = np.zeros((self.batch_size, self.state_size[0], self.state_size[1], self.state_size[2]))
        next_history = np.zeros((self.batch_size, self.state_size[0], self.state_size[1], self.state_size[2]))
        target = np.zeros((self.batch_size,))
        actions, rewards, dones = [], [], []
        for i in range(self.batch_size):
            history[i] = np.float32(mini_batch[i][0] / 255.)
            next_history[i] = np.float32(mini_batch[i][3] / 255.)
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            dones.append(mini_batch[i][4])

        expected_values = self.target_policy.predict(next_history)

        for x in range(self.batch_size):
            if dones[x]:
                target[x] = rewards[x]
            else:
                target[x] = rewards[x] + self.discount_factor * np.amax(expected_values[x])

        loss = self.optimizer([history, actions, target])
        self.avg_loss += loss[0]

    def save_weights(self, filename):
        print("save_weights:{}".format(filename))
        self.target_policy.save_weights(filename)

    def load_weights(self, filename):
        print("load_weights:{}".format(filename))
        self.target_policy.load_weights(filename)
        self.behavior_policy.load_weights(filename)

