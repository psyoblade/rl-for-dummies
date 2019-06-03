#!/usr/bin/env python
# -*- coding:utf-8 -*-
# TODO: intrinsic_reward 사용 시에 replay 메모리에 넣지 말고, 실제 사용 시에 intrinsic 을 써야 효과가 있다
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


class RNDAgent:

    def __init__(self, env, name, timestamp, simulation_step=100000, epsilon=1.0, epsilon_decay=0.999,
                 epsilon_min=0.01, batch_size=32, network_size=256, train_start=5000, update_target_rate=10000,
                 learning_rate=0.00025, learning_epsilon=0.01, discount_factor=0.99, max_replay=10000,
                 load_model=False, no_op_steps=30, intrinsic_reward_factor=0.00001):
        self.env = env
        self.name = name
        self.timestamp = timestamp
        self.simulation_step = simulation_step
        self.intrinsic_reward_factor = intrinsic_reward_factor
        self.histories = []
        self.load_model = load_model
        self.no_op_steps = no_op_steps
        self.width, self.height, self.color = env.observation_space.shape
        self.ages = 4  # length of image history
        self.state_size = (self.width, self.height, self.ages)
        self.action_size = env.action_space.n
        self.learning_rate = learning_rate
        self.learning_epsilon = learning_epsilon
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.network_size = network_size
        self.train_start = train_start
        self.update_target_rate = update_target_rate
        self.discount_factor = discount_factor
        # 행동 및 타깃 모델 초기화 리플레이 버퍼 생성
        self.flip_random_seed(False)
        self.random_policy = self.build_randnet(self.network_size, self.state_size, self.action_size)
        self.flip_random_seed(True)
        self.predict_policy = self.build_prednet(self.network_size, self.state_size, self.action_size)
        self.behavior_policy = self.build_model(self.network_size, self.state_size, self.action_size)
        self.target_policy = self.build_model(self.network_size, self.state_size, self.action_size)
        self.replay_memory = deque(maxlen=max_replay)
        self.update_model()
        self.normal_optimizer = self.build_normal_optimizer()
        self.explor_optimizer = self.build_explor_optimizer()
        # 텐서보드 설정
        self.session = tf.InteractiveSession()
        K.set_session(self.session)
        # 서머리 정보 저장
        self.avg_q_max, self.avg_loss = 0, 0
        self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
        self.summary_writer = tf.summary.FileWriter("summary/{}/{}".format(self.name, self.timestamp), self.session.graph)
        self.session.run(tf.global_variables_initializer())
        # 통계정보 저장
        self.mean = 0.
        self.std = 0.

    def __del__(self):
        pass

    def render(self):
        self.env.render()

    @staticmethod
    def debug(name, obs):
        import sys
        str = "\nname:{} \n shape:{} \n data:{}".format(name, obs.shape, obs)
        sys.stdout.write(str)
        sys.stdout.flush()

    @staticmethod
    def flip_random_seed(on=True):
        from com.modulabs.ctrl.utils.SeedNumbers import RandomSeeder
        if on:
            RandomSeeder.set_seed()
        else:
            RandomSeeder.reset()

    @staticmethod
    def rgb_to_gray(obs, width, height):
        return np.uint8(resize(rgb2gray(obs), (width, height), mode='constant') * 255)

    @staticmethod
    def get_norm_params(histories, width, height, ages):  # [(1, 84, 84, 4) ...]
        _mean = []
        _std = []
        mean_mtx = [[[] for _ in range(width)] for _ in range(height)]
        std_mtx = [[[] for _ in range(width)] for _ in range(height)]
        step = 0
        for history in histories:
            assert(history.shape == (1, width, height, ages))
            for nil in history:
                nrow = 0
                for row in nil:
                    ncol = 0
                    for col in row:
                        mean_mtx[nrow][ncol].append(col)
                        std_mtx[nrow][ncol].append(col)
                        ncol += 1
                    nrow += 1
            step += 1

        for nrow in range(width):
            for ncol in range(height):
                mean_mtx[nrow][ncol] = np.mean(mean_mtx[nrow][ncol])
                std_mtx[nrow][ncol] = np.mean(std_mtx[nrow][ncol])
        actual_mean = np.reshape(np.array(mean_mtx), (1, width, height, 1))
        actual_std = np.reshape(np.array(std_mtx), (1, width, height, 1))
        return actual_mean, actual_std + 1e-5

    @staticmethod
    def get_normalized_obs(history, mean, std, min=-5, max=5):
        norm_history = (history - mean)/std
        clip_history = np.clip(norm_history, min, max)
        return clip_history

    @staticmethod
    def calculate_reward_in(pred_net, rand_net, history, intrinsic_reward_factor):

        # RNDAgent.debug('history', history)
        # You may need to normalize history to -5 ~ 5
        next_pred_history = pred_net.predict(history)  # (1, 54)
        next_rand_history = rand_net.predict(history)  # (1, 84, 84, many)
        # RNDAgent.debug('next_pred_history', next_pred_history)
        # RNDAgent.debug('next_rand_history', next_rand_history)

        squared = np.square(next_pred_history - next_rand_history)
        reward = np.mean(squared)
        res = reward * intrinsic_reward_factor
        # print("calculate_reward_in:{}".format(res))
        return res

        # clipped_reward = np.clip(reward, -1, 1)
        # print("calculate_reward_in:{}, clipped:{}".format(reward, clipped_reward))
        # return clipped_reward

    @staticmethod
    def build_randnet(network_size, state_size, action_size):
        print("build_randnet({}, {}, {})".format(network_size, state_size, action_size))
        single = network_size
        double = single * 2
        model = Sequential()
        model.add(Dense(single, activation='relu', kernel_initializer='random_normal', input_shape=state_size))
        model.add(Dense(double, activation='relu', kernel_initializer='random_normal'))
        model.add(Dense(action_size, activation='linear', kernel_initializer='random_normal'))
        model.summary()
        return model

    @staticmethod
    def build_prednet(network_size, state_size, action_size):
        print("build_prednet({}, {}, {})".format(network_size, state_size, action_size))
        single = network_size
        double = single * 2
        model = Sequential()
        model.add(Dense(single, activation='relu', kernel_initializer='random_normal', input_shape=state_size))
        model.add(Dense(double, activation='relu', kernel_initializer='random_normal'))
        model.add(Dense(action_size, activation='linear', kernel_initializer='random_normal'))
        model.summary()
        return model

    @staticmethod
    def build_model(network_size, state_size, action_size):
        double = network_size * 2
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=state_size))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(double, activation='relu'))
        model.add(Dense(action_size))
        model.summary()
        return model

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

    def get_reward_in(self, history):
        return self.calculate_reward_in(self.predict_policy, self.random_policy, history, self.intrinsic_reward_factor)

    def simulation(self):
        init_steps = 0
        while True:
            history = self.reset()
            done = False
            while not done:
                action = self.get_random_action()
                next_history, _, done, _ = self.step(action, history)
                self.histories.append(next_history)
                init_steps += 1
                if (init_steps % 100) == 0:
                    print("init_steps:{}, simulation_step:{}".format(init_steps, self.simulation_step))
                if init_steps == self.simulation_step:
                    self.mean, self.std = self.get_norm_params(self.histories, self.width, self.height, self.ages)
                    return

    # state = 84 x 84 x 3 ( 84 * 84 rgb pixel image)
    def reset(self):
        do_nothing = 0
        obs = self.env.reset()
        for _ in range(random.randint(1, self.no_op_steps)):
            obs, _, _, _ = self.env.step(do_nothing)
        state = self.rgb_to_gray(obs, self.width, self.height)
        history = np.stack((state, state, state, state), axis=2)  # (84, 84, 4) axis 는 어느 차원에 넣을지를 결정 (0, 1, 2) 위치
        history = np.reshape([history], (1, self.width, self.height, self.ages))  # (1, 84, 84, 4)
        return history

    def close(self):
        self.env.close()

    def org_step(self, action, history):
        next_obs, reward, done, info = self.env.step(action)
        next_state = self.rgb_to_gray(next_obs, self.width, self.height)
        next_state = np.reshape([next_state], (1, self.width, self.height, 1))
        next_history = np.append(next_state, history[:, :, :, :3], axis=3)  # 다음 상태 + 지난 히스토리 3개
        return next_history, reward, done, info

    def step(self, action, history):
        return self.env_step(self.env, action, history)

    def env_step(self, _env, _action, _history):
        next_obs, reward, done, info = _env.step(_action)
        next_state = self.rgb_to_gray(next_obs, self.width, self.height)
        next_state = np.reshape([next_state], (1, self.width, self.height, 1))
        next_history = np.append(next_state, _history[:, :, :, :3], axis=3)
        return next_history, reward, done, info


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

    def build_normal_optimizer(self):
        a = K.placeholder(shape=(None,), dtype='int32')
        y = K.placeholder(shape=(None,), dtype='float32')

        behavior = self.behavior_policy.output
        a_one_hot = K.one_hot(a, self.action_size)
        q_value = K.sum(behavior * a_one_hot, axis=1)
        error = K.abs(y - q_value)
        quadratic_part = K.clip(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)

        print(" ##### build_normal_optimizer(input:{}, output:{}) #####\n".format(self.behavior_policy.input, self.behavior_policy.output))
        adam = Adam(lr=self.learning_rate, epsilon=self.learning_epsilon)
        updates = adam.get_updates(self.behavior_policy.trainable_weights, [], loss)
        optim = K.function([self.behavior_policy.input, a, y], [loss], updates=updates)
        return optim

    def build_explor_optimizer(self):
        a = K.placeholder(shape=(None,), dtype='int32')
        y = K.placeholder(shape=(None,), dtype='float32')

        pred_output = self.predict_policy.output
        rand_output = self.random_policy.output
        loss = K.mean(K.square(pred_output - rand_output))

        print(" ##### build_explor_optimizer(input:{}, output:{}, input:{}, output:{}) #####\n".format(self.random_policy.input, self.random_policy.output, self.predict_policy.input, self.predict_policy.output))
        adam = Adam(lr=self.learning_rate, epsilon=self.learning_epsilon)
        updates = adam.get_updates(self.predict_policy.trainable_weights, [], loss)
        optim = K.function([self.predict_policy.input, self.random_policy.input, a, y], [loss], updates=updates)
        return optim

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

        assert(not None in history)
        assert((self.batch_size, self.width, self.height, self.ages) == history.shape)

        loss_nor = self.normal_optimizer([history, actions, target])
        loss_exp = self.explor_optimizer([history, history, actions, target])
        loss = loss_exp[0] + loss_nor[0]
        # print("loss:{} = loss_exp:{} + loss_nor:{}".format(round(loss, 5), round(loss_exp[0], 5), round(loss_nor[0], 5)))
        self.avg_loss += loss

    def save_weights(self, filename):
        print("save_weights:{}".format(filename))
        self.target_policy.save_weights(filename)

    def load_weights(self, filename):
        print("load_weights:{}".format(filename))
        self.target_policy.load_weights(filename)
        self.behavior_policy.load_weights(filename)

    def debug_stats(self):
        print("mean:{},\nstd:{}".format(self.mean, self.std))
