import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class RNDAgent:

    def __init__(self, env, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.01, batch_size=64,
                 train_start=1000, learning_rate=0.001, learning_epsilon=0.01,
                 discount_factor=0.99, max_replay=10000, load_model=False):
        self.env = env
        self.load_model = load_model
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.learning_rate = learning_rate
        self.learning_epsilon = learning_epsilon
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.train_start = train_start
        self.discount_factor = discount_factor
        self.target_policy = self.build_model()
        self.random_policy = self.build_randnetwork()
        self.replay_memory = deque(maxlen=max_replay)

    def __del__(self):
        pass

    def render(self):
        self.env.render()

    def reset(self):
        state = self.env.reset()
        state = np.reshape(state, [1, self.state_size])
        return state

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        next_state = np.reshape(next_state, [1, self.state_size])
        return next_state, reward, done, info

    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(24, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def build_randnetwork(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu', kernel_initializer='random_normal'))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='random_normal'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_value = self.target_policy.predict(state)
        return np.argmax(q_value[0])

    def append_sample(self, state, action, reward, next_state, done):
        x = (state, action, reward, next_state, done)
        self.replay_memory.append(x)

    # TODO 학습 시에는 pred - rand 예측의 차이의 loss 값을 linear 하게 더해준다
    # TODO 매 step 시에 intrinsic reward 값을 더하는 함수를 노출한다
    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        mini_batch = random.sample(self.replay_memory, self.batch_size)
        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        actions, rewards, dones = [], [], []
        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            dones.append(mini_batch[i][4])

        target_values = self.target_policy.predict(states)  # predict((64,4)) -> (64, 2) : 좌.우
        expected_values = self.random_policy.predict(next_states)
        # explorer_bonus = K.get_value(K.mean(K.square(target_values - expected_values))) * 0.01

        # Bugfix, actions[x] 값이 실수라서 인덱스 오류
        for x in range(self.batch_size):
            if dones[x]:
                target_values[x][int(actions[x])] = rewards[x]
            else:
                target_values[x][int(actions[x])] = rewards[x] + self.discount_factor * np.amax(expected_values[x])
                # print("ext", rewards[x], "int", explorer_bonus, "xx", self.diff * np.amax(expected_values[x]))
                # target_values[x][int(actions[x])] = rewards[x] + explorer_bonus

        self.target_policy.fit(states, target_values, batch_size=self.batch_size, epochs=1, verbose=0)

    def save_weights(self, filename):
        self.target_policy.save_weights(filename)

    def load_weights(self, filename):
        self.target_policy.load_weights(filename)

