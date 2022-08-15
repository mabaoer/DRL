import numpy as np
import math
import pickle
import random


class QLearning(object):
    def __init__(self, cfg):
        self.lr = cfg.lr  # 学习率
        self.gamma = cfg.gamma
        self.epsilon = 0
        self.sample_count = 0
        self.epsilon_start = cfg.epsilon_start
        self.epsilon_end = cfg.epsilon_end
        self.epsilon_decay = cfg.epsilon_decay
        self.Q_table = {}

    def choose_action(self, state, mark, state_unused):
        if str(state) not in self.Q_table:
            self.add_new_state(state, mark, state_unused)
        self.sample_count += 1
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                       math.exp(-1. * self.sample_count / self.epsilon_decay)  # epsilon是会递减的，这里选择指数递减
        # e-greedy 策略
        if np.random.uniform(0, 1) > self.epsilon:
            action_pos = np.argmax(self.Q_table[str(state)])  # 选择Q(s,a)最大对应的动作
            return {'mark': mark, 'pos': action_pos}
        else:
            return self.random_action(mark, state_unused)

    def random_action(self, mark, state_unused):
        action_pos = random.choice(state_unused)
        return {'mark': mark, 'pos': action_pos}

    def predict(self, state, mark):
        action_pos = np.argmax(self.Q_table[str(state)])
        return {'mark': mark, 'pos': action_pos}

    def update(self, state, action, reward, next_state, done):
        Q_predict = self.Q_table[str(state)][action]
        if done:  # 终止状态
            Q_target = reward
        else:
            Q_target = reward + self.gamma * np.max(self.Q_table[str(next_state)])
        self.Q_table[str(state)][action] += self.lr * (Q_target - Q_predict)

    def overTurn(self, state):  # 翻转状态
        state_ = state.copy()
        for i, row in enumerate(state_):
            for j, one in enumerate(row):
                if one != 0: state_[i][j] *= -1
        return state_

    def add_new_state(self, state, mark, state_unused):
        state = state if mark == 'blue' else self.overTurn(state)  # 如果是红方行动则翻转状态
        if str(state) not in self.Q_table:
            self.Q_table[str(state)] = {}
            for action in state_unused:
                self.Q_table[str(state)][str(action)] = 0

    def save(self, path):
        with open(path + 'Q_table_dict.pkl', 'wb') as f:
            pickle.dump(self.Q_table, f)
            print("保存模型成功！")

    def load(self, path):
        try:
            with open(path + 'Q_table_dict.pkl', 'rb') as f:
                self.Q_table = pickle.load(f)
        except (Exception,):
            self.Q_table = {}
        print("加载模型成功！")
