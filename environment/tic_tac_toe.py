import time

import gym
import numpy as np
from gym.envs.classic_control import rendering


class TicTacToeEnv(gym.Env):
    def __init__(self, cfg):
        self.rows_cols = cfg.rows_cols  # 井字棋的行列数
        self.board_size = self.rows_cols * self.rows_cols
        self.state = np.zeros([self.rows_cols, self.rows_cols])
        self.state_unused = [(i, j) for i in range(self.rows_cols) for j in range(self.rows_cols)]
        self.winner = None
        self.line_size = cfg.line_size
        width, height = self.rows_cols * self.line_size, self.rows_cols * self.line_size
        self.viewer = rendering.Viewer(width, height)
        self.current_action = None
        self.count = -1
        self.render_times = cfg.render_times

    def reset(self):
        self.state = np.zeros([self.rows_cols, self.rows_cols])
        self.state_unused = [(i, j) for i in range(self.rows_cols) for j in range(self.rows_cols)]
        self.winner = None
        if self.count % self.render_times == 1:
            time.sleep(1)
            self.viewer.geoms.clear()  # 清空画板中需要绘制的元素
            self.viewer.onetime_geoms.clear()
        self.count += 1
        return self.state

    def step(self, action):
        # 动作的格式：action = {'mark':'circle'/'cross', 'pos':(x,y)}# 产生状态
        self.current_action = action
        x = action['pos'][0]
        y = action['pos'][1]
        if action['mark'] == 'blue':
            self.state[x][y] = 1
        elif action['mark'] == 'red':
            self.state[x][y] = -1
        self.state_unused.remove((x, y))
        # 奖励
        done = self.judge_end()
        if done:
            if self.winner == 'blue':
                reward = 1
            elif self.winner == 'red':
                reward = -1
            else:
                reward = 0
        else:
            reward = 0
        # 报告
        info = {}
        return self.state, reward, done, info, self.state_unused

    def judge_end(self):
        res = []
        # 检查N行N列
        for i in range(self.rows_cols):
            res.append(sum(self.state[:, i]))
        for j in range(self.rows_cols):
            res.append(sum(self.state[j, :]))
        # 检查对角线
        diag1, diag2 = 0, 0
        for i in range(self.rows_cols):
            diag1 += self.state[i][i]
            diag2 += self.state[i][self.rows_cols - i - 1]
        res.append(diag1)
        res.append(diag2)
        for flag in res:
            if flag == self.rows_cols:
                self.winner = 'blue'
                return True
            if flag == -self.rows_cols:
                self.winner = 'red'
                return True
        if len(self.state_unused) == 0:
            return True
        return False

    def render(self, mode='human'):
        if self.count % self.render_times == 0:
            for i in range(self.rows_cols - 1):
                line1 = rendering.Line((0, self.line_size * (i + 1)),
                                       (self.line_size * self.rows_cols, self.line_size * (i + 1)))
                line2 = rendering.Line((self.line_size * (i + 1), 0),
                                       (self.line_size * (i + 1), self.line_size * self.rows_cols))
                line1.set_color(0, 0, 0)
                line2.set_color(0, 0, 0)
                # 将绘画元素添加至画板中
                self.viewer.add_geom(line1)
                self.viewer.add_geom(line2)
            time.sleep(0.1)
            # 根据self.state画占位符
            i, j = self.current_action['pos'][0], self.current_action['pos'][1]
            if self.state[i][j] == 1:
                circle = rendering.make_circle(self.line_size / 3)  # 画直径为30的圆
                circle.set_color(135 / 255, 206 / 255, 250 / 255)  # mark = blue
                move = rendering.Transform(
                    translation=(i * self.line_size + self.line_size / 2,
                                 j * self.line_size + self.line_size / 2))  # 创建平移操作
                circle.add_attr(move)  # 将平移操作添加至圆的属性中
                self.viewer.add_geom(circle)  # 将圆添加至画板中

            if self.state[i][j] == -1:
                circle = rendering.make_circle(self.line_size / 3)
                circle.set_color(255 / 255, 182 / 255, 193 / 255)  # mark = red
                move = rendering.Transform(
                    translation=(i * self.line_size + self.line_size / 2,
                                 j * self.line_size + self.line_size / 2))
                circle.add_attr(move)
                self.viewer.add_geom(circle)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
