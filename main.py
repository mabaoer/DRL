import gym
import random
import time
from environment.tic_tac_toe import TicTacToeEnv


def randomAction(env_, mark):  # 随机选择未占位的格子动作
    action_pos = random.choice(env_.state_unused)
    action = {'mark': mark, 'pos': action_pos}
    return action


def randomFirst():
    if random.random() > 0.5:  # 随机先后手
        first_, second_ = 'blue', 'red'
    else:
        first_, second_ = 'red', 'blue'
    return first_, second_


env = TicTacToeEnv(rows_cols=5)
env.reset()  # 在第一次step前要先重置环境 不然会报错
first, second = randomFirst()
while True:
    # 先手行动
    action = randomAction(env, first)
    state, reward, done, info = env.step(action)
    env.render()
    time.sleep(0.1)
    if done:
        time.sleep(1)
        env.reset()
        env.render()
        first, second = randomFirst()
        continue
    # 后手行动
    action = randomAction(env, second)
    state, reward, done, info = env.step(action)
    env.render()
    time.sleep(0.1)
    if done:
        time.sleep(1)
        env.reset()
        env.render()
        first, second = randomFirst()
        continue
