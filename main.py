import random
import time
from environment.tic_tac_toe import TicTacToeEnv
from agent.qlearning import QLearning
import os

curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径


class Config:
    def __init__(self):
        # 环境超参数
        self.env_name = "tic_tac_toe"
        self.rows_cols = 3  # 井字棋的行列数
        self.line_size = 100  # 井字棋每个各自的长宽
        self.render_times = 10  # 每隔多久渲染一次

        ################################## 环境超参数 ###################################
        self.algo_name = 'Q-learning'  # 算法名称
        self.seed = 10  # 随机种子，置0则不设置随机种子
        self.train_eps = 400  # 训练的回合数
        self.test_eps = 30  # 测试的回合数
        ################################################################################

        ################################## 算法超参数 ###################################
        self.gamma = 0.90  # 强化学习中的折扣因子
        self.epsilon_start = 0.95  # e-greedy策略中初始epsilon
        self.epsilon_end = 0.01  # e-greedy策略中的终止epsilon
        self.epsilon_decay = 300  # e-greedy策略中epsilon的衰减率
        self.lr = 0.1  # 学习率
        ################################################################################

        ################################# 保存结果相关参数 ################################
        self.result_path = curr_path + "/outputs/" + self.env_name + '/results/'  # 保存结果的路径
        self.model_path = curr_path + "/outputs/" + self.env_name + '/models/'  # 保存模型的路径
        self.save = True  # 是否保存图片
        ################################################################################


def random_first():
    if random.random() > 0.5:  # 随机先后手
        first_, second_ = 'blue', 'red'
    else:
        first_, second_ = 'red', 'blue'
    return first_, second_


if __name__ == "__main__":
    cfg = Config()
    env = TicTacToeEnv(cfg)
    agent = QLearning(cfg)
    env.reset()  # 在第一次step前要先重置环境 不然会报错
    first, second = random_first()
    while True:
        # 先手行动
        action = agent.choose_action(env.state, first, env.state_unused)
        state, reward, done, info = env.step(action)
        env.render(action)
        if done:
            env.reset()
            first, second = random_first()
            continue

        # 后手行动
        action = agent.choose_action(env.state, second, env.state_unused)
        state, reward, done, info = env.step(action)
        env.render()
        if done:
            env.reset()
            first, second = random_first()
            continue
