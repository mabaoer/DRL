import random

from tqdm import tqdm

from environment.tic_tac_toe import TicTacToeEnv
from agent.qlearning import QLearning
from config import Config




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
    agent.load("")
    for _ in tqdm(range(cfg.train_eps)):
        state = env.reset().copy()
        state_unused = env.state_unused.copy()
        first, second = random_first()
        while True:
            # 先手行动
            action = agent.choose_action(state, first, state_unused)
            next_state, reward, done, info, next_state_unused = env.step(action)
            env.render(action)
            agent.update(state, action, reward, next_state, done, state_unused, next_state_unused)
            if done:
                state = env.reset()
                state_unused = env.state_unused.copy()
                first, second = random_first()
                break
            state = next_state.copy()
            state_unused = next_state_unused.copy()
            # 后手行动

            action = agent.choose_action(state, second, state_unused)
            next_state, reward, done, info, next_state_unused = env.step(action)
            env.render(action)
            agent.update(state, action, reward, next_state, done, state_unused, next_state_unused)
            state = next_state.copy()
            state_unused = next_state_unused.copy()
            if done:
                state = env.reset()
                state_unused = env.state_unused.copy()
                first, second = random_first()
                break

    agent.save("")

    cfg = Config()
    cfg.render_times = 2
    env = TicTacToeEnv(cfg)
    for _ in range(cfg.test_eps):
        state = env.reset().copy()
        state_unused = env.state_unused.copy()
        first, second = random_first()
        while True:
            # 先手行动
            action = agent.choose_action_determinate(state, first, state_unused)
            next_state, reward, done, info, next_state_unused = env.step(action)
            env.render(action)
            agent.update(state, action, reward, next_state, done, state_unused, next_state_unused)
            if done:
                state = env.reset()
                state_unused = env.state_unused.copy()
                first, second = random_first()
                break
            state = next_state.copy()
            state_unused = next_state_unused.copy()
            # 后手行动

            action = agent.choose_action_determinate(state, second, state_unused)
            next_state, reward, done, info, next_state_unused = env.step(action)
            env.render(action)
            agent.update(state, action, reward, next_state, done, state_unused, next_state_unused)
            state = next_state.copy()
            state_unused = next_state_unused.copy()
            if done:
                state = env.reset()
                state_unused = env.state_unused.copy()
                first, second = random_first()
                break

