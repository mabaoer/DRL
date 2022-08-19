from environment.tic_tac_toe import TicTacToeEnv
from agent.qlearning import QLearning
from config import Config

cfg = Config()
env = TicTacToeEnv(cfg)
agent = QLearning(cfg)
agent.load("")
state = env.reset()
print(agent.Q_table[str(state)])
print(max(agent.Q_table[str(state)].values()))