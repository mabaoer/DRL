#######################################################################
# Copyright (C)                                                       #
# 2016 - 2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)           #
# 2016 Jan Hakenberg(jan.hakenberg@gmail.com)                         #
# 2016 Tian Jun(tianjun.cpp@gmail.com)                                #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
# https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter01/tic_tac_toe.py
from tqdm import tqdm

from environment.state import State, all_states
from agent.player import Player


class Judger:
    # @player1: the player who will move first, its chessman will be 1
    # @player2: another player with a chessman -1
    def __init__(self, player1, player2):
        self.p1 = player1
        self.p2 = player2
        self.current_player = None
        self.p1_symbol = 1
        self.p2_symbol = -1
        self.p1.set_symbol(self.p1_symbol)
        self.p2.set_symbol(self.p2_symbol)
        self.current_state = State()

    def reset(self):
        self.p1.reset()
        self.p2.reset()

    def alternate(self):
        while True:
            yield self.p1
            yield self.p2

    # @print_state: if True, print each board during the game
    def play(self, print_state=False):
        alternator = self.alternate()
        self.reset()
        current_state = State()
        self.p1.set_state(current_state)
        self.p2.set_state(current_state)
        if print_state:
            current_state.print_state()
        while True:
            player = next(alternator)
            i, j, symbol = player.act()
            next_state_hash = current_state.next_state(i, j, symbol).hash()
            current_state, is_end, _ = all_states[next_state_hash]
            self.p1.set_state(current_state)
            self.p2.set_state(current_state)
            if print_state:
                current_state.print_state()
            if is_end:
                return current_state.winner


def train(epochs, print_every_n=1000):
    player1 = Player(epsilon=0.01)
    player2 = Player(epsilon=0.01)
    judger = Judger(player1, player2)
    player1_win = 0.0
    player2_win = 0.0
    for i in tqdm(range(1, epochs + 1)):
        winner = judger.play(print_state=False)
        if winner == 1:
            player1_win += 1
        if winner == -1:
            player2_win += 1
        if i % print_every_n == 0:
            print('Epoch %d, player 1 winrate: %.02f, player 2 winrate: %.02f' % (i, player1_win / i, player2_win / i))
        player1.backup()
        player2.backup()
        judger.reset()
    player1.save_policy()
    player2.save_policy()


def compete(turns):
    player1 = Player(epsilon=0)
    player2 = Player(epsilon=0)
    judger = Judger(player1, player2)
    player1.load_policy()
    player2.load_policy()
    player1_win = 0.0
    player2_win = 0.0
    for _ in range(turns):
        winner = judger.play()
        if winner == 1:
            player1_win += 1
        if winner == -1:
            player2_win += 1
        judger.reset()
    print('%d turns, player 1 win %.02f, player 2 win %.02f' % (turns, player1_win / turns, player2_win / turns))


# class HumanPlayer:
#     def __init__(self, **kwargs):
#         self.symbol = None
#         self.keys = ['q', 'w', 'e', 'a', 's', 'd', 'z', 'x', 'c']
#         self.state = None
#
#     def reset(self):
#         pass
#
#     def set_state(self, state):
#         self.state = state
#
#     def set_symbol(self, symbol):
#         self.symbol = symbol
#
#     def act(self):
#         self.state.print_state()
#         key = input("Input your position:")
#         data = self.keys.index(key)
#         i = data // BOARD_COLS
#         j = data % BOARD_COLS
#         return i, j, self.symbol
#
# # The game is a zero sum game. If both players are playing with an optimal strategy, every game will end in a tie.
# # So we test whether the AI can guarantee at least a tie if it goes second.
# def play():
#     while True:
#         player1 = HumanPlayer()
#         player2 = Player(epsilon=0)
#         judger = Judger(player1, player2)
#         player2.load_policy()
#         winner = judger.play()
#         if winner == player2.symbol:
#             print("You lose!")
#         elif winner == player1.symbol:
#             print("You win!")
#         else:
#             print("It is a tie!")

train(int(1e4))
compete(int(1e3))
