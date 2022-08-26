import numpy as np


class State:
    def __init__(self, BOARD_ROWS_COLS=3):
        # the board is represented by an n * n array,
        # 1 represents a chessman of the player who moves first,
        # -1 represents a chessman of another player
        # 0 represents an empty position
        # 记录一下每一个井字棋的每一个State，原先的状态是一个数组，我们这里将其映射为hash，由于每一个位置的值是
        # 1，-1，0， 分别标志 第一个人，第二个人，空 的状态，把他们加1后可以看作是一个三进制，然后将其转为10进制即可作为唯一的hash值
        self.rows_cols = BOARD_ROWS_COLS
        self.data = np.zeros((self.rows_cols, self.rows_cols), dtype=int)
        self.state_unused = [[i, j] for i in range(self.rows_cols) for j in range(self.rows_cols)]
        self.winner = None
        self.hash_val = None
        self.end = None

    # compute the hash value for one state, it's unique
    def hash(self):
        if self.hash_val is None:
            self.hash_val = 0
            for i in range(self.rows_cols):
                for j in range(self.rows_cols):
                    self.hash_val = self.hash_val * 3 + self.data[i][j] + 1
        return self.hash_val

    # check whether a player has won the game, or it's a tie
    def is_end(self):
        if self.end is not None:
            return self.end
        res = []
        # 检查一下每一行每一列
        # check row
        for i in range(self.rows_cols):
            res.append(np.sum(self.data[i, :]))
        # check columns
        for i in range(self.rows_cols):
            res.append(np.sum(self.data[:, i]))

        # 检查对角线
        # check diagonals
        diag1, diag2 = 0, 0
        for i in range(self.rows_cols):
            diag1 += self.data[i][i]
            diag2 += self.data[i][self.rows_cols - i - 1]
        res.append(diag1)
        res.append(diag2)

        for flag in res:
            if flag == self.rows_cols:
                self.winner = 1
                self.end = True
                return self.end
            if flag == -self.rows_cols:
                self.winner = -1
                self.end = True
                return self.end

        # whether it's a tie
        if len(self.state_unused) == 0:
            self.winner = 0
            self.end = True
            return self.end
        # game is still going on
        self.end = False
        return self.end

    # @symbol: 1 or -1
    # put chessman symbol in position [i, j]
    # 在 i，j 位置处下棋，并将 i，j 从 state_unused 中 remove
    def next_state(self, i, j, symbol):
        new_state = State()
        new_state.data = self.data.copy()
        new_state.data[i, j] = symbol
        new_state.state_unused = self.state_unused.copy()
        new_state.state_unused.remove([i, j])
        return new_state

    # print the board
    def print_state(self):
        for i in range(self.rows_cols):
            print('-------------')
            out = '| '
            for j in range(self.rows_cols):
                if self.data[i, j] == 1:
                    token = '*'
                elif self.data[i, j] == -1:
                    token = 'x'
                else:
                    token = '0'
                out += token + ' | '
            print(out)
        print('-------------')


# 递归的获取所有状态的 hash is_end state_unused
def get_all_states_impl(current_state, current_symbol, all_states):
    print(len(all_states), current_state.data)
    for i in range(current_state.rows_cols):
        for j in range(current_state.rows_cols):
            if current_state.data[i][j] == 0:
                new_state = current_state.next_state(i, j, current_symbol)
                new_hash = new_state.hash()
                if new_hash not in all_states:
                    is_end = new_state.is_end()
                    state_unused = new_state.state_unused
                    all_states[new_hash] = (new_state, is_end, state_unused)
                    if not is_end:
                        get_all_states_impl(new_state, -current_symbol, all_states)


def get_all_states():
    current_symbol = 1
    current_state = State()
    all_states = dict()
    all_states[current_state.hash()] = (current_state, current_state.is_end(), current_state.state_unused)
    get_all_states_impl(current_state, current_symbol, all_states)
    return all_states


# all possible board configurations
all_states = get_all_states()
