import numpy as np

class Connect4:
    def __init__(self, rows=6, cols=7):
        self.rows = rows
        self.cols = cols
        self.board = np.zeros((rows, cols))
        self.turns = 0
        self.status = 0  

    def make_move(self, col: int, id: int):
        "Attempt to play piece in given col. Raise error if illegal move."

        if col in self.valid_moves():
            row = self.rows - self.col_height(col) - 1
            self.board[row][col] = id
            self.turns += 1
            self.check_winning_move(col)
        else:
            raise Exception(
                f"Illegal move. Attemped to play in column {col}. Valid moves: {self.valid_moves()}")

    def col_height(self, col):
        "Return number of game pieces played in given col"

        col = self.board[:, col]
        col = np.flip(col, axis=0)  # make index 0 bottom row

        if col[-1] != 0:  # col entirely filled
            return self.rows
        else:  # first empty index in col (num pieces played in col)
            return np.where(col == 0)[0][0]

    def valid_moves(self):
        "Return list of cols with open spaces"
        valid_cols = []
        for i in range(self.cols):
            if 0 in self.board[:, i]:
                valid_cols.append(i)
        return valid_cols

    def check_winning_move(self, col):
        "Check if most recent move (given by col) is a winning move. Update game status"
        if self.turns < 7:
            return False

        if self.turns == self.rows * self.cols:  # tie
            self.status = 'tie'
            return False

        # adjust for empty col... OR raise error here if called on empty col???
        row = self.rows - 1 - max(0, self.col_height(col) - 1)
        id = self.board[row][col]

        # vertical win
        if row + 3 >= self.rows:
            pass  # not enough pieces in col for vertical win.
        elif (self.board[row+1][col] == id and
              self.board[row+2][col] == id and
              self.board[row+3][col] == id):

            self.status = id
            return True

        # horizontal win
        left_most = max(0, col-3)
        right_most = min(self.cols-1, col+3)

        for c in range(left_most, right_most - 2):
            if (self.board[row][c] == id and
                self.board[row][c+1] == id and
                self.board[row][c+2] == id and
                    self.board[row][c+3] == id):

                self.status = id
                return True

        # negative diagonal win
        k = col - row  # diagonal offset
        neg_diag = np.diag(self.board, k=k)

        if len(neg_diag) > 3:
            for i in range(len(neg_diag) - 3):
                if (neg_diag[i] == id and
                    neg_diag[i+1] == id and
                    neg_diag[i+2] == id and
                        neg_diag[i+3] == id):

                    self.status = id
                    return True

        # positive diagonal win
        flipped_board = np.flip(self.board, axis=0)
        flipped_row = (self.rows - 1) - row
        k = col - flipped_row
        pos_diag = np.diag(flipped_board, k=k)

        if len(pos_diag) > 3:
            for i in range(len(pos_diag) - 3):
                if (pos_diag[i] == id and
                    pos_diag[i+1] == id and
                    pos_diag[i+2] == id and
                        pos_diag[i+3] == id):

                    self.status = id
                    return True

        return False

    def __repr__(self):
        "Return a string represetation of board state"
        s = ''
        for row in self.board:
            s += str(row) + '\n'
        return s

if __name__ == '__main__':
    pass