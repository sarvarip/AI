from copy import deepcopy
import os.path
#import numpy as np

WIN_REWARD = 1
DRAW_REWARD = 0
LOSS_REWARD = -1

class QLearner:

    def __init__(self, alpha=.7, gamma=0.7, initial_value=0.5, side=None): #gamma 0.7 -> 94%
        if not (0 < gamma <= 1):
            raise ValueError("An MDP must have 0 < gamma <= 1")

        self.side = side
        self.alpha = alpha
        self.gamma = gamma
        self.q_values = {}
        self.history_states = []
        self.initial_value = initial_value
        self.n = None
        # self.state = ?

    def set_side(self, side):
        self.side = side

    def Q(self, state):
        if state not in self.q_values:
            q_val = np.zeros((self.n, self.n))
            q_val.fill(self.initial_value)
            self.q_values[state] = q_val
        return self.q_values[state]

    def _select_best_move(self, go):
        state = go.encode_state(self.side)
        self.n = go.size
        n = self.n
        
        if go.n_move == 0:
            return round((n-1)/2), round((n-1)/2)
        
#         #does not seem to make it better in case of 3*3, try 5*5
#         if (go.n_move == 1 or go.n_move==2):
#             if go.board[round((n-1)/2)][round((n-1)/2)] == 0:
#                 return round((n-1)/2), round((n-1)/2)
#             else:
#                 return round((n-1)/2)+1, round((n-1)/2)
                    
        if state not in self.q_values:
            q_values = self.Q(state)
            i,j = go.absearch(self.side, 2)
            if i == "PASS":
                q_values.fill(-1)
            return i, j
        q_values = self.q_values[state]
        if q_values.max() == -1:
            return "PASS", "PASS" 
        ind = np.argmax(np.random.random(q_values.shape) * (q_values==q_values.max()))
        i = int(ind / n)
        j = ind % n
#             #debug
#             print('Step was ' + str(i) + ' and ' + str(j))
        return i, j
            
    def move(self, go, learn):
        """ make a move
        """
        i, j = self._select_best_move(go) #row, col
        if i == "PASS":
            go.set_passed(self.side)
        else: #Need to encode state BEFORE we update it with our step!!!!
            prev_state_encoded = go.encode_state(self.side)
            board = go.board
            go.previous_board = deepcopy(board)
            board[i][j] = self.side
            go.update_board(board)
            go.n_move += 1
            go.died_pieces = go.remove_died_pieces(3 - self.side) 
            if learn:
                self.history_states.append((prev_state_encoded, (i, j)))
        return go
       
    def learn(self, game_result):
        """ when games ended, this method will be called to update the qvalues
        """
        #print("learning")
        #print("states at beginning")
        #for key, value in self.q_values.items():
            #print(key + ' : ' + str(value))
        if game_result == 0:
            reward = DRAW_REWARD
        elif game_result == self.side:
            reward = WIN_REWARD
        else:
            reward = LOSS_REWARD
        self.history_states.reverse()
        max_q_value = -1.0
        for hist in self.history_states:
            state, move = hist
            q = self.Q(state)
            if max_q_value < 0:
                q[move[0]][move[1]] = reward
            else:
                q[move[0]][move[1]] = q[move[0]][move[1]] * (1 - self.alpha) + self.alpha * self.gamma * max_q_value
            max_q_value = np.max(q)
        #print(self.history_states)
        self.history_states = []

class Minmax():
    def __init__(self, depth, side=None):
        self.side = side
        self.depth = depth

    def get_input(self, go):
        n = go.size
        
        if go.n_move == 0:
            return round((n-1)/2), round((n-1)/2)
        
        #does not seem to make it better in case of 3*3, try 5*5
        if (go.n_move == 1 or go.n_move==2):
            if go.board[round((n-1)/2)][round((n-1)/2)] == 0:
                return round((n-1)/2), round((n-1)/2)
            else:
                return round((n-1)/2)+1, round((n-1)/2)
                    
        i,j = go.absearch(self.side, self.depth)
        return i, j
            
    def move(self, go, learn):
        """ make a move
        """
        i, j = self.get_input(go) #row, col
        if i == "PASS":
            go.set_passed(self.side)
        else:
            board = go.board
            go.previous_board = deepcopy(board)
            board[i][j] = self.side
            go.update_board(board)
            go.n_move += 1
            go.died_pieces = go.remove_died_pieces(3 - self.side) 
        return go
        
    def set_side(self, side):
        self.side = side

class GO:
    def __init__(self, n):
        """
        Go game.

        :param n: size of the board n*n
        """
        self.size = n
        #self.previous_board = None # Store the previous board
        self.X_move = True # X chess plays first
        self.died_pieces = [] # Intialize died pieces to be empty
        self.n_move = 0 # Trace the number of moves
        self.max_move = n * n - 1 # The max movement of a Go game
        self.komi = n/2 # Komi rule
        self.verbose = False # Verbose only when there is a manual player

    def init_board(self, n):
        '''
        Initialize a board with size n*n.

        :param n: width and height of the board.
        :return: None.
        '''
        board = [[0 for x in range(n)] for y in range(n)]  # Empty space marked as 0
        # 'X' pieces marked as 1
        # 'O' pieces marked as 2
        self.board = board
        self.previous_board = deepcopy(board)

    def set_board(self, piece_type, previous_board, board):
        '''
        Initialize board status.
        :param previous_board: previous board state.
        :param board: current board state.
        :return: None.
        '''

        # 'X' pieces marked as 1
        # 'O' pieces marked as 2

        for i in range(self.size):
            for j in range(self.size):
                if previous_board[i][j] == piece_type and board[i][j] != piece_type:
                    self.died_pieces.append((i, j))

        # self.piece_type = piece_type
        self.previous_board = previous_board
        self.board = board

    def compare_board(self, board1, board2):
        for i in range(self.size):
            for j in range(self.size):
                if board1[i][j] != board2[i][j]:
                    return False
        return True

    def copy_board(self):
        '''
        Copy the current board for potential testing.

        :param: None.
        :return: the copied board instance.
        '''
        return deepcopy(self)

    def copy_go(self):
        '''
        Copy the current board for potential testing.

        :param: None.
        :return: the copied board instance.
        '''
        return deepcopy(self)
    
    def detect_neighbor(self, i, j):
        '''
        Detect all the neighbors of a given stone.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: a list containing the neighbors row and column (row, column) of position (i, j).
        '''
        board = self.board
        neighbors = []
        # Detect borders and add neighbor coordinates
        if i > 0: neighbors.append((i-1, j))
        if i < len(board) - 1: neighbors.append((i+1, j))
        if j > 0: neighbors.append((i, j-1))
        if j < len(board) - 1: neighbors.append((i, j+1))
        return neighbors

    def detect_neighbor_ally(self, i, j):
        '''
        Detect the neighbor allies of a given stone.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: a list containing the neighbored allies row and column (row, column) of position (i, j).
        '''
        board = self.board
        neighbors = self.detect_neighbor(i, j)  # Detect neighbors
        group_allies = []
        # Iterate through neighbors
        for piece in neighbors:
            # Add to allies list if having the same color
            if board[piece[0]][piece[1]] == board[i][j]:
                group_allies.append(piece)
        return group_allies

    def ally_dfs(self, i, j):
        '''
        Using DFS to search for all allies of a given stone.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: a list containing the all allies row and column (row, column) of position (i, j).
        '''
        stack = [(i, j)]  # stack for DFS serach
        ally_members = []  # record allies positions during the search
        while stack:
            piece = stack.pop()
            ally_members.append(piece)
            neighbor_allies = self.detect_neighbor_ally(piece[0], piece[1])
            for ally in neighbor_allies:
                if ally not in stack and ally not in ally_members:
                    stack.append(ally)
        return ally_members

    def find_liberty(self, i, j):
        '''
        Find liberty of a given stone. If a group of allied stones has no liberty, they all die.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: boolean indicating whether the given stone still has liberty.
        '''
        board = self.board
        ally_members = self.ally_dfs(i, j)
        for member in ally_members:
            neighbors = self.detect_neighbor(member[0], member[1])
            for piece in neighbors:
                # If there is empty space around a piece, it has liberty
                if board[piece[0]][piece[1]] == 0:
                    return True
        # If none of the pieces in a allied group has an empty space, it has no liberty
        return False

    def find_died_pieces(self, piece_type):
        '''
        Find the died stones that has no liberty in the board for a given piece type.

        :param piece_type: 1('X') or 2('O').
        :return: a list containing the dead pieces row and column(row, column).
        '''
        board = self.board
        died_pieces = []

        for i in range(len(board)):
            for j in range(len(board)):
                # Check if there is a piece at this position:
                if board[i][j] == piece_type:
                    # The piece die if it has no liberty
                    if not self.find_liberty(i, j):
                        died_pieces.append((i,j))
        return died_pieces

    def remove_died_pieces(self, piece_type):
        '''
        Remove the dead stones in the board.

        :param piece_type: 1('X') or 2('O').
        :return: locations of dead pieces.
        '''

        died_pieces = self.find_died_pieces(piece_type)
        if not died_pieces: return []
        self.remove_certain_pieces(died_pieces)
        return died_pieces

    def remove_certain_pieces(self, positions):
        '''
        Remove the stones of certain locations.

        :param positions: a list containing the pieces to be removed row and column(row, column)
        :return: None.
        '''
        board = self.board
        for piece in positions:
            board[piece[0]][piece[1]] = 0
        self.update_board(board)

    def place_chess(self, i, j, piece_type):
        '''
        Place a chess stone in the board.

        :param i: row number of the board.
        :param j: column number of the board.
        :param piece_type: 1('X') or 2('O').
        :return: boolean indicating whether the placement is valid.
        '''
        board = self.board

        valid_place = self.valid_place_check(i, j, piece_type)
        if not valid_place:
            return False
        self.previous_board = deepcopy(board)
        board[i][j] = piece_type
        self.update_board(board)
        # Remove the following line for HW2 CS561 S2020
        # self.n_move += 1
        return True

    def valid_place_check(self, i, j, piece_type, test_check=False):
        '''
        Check whether a placement is valid.

        :param i: row number of the board.
        :param j: column number of the board.
        :param piece_type: 1(white piece) or 2(black piece).
        :param test_check: boolean if it's a test check.
        :return: boolean indicating whether the placement is valid.
        '''   
        board = self.board
        verbose = self.verbose
        if test_check:
            verbose = False

        # Check if the place is in the board range
        if not (i >= 0 and i < len(board)):
            if verbose:
                print(('Invalid placement. row should be in the range 1 to {}.').format(len(board) - 1))
            return False
        if not (j >= 0 and j < len(board)):
            if verbose:
                print(('Invalid placement. column should be in the range 1 to {}.').format(len(board) - 1))
            return False
        
        # Check if the place already has a piece
        if board[i][j] != 0:
            if verbose:
                print('Invalid placement. There is already a chess in this position.')
            return False
        
        # Copy the board for testing
        test_go = self.copy_board()
        test_board = test_go.board

        # Check if the place has liberty
        test_board[i][j] = piece_type
        test_go.update_board(test_board)
        if test_go.find_liberty(i, j):
            return True

        # If not, remove the died pieces of opponent and check again
        test_go.remove_died_pieces(3 - piece_type)
        if not test_go.find_liberty(i, j):
            if verbose:
                print('Invalid placement. No liberty found in this position.')
            return False

        # Check special case: repeat placement causing the repeat board state (KO rule)
        else:
            if self.died_pieces and self.compare_board(self.previous_board, test_go.board):
                if verbose:
                    print('Invalid placement. A repeat move not permitted by the KO rule.')
                return False
        return True
        
    def update_board(self, new_board):
        '''
        Update the board with new_board

        :param new_board: new board.
        :return: None.
        '''   
        self.board = new_board

    def visualize_board(self):
        '''
        Visualize the board.

        :return: None
        '''
        board = self.board

        print('-' * len(board) * 2)
        for i in range(len(board)):
            for j in range(len(board)):
                if board[i][j] == 0:
                    print(' ', end=' ')
                elif board[i][j] == 1:
                    print('X', end=' ')
                else:
                    print('O', end=' ')
            print()
        print('-' * len(board) * 2)

    # def game_end(self, piece_type, action="MOVE"):
        # '''
        # Check if the game should end.

        # :param piece_type: 1('X') or 2('O').
        # :param action: "MOVE" or "PASS".
        # :return: boolean indicating whether the game should end.
        # '''

        # # Case 1: max move reached
        # if self.n_move >= self.max_move:
            # return True
        # # Case 2: two players all pass the move.
        # if self.compare_board(self.previous_board, self.board) and action == "PASS":
            # return True
        # return False

    def score(self, piece_type):
        '''
        Get score of a player by counting the number of stones.

        :param piece_type: 1('X') or 2('O').
        :return: boolean indicating whether the game should end.
        '''

        board = self.board
        cnt = 0
        for i in range(self.size):
            for j in range(self.size):
                if board[i][j] == piece_type:
                    cnt += 1
        return cnt          

    def judge_winner(self):
        '''
        Judge the winner of the game by number of pieces for each player.

        :param: None.
        :return: piece type of winner of the game (0 if it's a tie).
        '''        

        cnt_1 = self.score(1)
        cnt_2 = self.score(2)
        if cnt_1 > cnt_2 + self.komi: return 1
        elif cnt_1 < cnt_2 + self.komi: return 2
        else: return 0
        
    def encode_state(self, piece_type):
        """ encode the current state of the board as a string
        50 long array with 1 or 0
        first 25: location of own stones
        last 25: location of opponent's stones
        """
        board = self.board
        n = self.size
        numsquares = n*n
        planes = [0]*numsquares
        for i in range(n):
            for j in range(n):
                if board[i][j] == piece_type:
                    idx = i*n + j
                    planes[idx] = 1
                elif board[i][j] == (3-piece_type):
                    idx = i*n + j
                    planes[idx] = -1
        return ''.join([str(i) for i in planes])
        
    def heuristic(self, piece_type): #by def max is the one from where the search starts (whose turn it is)

        cnt_1 = self.score(1)
        cnt_2 = self.score(2)
        if piece_type == 1:
            return cnt_1 - cnt_2 - self.komi
        if piece_type == 2:
            return cnt_2 - cnt_1 + self.komi
    
    def organize_tree(self, piece_type):
        n = self.size
        board = self.board
        empty = []
        enemy_stones = []
        own_stones = []
        for i in range(n):
            for j in range(n):
                if board[i][j] == 0:
                    empty.append((i,j))
                elif board[i][j] == piece_type:
                    own_stones.append((i,j))
                else:
                    enemy_stones.append((i,j))
        border_enemy = []
        close_friendly = []
        for t in enemy_stones:
            i = t[0]
            j = t[1]
            border_enemy = border_enemy + [(i+1,j), (i-1,j), (i, j+1), (i, j-1)]
        for t in own_stones:
            i = t[0]
            j = t[1]
            close_friendly = close_friendly + [(i+1,j), (i-1,j), (i, j+1), (i, j-1)]
            
        border_enemy = [x for x in border_enemy if x in empty]
        close_friendly = [x for x in close_friendly if x in empty]
        seq = border_enemy + close_friendly + empty
        seen = set()
        return [x for x in seq if not (x in seen or seen.add(x))]
        
    def absearch(self, piece_type, max_depth):
        #piece type denotes piece type for from where the search started
        #no need to worry about other person passing, since as soon as we pass, we stop the search
        #so we don't know if other person will pass or not, they will just have to make their move
        iters = 0
        alpha = -10000
        beta = 10000
        val, a1, a2 = self.max_heuristic(piece_type, iters, alpha, beta, max_depth)
        return a1, a2

    def max_heuristic(self, piece_type, iters, alpha, beta, max_depth): #max is first player from wherever the searchstarts (could be black or white)

        if (iters >= max_depth) or (self.n_move >= self.max_move): #maxiter is 4, OR STEPS IS MAX!!!
            return self.heuristic(piece_type), 0, 0 #does not matter what we return as second and third        
        
        n = self.size
        side = piece_type #max is the first, 3-piece_type for min
        max_value = -10000
        for t in self.organize_tree(side): #random.sample(range(n), n)
            i = t[0]
            j = t[1]
            if self.valid_place_check(i, j, side, test_check=False): 
                test_go = self.copy_go()
                test_board = test_go.board
                test_board[i][j] = side
                test_go.update_board(test_board)
                test_go.n_move = self.n_move + 1
                dp = test_go.remove_died_pieces(3 - side)
                cur_val, ac1, ac2 = test_go.min_heuristic(piece_type, iters+1, alpha, beta, max_depth)
                if cur_val > max_value:
                    if cur_val >= beta: #pruning, note beta starts from infinity (10000)
                        return cur_val, i, j
                    max_value = cur_val
                    a1 = i
                    a2 = j
            alpha = max(alpha, max_value)
        if max_value == -10000: #all actions were invalid
            a1 = "PASS"
            a2 = "PASS"
        return max_value, a1, a2

    def min_heuristic(self, piece_type, iters, alpha, beta, max_depth): 

        if (iters >= max_depth) or (self.n_move >= self.max_move): #maxiter is 4, OR STEPS IS MAX!!!
            return self.heuristic(piece_type), 0, 0

        n = self.size
        side = 3-piece_type #max is the first, 3-piece_type for min
        min_value = 10000
        for t in self.organize_tree(side):
            i = t[0]
            j = t[1]
            if self.valid_place_check(i, j, side, test_check=False): 
                test_go = self.copy_go()
                test_board = test_go.board
                test_board[i][j] = side
                test_go.update_board(test_board)
                test_go.n_move = self.n_move + 1
                dp = test_go.remove_died_pieces(3 - side)
                cur_val, ac1, ac2 = test_go.max_heuristic(piece_type, iters+1, alpha, beta, max_depth)
                if cur_val < min_value:
                    if cur_val <= alpha: #pruning, note alpha starts from negative infinity (10000)
                        return cur_val, i, j
                    min_value = cur_val
                    a1 = i
                    a2 = j
            beta = min(beta, min_value) #can be done within if
        if min_value == 10000: #all actions were invalid
            a1 = "PASS"
            a2 = "PASS"
        return min_value, a1, a2
        
    def getCount(self, path="count.txt"):
        if os.path.exists(path):
            with open(path, 'r+') as f:
                if sum(sum(self.board, [])) == 0:
                    f.seek(0)
                    f.truncate(0)
                    f.write(str(1)) #after our move it will be one
                    self.n_move = 0
                elif sum(sum(self.board, [])) == 1:
                    f.seek(0)
                    f.truncate(0)
                    f.write(str(2)) #after our move it will be two
                    self.n_move = 1
                else:
                    lins = f.readlines()
                    nummoves = lins[0]
                    self.n_move = int(nummoves) + 1 #need to increment by one, because opposition does not increment it
                    f.seek(0)
                    f.truncate(0)
                    f.write(str(self.n_move + 1)) #after our move it will be one more
        else:
            with open('count.txt', 'w+') as f:
                if sum(sum(self.board, [])) == 0:
                    f.write(str(1)) #after our move it will be one
                    self.n_move = 0
                elif sum(sum(self.board, [])) == 1:
                    f.write(str(2)) #after our move it will be two
                    self.n_move = 1
        

def play(go, player1, player2, learn, show_steps=False):
    """ Player 1 -> X, X goes first
        player 2 -> O
    """
    while not go.game_end(show_steps):
        go = player1.move(go,learn)
        if show_steps == True:
            go.visualize_board()
        if not go.game_end(show_steps):
            go = player2.move(go,learn)
            if show_steps == True:
                go.visualize_board()
    
    game_result = go.judge_winner()

    if learn == True:
        player1.learn(game_result)
        player2.learn(game_result)

    return game_result
        
def battle(player1, player2, iternum, learn=False, show_result=True, show_steps=False):
    player1.set_side(PLAYER_X)
    player2.set_side(PLAYER_O)
    p1_stats = [0, 0, 0] # draw, win, lose
    for i in range(0, iternum):
        #print(i)
        N = 5
        go = GO(N)
        go.init_board()
        result = play(go, player1, player2, learn, show_steps)
        p1_stats[result] += 1
        if i % 1 == 0:
            print(i)

    p1_stats = [round(x / iternum * 100.0, 1) for x in p1_stats]
    if show_result:
        print('_' * 60)
        print('{:>15}(X) | Wins:{}% Draws:{}% Losses:{}%'.format(player1.__class__.__name__, p1_stats[1], p1_stats[0], p1_stats[2]).center(50))
        print('{:>15}(O) | Wins:{}% Draws:{}% Losses:{}%'.format(player2.__class__.__name__, p1_stats[2], p1_stats[0], p1_stats[1]).center(50))
        print('_' * 60)
        print()

    return p1_stats

def readInput(n, path="input.txt"):

    with open(path, 'r') as f:
        lins = f.readlines()

        side = int(lins[0])

        prev_board = [[int(x) for x in line.rstrip('\n')] for line in lins[1:n+1]]
        board = [[int(x) for x in line.rstrip('\n')] for line in lins[n+1: 2*n+1]]

        return side, prev_board, board
    
def writeOutput(i, j, filepath="output.txt"):
    out = ""
    if i == "PASS":
    	out = "PASS"
    else:
	    out += str(i) + ',' + str(j)

    with open(filepath, 'w') as f:
        f.write(out)

if __name__ == "__main__":
    N = 5
    piece_type, previous_board, board = readInput(N)
    go = GO(N)
    go.set_board(piece_type, previous_board, board)
    go.getCount()
    player = Minmax(5, piece_type)
    i,j = player.get_input(go)
    writeOutput(i,j)
