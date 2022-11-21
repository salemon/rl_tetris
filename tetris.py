"""
Tetris Simulator

Author - Anqi Li (anqil4@cs.washington.edu)
Adapted from the java simulator from Drew Bagnell's
course at Carnegie Mellon University

"""


import gym
from gym.utils import seeding
import numpy as np
import copy 

class TetrisState:
    """
    the tetris state
    """
    def __init__(self, field, top, next_piece, lost, turn, cleared):
        # the board configuration
        self.field = field
        # the top position
        self.top = top
        # the piece ID of the next piece
        self.next_piece = next_piece
        # whether the game has lost
        self.lost = lost
        # the current turn
        self.turn = turn
        # the number of rows cleared so far
        self.cleared = cleared        

    def copy(self):
        return TetrisState(
            self.field.copy(),
            self.top.copy(),
            self.next_piece,
            self.lost,
            self.turn,
            self.cleared
                    )


class TetrisEnv(gym.Env):
    metadata = {
        'render.modes': ['ascii']
    }

    def __init__(self):
        print('********')
        self.n_cols = 10
        self.n_rows = 21
        self.n_pieces = 7
        self.state_size = 7
        self.score = 0
        self.game_end = False
        
        # the next several lists define the piece vocabulary in detail
        # width of the pieces [piece ID][orientation]
        # pieces: O, I, L, J, T, S, Z
        self.piece_orients = [1, 2, 4, 4, 4, 2, 2]
        self.piece_width = [
            [2],
            [1, 4],
            [2, 3, 2, 3],
            [2, 3, 2, 3],
            [2, 3, 2, 3],
            [3, 2],
            [3, 2]
        ]
        # height of pieces [piece ID][orientation]
        self.piece_height = [
            [2],
            [4, 1],
            [3, 2, 3, 2],
            [3, 2, 3, 2],
            [3, 2, 3, 2],
            [2, 3],
            [2, 3]
        ]
        self.piece_bottom = [
            [[0, 0]],
            [[0], [0, 0, 0, 0]],
            [[0, 0], [0, 1, 1], [2, 0], [0, 0, 0]],
            [[0, 0], [0, 0, 0], [0, 2], [1, 1, 0]],
            [[0, 1], [1, 0, 1], [1, 0], [0, 0, 0]],
            [[0, 0, 1], [1, 0]],
            [[1, 0, 0], [0, 1]]
        ]
        self.piece_top = [
            [[2, 2]],
            [[4], [1, 1, 1, 1]],
            [[3, 1], [2, 2, 2], [3, 3], [1, 1, 2]],
            [[1, 3], [2, 1, 1], [3, 3], [2, 2, 2]],
            [[3, 2], [1, 2, 1], [2, 3], [1, 2, 1]],
            [[1, 2, 2], [3, 2]],
            [[2, 2, 1], [2, 3]]
        ]
        self.piece_real_weight = [
            [[2, 2]],
            [[1,1,1,1],[4]],
            [[2,1,1],[1,3],[1,1,2],[3,1]],
            [[2,1,1],[3,1],[1,1,2],[1,3]],
            [[1,2,1],[1,3],[1,2,1],[3,1]],
            [[2,2],[1,2,1]],
            [[2,2],[1,2,1]]
        ]

        # initialize legal moves for all pieces
        self.legal_moves = []
        for i in range(self.n_pieces):
            piece_legal_moves = []
            for j in range(self.piece_orients[i]):
                for k in range(self.n_cols + 1 - self.piece_width[i][j]):
                    piece_legal_moves.append([j, k])
            self.legal_moves.append(piece_legal_moves)

        self.state = None
        self.cleared_current_turn = 0

    def seed(self, seed=None):
        """
        set the random seed for the environment
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """
        make a move based on the orientation and slot
        """
        orient, slot = action
        self.state.turn += 1

        # height of the field
        height = max(
            self.state.top[slot+c] - self.piece_bottom[self.state.next_piece][orient][c]
            for c in range(self.piece_width[self.state.next_piece][orient])
        )

        # check if game ended
        if height + self.piece_height[self.state.next_piece][orient] >= self.n_rows:
            self.state.lost = True
            self.score += self._get_reward()
            return self.state, self._get_reward(), True, 0

        # for each column in the piece - fill in the appropriate blocks
        for i in range(self.piece_width[self.state.next_piece][orient]):
            # from bottom to top of brick
            for h in range(height + self.piece_bottom[self.state.next_piece][orient][i], height + self.piece_top[self.state.next_piece][orient][i]):
                self.state.field[h, i+slot] = self.state.turn

        # adjust top
        for c in range(self.piece_width[self.state.next_piece][orient]):
            self.state.top[slot+c] = height + self.piece_top[self.state.next_piece][orient][c]

        # check for full rows - starting at the top
        self.cleared_current_turn = 0
        for r in range(height + self.piece_height[self.state.next_piece][orient] - 1, height - 1, -1):
            # if the row was full - remove it and slide above stuff down
            if np.all(self.state.field[r] > 0):
                self.cleared_current_turn += 1
                self.state.cleared += 1
                # for each column
                for c in range(self.n_cols):
                    # slide down all bricks
                    self.state.field[r:self.state.top[c], c] = self.state.field[(r+1):(self.state.top[c]+1), c]
                    # lower the top
                    self.state.top[c] -= 1
                    while self.state.top[c] >= 1 and self.state.field[self.state.top[c]-1, c] == 0:
                        self.state.top[c] -= 1

        self.score += self.cleared_current_turn
        # pick a new piece
        self.state.next_piece = self._get_random_piece()
        return self.state.copy(), self._get_reward(), False, self.cleared_current_turn



    def reset(self):
        lost = False
        turn = 0
        cleared = 0

        field = np.zeros((self.n_rows, self.n_cols), dtype=np.int)
        top = np.zeros(self.n_cols, dtype=np.int)
        next_piece = self._get_random_piece()
        self.state = TetrisState(field, top, next_piece, lost, turn, cleared)
        self.score = 0
        return self.get_features(np.flip(copy.deepcopy(field)),self.state.top,0,0,False)

    def render(self, mode='ascii'):
        print('\nThe wall:')
        print('-' * (2 * self.n_cols + 1))
        for r in range(self.n_rows - 1, -1, -1):
            render_string = '|'
            for c in range(self.n_cols):
                if self.state.field[r, c] > 0:
                    render_string += '*|'
                else:
                    render_string += ' |'
            render_string += ''
            print(render_string)
        print('-' * (2 * self.n_cols + 1))

        print('\nThe next piece:')
        if self.state.next_piece == 0:
            print('**\n**')
        elif self.state.next_piece == 1:
            print('****')
        elif self.state.next_piece == 2:
            print('*\n*\n**')
        elif self.state.next_piece == 3:
            print(' *\n *\n**')
        elif self.state.next_piece == 4:
            print(' * \n***')
        elif self.state.next_piece == 5:
            print(' **\n**')
        elif self.state.next_piece == 6:
            print('**\n **')



    def close(self):
        pass

    def _get_random_piece(self):
        """
        return an random integer 0-6
        """
        return np.random.randint(self.n_pieces)

    def _get_reward(self):
        """
        reward function
        """
        return 1+self.score
        # return 0.0

    def get_actions(self):
        """
        gives the legal moves for the next piece
        :return:
        """
        return self.legal_moves[self.state.next_piece]

    def set_state(self, state):
        """
        set the field and the next piece
        """
        self.state = state.copy()

    def clear_lines(self,field,orient,slot):
        
        lines_to_clear = [index for index, row in enumerate(field) if np.all(field[index] > 0)]
        eroded = 0
        clear_piece = 0
        if lines_to_clear != []:
            for i in range(self.piece_height[self.state.next_piece][orient]):
                if (self.n_rows-(self.state.top[slot]+(self.piece_height[self.state.next_piece][orient]-(i+1)))) in lines_to_clear:
                    clear_piece += (self.piece_real_weight[self.state.next_piece][orient][i])
            eroded = len(lines_to_clear) * clear_piece
            field = [row for index, row in enumerate(field) if index not in lines_to_clear]
            # Add new lines at the top
            for _ in lines_to_clear:
                field.insert(0, [0 for _ in range(self.n_cols)])

        return len(lines_to_clear),eroded, np.array(field)



    def get_wells_height(self, field):
        WIDTH = 10
        wells_height = 0
        HEIGHT = 21
        for x in range(1, WIDTH-1):
            for y in range(HEIGHT):
                if field[y][x] == 0 and field[y][x-1] != 0 \
                        and field[y][x+1] != 0:
                    wells_height += 1
                    for _y in range(y+1, 21):
                        if field[_y][x] != 0:
                            break
                        wells_height += 1

        for y in range(HEIGHT):
            # check wells in the leftmost boarder of the board
            if field[y][0] == 0 and field[y][1] != 0:
                wells_height += 1
                for _y in range(y+1, 21):
                    if field[_y][x] != 0:
                        break
                    wells_height += 1

            # check wells in the rightmost border of the board
            if field[y][WIDTH-1] == 0 and field[y][WIDTH-2] != 0:
                wells_height += 1
                for _y in range(y+1, 21):
                    if field[_y][x] != 0:
                        break
                    wells_height += 1

        return wells_height

    def get_holes_at_square(self, field,row_index,col_index):
        row_index = 0
        holes_at_square = 1

        while row_index < len(field) - 1:
            if field[row_index][col_index] == 0:
                holes_at_square += 1
                row_index += 1
            else:
                break

        return holes_at_square

    def get_holes(self, field):
       
        holes = 0

        for row_index, row in enumerate(field):
            if row_index == 0: continue
            for col_index, square in enumerate(row):
                if square == 0 and field[row_index - 1][col_index] > 0:
                    holes += self.get_holes_at_square(field, row_index, col_index)

        return holes

    def get_bumpiness(self,field):
        total_bumpiness = 0
        max_bumpiness = 0
        min_ys = []

        for col in zip(*field):
            i = 0
            while i < self.n_rows and col[i] ==0:
                i += 1
            min_ys.append(i)
        
        for i in range(len(min_ys) - 1):
            bumpiness = abs(min_ys[i] - min_ys[i+1])
            max_bumpiness = max(bumpiness, max_bumpiness)
            total_bumpiness += abs(min_ys[i] - min_ys[i+1])

        return total_bumpiness, max_bumpiness

    def get_height(self,field):
        sum_height = 0
        max_height = 0
        min_height = self.n_rows

        for col in zip(*field):
            i = 0
            while i < self.n_rows and col[i] == 0:
                i += 1
            height = self.n_rows - i
            sum_height += height
            if height > max_height:
                max_height = height
            elif height < min_height:
                min_height = height

        return sum_height, max_height, min_height

    def get_row_transitions(self,field):
        total = 0
        for r in range(self.n_rows):
            row_count = 0
            last_empty = False
            for c in range(self.n_cols):
                empty = field[r][c] == 0
                if last_empty != empty:
                    row_count += 1
                    last_empty = empty
            if last_empty:
                row_count += 1
            if last_empty and row_count == 2:
                continue
            total += row_count
        return total

    def get_cumulative_wells(self,field):
        wells = [0 for i in range(self.n_cols)]
        for y, row in enumerate(field):
            left_empty = True
            for x, code in enumerate(row):
                if code == 0:
                    well = False
                    right_empty = self.n_cols > x + 1 >= 0 and field[y][x + 1] == 0
                    if left_empty or right_empty:
                        well = True
                    wells[x] = 0 if well else wells[x] + 1
                    left_empty = True
                else:
                    left_empty = False
        return sum(wells)

    def get_col_transitions(self,field):
        total = 0
        for c in range(self.n_cols):
            column_count = 0
            last_empty = False
            for r in reversed(range(self.n_rows)):
                empty = field[r][c] == 0
                if last_empty and not empty:
                    column_count += 2
                last_empty = empty
            if last_empty and column_count == 1:
                continue
            total += column_count
        return total

    def get_row_holes(self, field):
        row_holes = 0

        for row_index, row in enumerate(field):
            if row_index == 0: continue
            for col_index, square in enumerate(row):
                if square == 0:
                    row_holes+=row_holes
                    break
        return row_holes
    
    def get_hole_count(self,field):
        hole_count = 0
        for x in range(self.n_cols):
            below = False
            for y in range(self.n_rows):
                empty = field[y][x] == 0
                if not below and not empty:
                    below = True
                elif below and empty:
                    hole_count += 1

        return hole_count

    def get_features(self,field,top,orient,slot,start):
        landheight = 0
        max_height=0
        eroded = 0
        col_trans = 0
        row_trans = 0
        holes = 0
        well = 0
        lines = 0
        total_bumpiness = 0
        hole_count = 0
        top_position = 0
        depth=0
        sum_height = 0
        if start:
            for i in range(self.piece_width[self.state.next_piece][orient]):
                top_position = max(top_position,self.state.top[slot+i])
                landheight = max(landheight,self.piece_top[self.state.next_piece][orient][i])
            
            landheight = top_position+landheight
            lines,eroded,field = self.clear_lines(field,orient,slot)
            total_bumpiness, max_bumpiness = self.get_bumpiness(field)

            col_trans = self.get_col_transitions(field)
            row_trans = self.get_row_transitions(field)

            well = self.get_wells_height(field)
            sum_height, max_height, min_height = self.get_height(field)
            hole_count = self.get_hole_count(field)
            depth = self.get_row_holes(field)
        top_list = top.tolist()
        diff_list = [abs(top_list[j]-top_list[j+1]) for j in range(len(top_list)-1) ]
        result = [landheight,eroded,row_trans,col_trans,hole_count,well,depth]
        return result

    def get_next_states(self):
        next_states = {}
        legal_moves = self.get_actions()
        for orient, slot in legal_moves:
            out_flag = 0
            height = max(
            self.state.top[slot+c] - self.piece_bottom[self.state.next_piece][orient][c]
            for c in range(self.piece_width[self.state.next_piece][orient])
            )

            field = copy.deepcopy(self.state.field)
            # for each column in the piece - fill in the appropriate blocks
            for i in range(self.piece_width[self.state.next_piece][orient]):
                # from bottom to top of brick
                for h in range(height + self.piece_bottom[self.state.next_piece][orient][i], height + self.piece_top[self.state.next_piece][orient][i]):
                    if h >= self.n_cols:
                        out_flag = 1
                        break
                    field[h, i+slot] = self.state.turn + 1
                if out_flag == 1:
                    break
            if out_flag == 1:
                next_states[(orient, slot)] = self.get_features(np.flip(field),self.state.top,orient,slot,True)
            if out_flag ==0 :
                next_states[(orient, slot)] = self.get_features(np.flip(field),self.state.top,orient,slot,True)

        return next_states

    def get_state_size(self):
        return self.state_size

    def get_score(self):
        return self.score
            
            

if __name__ == "__main__":

    # run a random policy on the tetris simulator

    # np.random.seed(1)
    env = TetrisEnv()
    env.reset()
    #env.render()
    

    for _ in range(10):

        actions = env.get_actions()
        action = actions[np.random.randint(len(actions))]
        orient, slot = action
        action_state_dict = env.get_next_states()
        print(env.get_features(np.flip(env.state.field),env.state.top,orient,slot,True))
        state, reward, done, _ = env.step(action)
        env.render()
        if done:
            break





