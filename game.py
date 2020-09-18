# -*- coding: utf-8 -*-
"""
@author: Junxiao Song
"""

from __future__ import print_function
import numpy as np
import itertools


class Board(object):
    """board for the game"""

    'state[where_to_put]= koma_int'
    'availables_koma_int = koma_int'

    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 4))
        self.height = int(kwargs.get('height', 4))
        # board states stored as a dict,
        # key: move as location on the board,
        # value: player as pieces type
        self.states = {}
        # need how many pieces in a row to win
        self.n_in_row = int(kwargs.get('n_in_row', 4))
        self.players = [1, 2]  # player1 and player2
        
    def init_board(self, start_player=0):
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width and height can not be '
                            'less than {}'.format(self.n_in_row))
        self.current_player = self.players[start_player]  # start player
        # keep available moves in a list
        self.availables_field = list(range(self.width * self.height))
        #self.availables_koma = [list(v)for v in set(itertools.permutations([-1,-1,-1,-1,1,1,1,1],4))]:
        self.availables_koma_int = list(range(16))
        
        self.states = {}
        self.last_move = []
        self.kept_koma_int = -1

    def where_to_put_to_location(self, where_to_put):
        """
        3*3 board's moves like:
        6 7 8
        3 4 5
        0 1 2
        and move 5's location is (1,2)
        """
        h = where_to_put // self.width
        w = where_to_put % self.width
        return [h, w]

    def location_to_where_to_put(self, location):
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        where_to_put = h * self.width + w
        if where_to_put not in range(self.width * self.height):
            return -1
        return where_to_put
    
    def koma_int_to_koma_one(self, koma_int):
        if koma_int == -1:
            return [0,0,0,0]

        arr_bin = [str(c) for c in bin(int(koma_int))][2:]
        tmp_arr = [int(x.replace("0", "-1")) for x in arr_bin]
        return_arr = [-1,-1,-1,-1]
        if len(tmp_arr) < 4:
            for i in range(len(tmp_arr)):
                return_arr[4 - len(tmp_arr) + i] = tmp_arr[i]
            return return_arr
        return tmp_arr  

    def koma_one_to_koma_int(self, koma_one):
        if koma_one == -1 or koma_one == [0,0,0,0]:
            return -1

        arr_bin = (np.array(koma_one) + np.array([1,1,1,1]) )/2
        tmp = np.array([8, 4, 2, 1])
        return np.sum(arr_bin * tmp)

    def current_state(self):
        """return the board state from the perspective of the current player.
        state shape: 4*width*height*4
        """
        #ここだけkoma_oneを使う。
        square_state = np.zeros((7, self.width, self.height))
        if self.states:
            where_to_put, koma_ints = np.array(list(zip(*self.states.items())))
            for i in range(len(where_to_put)):
                koma_one = self.koma_int_to_koma_one(koma_ints[i])
                square_state[0][int(where_to_put[i] // self.width),
                                int(where_to_put[i] % self.height)] = koma_one[0]
                square_state[1][int(where_to_put[i] // self.width),
                                int(where_to_put[i] % self.height)] = koma_one[1]
                square_state[2][int(where_to_put[i] // self.width),
                                int(where_to_put[i] % self.height)] = koma_one[2]
                square_state[3][int(where_to_put[i] // self.width),
                                int(where_to_put[i] % self.height)] = koma_one[3]
                            
            # indicate the last move location
            square_state[4][int(self.last_move[0] // self.width),
                                int(self.last_move[0] % self.height)] = self.last_move[1]

            square_state[5][:] = self.koma_int_to_koma_one(self.kept_koma_int)
            if len(self.states) % 2 == 0:
                square_state[6][:,:] = [1,1,1,1]  # indicate the colour to play
        # TODO -1の意味を理解する
        return square_state[:, ::-1, :]
    
    def do_move(self,move):
        # 昔 move: int
        # 今 move: [int,int]
        where_to_put,koma_int = move
        self.states[where_to_put] = self.kept_koma_int
        self.availables_field.remove(where_to_put)
        self.availables_koma_int.remove(koma_int)
        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )
        self.last_move = move
        self.kept_koma_int = koma_int

    def do_last_move(self):
        assert len(self.availables_field) == 1 and len(self.availables_koma_int) == 0
        last_field = self.availables_field[0]
        self.states[last_field] = self.kept_koma_int
        self.availables_field.remove(last_field)
        self.last_move = (last_field, -1)
        self.kept_koma_int = -1

        

    def has_a_winner(self):
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        moved = list(set(range(width * height)) - set(self.availables_field))
        if len(moved) < self.n_in_row -1:
            return False

        for m in moved:
            h = m // width
            w = m % width
            # koma = states[m]

            if w in range(width - n + 1):
                arr = np.array([self.koma_int_to_koma_one(states.get(i, -1)) for i in range(m, m + n)])
                arr = np.sum(arr, axis=0)
                if 4 in arr or - 4 in arr:
                    return True

            if h in range(height - n + 1):
                arr = np.array([self.koma_int_to_koma_one(states.get(i, -1)) for i in range(m, m + n * width, width)])
                arr = np.sum(arr, axis=0)
                if 4 in arr or - 4 in arr:
                    return True

            if w in range(width - n + 1) and h in range(height - n + 1):
                arr = np.array([self.koma_int_to_koma_one(states.get(i, -1)) for i in range(m, m + n * (width + 1), width + 1)])
                arr = np.sum(arr, axis=0)
                if 4 in arr or - 4 in arr:
                    return True

            if w in range(n - 1, width) and h in range(height - n + 1):
                arr = np.array([self.koma_int_to_koma_one(states.get(i, -1)) for i in range(m, m + n * (width - 1), width - 1)])
                arr = np.sum(arr, axis=0)
                if 4 in arr or - 4 in arr:
                    return True

        return False

    def game_end(self):
        """Check whether the game is ended or not"""
        win = self.has_a_winner()
        if win:
            return True, self.current_player
        elif not len(self.availables_field):
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player


class Game(object):
    """game server"""

    def __init__(self, board, **kwargs):
        self.board = board

    def graphic(self, board, player1, player2):
        """Draw the board and show game info"""
        width = board.width
        height = board.height

        #print("Player", player1, "with X".rjust(3))
        #print("Player", player2, "with O".rjust(3))
        #print()
        for x in range(width):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(height - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                loc = i * width + j
                p = board.koma_int_to_koma_one(board.states.get(loc, -1))
                print(str(p).center(8), end='')
                #if p == player1:
                 #   print('X'.center(8), end='')
                #elif p == player2:
                 #   print('O'.center(8), end='')
                #else:
                 #   print('_'.center(8), end='')
            print('\r\n\r\n')
        print("Kept koma is: ", self.board.kept_koma_int)
        print("")
        print("available location is: ", self.board.availables_field)
        print("")
        print("available koma is: ", self.board.availables_koma_int)
        print("")
        print("state is: ", self.board.states)
        print("")

    def start_play(self, player1, player2, start_player=0, is_shown=1):
        """start a game between two players"""
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')
        self.board.init_board(start_player)
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        
        if is_shown:
            self.graphic(self.board, player1.player, player2.player)
        current_player = self.board.get_current_player()
        player_in_turn = players[current_player]
        self.board.kept_koma_int = player_in_turn.initiate_koma(self.board)
        self.board.availables_koma_int.remove(self.board.kept_koma_int)
        if is_shown:
            self.graphic(self.board, player1.player, player2.player)
        while True:
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.board)
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, player1.player, player2.player)
            if len(self.board.availables_field) == 1 and len(self.board.availables_koma_int) == 0:
                self.board.do_last_move()
                if is_shown:
                    self.graphic(self.board, player1.player, player2.player)

            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is", winner)
                    else:
                        print("Game end. Tie")
                return winner

    def start_self_play(self, player, is_shown=0, temp=1e-3):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        self.board.init_board()
        p1, p2 = self.board.players
        
        self.board.kept_koma_int=0
        self.board.availables_koma_int.remove(self.board.kept_koma_int)
        
        states, mcts_probs, current_players = [], [], []
        while True:
            # field, koma_int,field_probs,koma_int_probsとかく。
            move, move_probs = player.get_action(self.board,
                                                 temp=temp,
                                                 return_prob=1)
            # store the data
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            # perform a move
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, p1, p2)
            if len(self.board.availables_field) == 1 and len(self.board.availables_koma_int) == 0:
                self.board.do_last_move()
                if is_shown:
                    self.graphic(self.board, player1.player, player2.player)

            end, winner = self.board.game_end()
            if end:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)
