# -*- coding: utf-8 -*-
"""
human VS AI models
Input your move in the format: 2,3

@author: Junxiao Song
"""

from __future__ import print_function
import pickle
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from collections import defaultdict
#from policy_value_net_numpy import PolicyValueNetNumpy
# from policy_value_net import PolicyValueNet  # Theano and Lasagne
from policy_value_net_pytorch import PolicyValueNet  # Pytorch
# from policy_value_net_tensorflow import PolicyValueNet # Tensorflow
# from policy_value_net_keras import PolicyValueNet  # Keras


class Human(object):
    """
    human player
    """

    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        try:
            location = input("Your position: ")
            if isinstance(location, str):  # for python3
                location = [int(n, 10) for n in location.split(",")]
            field = board.location_to_where_to_put(location)

            koma_one = input("Your koma: ")
            if isinstance(koma_one, str):  # for python3
                koma_one = [int(n, 10) for n in koma_one.split(",")]
            koma_int = board.koma_one_to_koma_int(koma_one)

            move = (field,koma_int)
        except Exception as e:
            move = -1
        if move == -1 or field not in board.availables_field or koma_int not in board.availables_koma_int:
            print("invalid move")
            move = self.get_action(board)
        return move

    def initiate_koma(self,board):
        try:
            koma_one = input("Your first koma: ")
            if isinstance(koma_one, str):  # for python3
                koma_one = [int(n, 10) for n in koma_one.split(",")]
            koma_int = board.koma_one_to_koma_int(koma_one)
        except Exception as e:
            koma_int = -1
        if koma_int == -1 or koma_int not in board.availables_koma_int:
            print("invalid move")
            move = self.initiate_koma(board)
        return koma_int


    def __str__(self):
        return "Human {}".format(self.player)


def run_human_vs_human():
    n = 4
    width, height = 4, 4
    try:
        board = Board(width=width, height=height, n_in_row=n)
        game = Game(board)

        # human player, input your move in the format: 2,3
        human1 = Human()
        human2 = Human()

        # set start_player=0 for human first
        game.start_play(human1, human2, start_player=1, is_shown=1)
    except KeyboardInterrupt:
        print('\n\rquit')

def run_human_vs_mcts():
    n = 4
    width, height = 4, 4
    model_file = 'best_policy?super.model'
    try:
        board = Board(width=width, height=height, n_in_row=n)
        game = Game(board)

        # ############### human VS AI ###################
        # load the trained policy_value_net in either Theano/Lasagne, PyTorch or TensorFlow

        # best_policy = PolicyValueNet(width, height, model_file = model_file)
        # mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=400)

        # load the provided model (trained in Theano/Lasagne) into a MCTS player written in pure numpy
        best_policy = PolicyValueNet(width, height, model_file)
        mcts_player = MCTSPlayer(best_policy.policy_value_fn,
                                 c_puct=5,
                                 n_playout=400)  # set larger n_playout for better performance

        # uncomment the following line to play with pure MCTS (it's much weaker even with a larger n_playout)
        # mcts_player = MCTS_Pure(c_puct=5, n_playout=1000)

        # human player, input your move in the format: 2,3
        human = Human()

        # set start_player=0 for human first
        game.start_play(human, mcts_player, start_player=1, is_shown=1)
    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    run_human_vs_mcts()
