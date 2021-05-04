# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np


class Game_Server(object):
    def __init__(self, setting=None):
        self.TENHOU_HOST = "133.242.10.78"
        self.TENHOU_PORT = 10080
        self.USER_ID = "NoName"
        self.LOBBY = "0"
        self.WAITING_GAME_TIMEOUT_MINUTES = 10
        self.IS_TOURNAMENT = False
        self.STAT_SERVER_URL = ""
        self.STAT_TOKEN = ""
        self.PAPERTRAIL_HOST_AND_PORT = ""
        self.SENTRY_URL = ""
        self.LOG_PREFIX = ""
        self.PRINT_LOGS = True
        self.TOURNAMENT_API_TOKEN = None
        self.TOURNAMENT_API_URL = None
        self.GAME_TYPE = "1"
        if setting:
            #自定义部分
            pass
        
    def self_play(self,players):
        # init_seat_and_score
        # while not_end
        #  init_round
        #  send_init_round_message
        #  while not_round_end
        #   ask_and_deal_action
        #  score_change
        #  _get_one_part_data
        # send_data
    def online_play(self,player):
        #get conect
        #search game
        #start game
        #end game
'''
class Game(object):

    def __init__(self, board, **kwargs):
        self.board = board

    def graphic(self, board, player1, player2):
        # Draw the board and show game info
        width = board.width
        height = board.height

        print("Player", player1, "with X".rjust(3))
        print("Player", player2, "with O".rjust(3))
        print()
        for x in range(width):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(height - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                loc = i * width + j
                p = board.states.get(loc, -1)
                if p == player1:
                    print('X'.center(8), end='')
                elif p == player2:
                    print('O'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')

    def start_play(self, player1, player2, start_player=0, is_shown=1):
        # start a game between two players
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
        while True:
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.board)
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, player1.player, player2.player)
            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")
                return winner

    def start_self_play(self, player, is_shown=0, temp=1e-3):
        # start a self-play game using a MCTS player, reuse the search tree,
        # and store the self-play data: (state, mcts_probs, z) for training
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []
        while True:
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
'''
