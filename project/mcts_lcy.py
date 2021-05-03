# -*- coding: utf-8 -*-
"""
Monte Carlo Tree Search in AlphaGo Zero style, which uses a policy-value
network to guide the tree search and evaluate the leaf nodes

@author: Junxiao Song

根据立直麻将的特点，对原MCTS进行改动
@author: Chaoyang Li
"""

import numpy as np
import copy


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs
"""
opt: 0,draw 1,ron 2,kong 3,discard 4,pong 5,chow 6,dora 7,end
"""
def MCT_Nex(player,opt,state,choice,close_kong=0):
    """
    状态转移函数
    """
    if opt==0 :
        if state==0 :
            return player,1,0
        else :
            return player,1,2
    if opt==1 :
        if state==0 :
            if choice==0 :
                return player,2,0
            else :
                return player,7,0
        if state==1 :
            if choice==0 :
                return player,2,1
            else :
                return player,7,0
        if state==2 :
            if choice==0 :
                return player,2,2
            else :
                return player,7,0
        if state==3 :
            if choice==0 :
                return player,6,2
            else :
                return player,7,0
    if opt==2 :
        if state==0:
            if choice==0 :
                return player,3,0
            else :
                if close_kong==0:
                    return player,0,1
                else :
                    return player,6,0
        if state==1 :
            if choice==0 :
                return player,4,0
            else :
                return (player+choice-1)%4,0,1
        if state==2:
            if choice==0 :
                return player,3,1
            else :
                if close_kong==0:
                    return player,6,1
                else :
                    return player,6,3
    if opt==3 :
        if state==0 :
            return player,1,1
        if state==1 :
            return player,1,3
    if opt==4 :
        if choice==0 :
            return player,5,0
        else :
            return (player+choice)%4,3,0
    if opt==5 :
        if choice==0 :
            return (player+1)%4,0,0
        else :
            return (player+1)%4,3,0
    if opt==6 :
        if state==0:
            return player,0,0
        if state==1:
            return player,0,1
        if state==2:
            return player,2,1
        if state==3:
            return player,6,0

class MCT_TreeNode(object):
    def __init__(self, _parent, _prior_p, _main_player_id, _main_player_can_see, _now_player_id, _now_player_opt, _now_player_opt_state):
        """
        _parent:父节点
        _prior_p:先验概率
        _main_player_id:主视角id(0-3)
        _now_player_id:当前玩家id(0-3)
        _now_player_opt:当前玩家正在进行的操作(0-6)
        _now_player_opt_state:当前玩家正在进行的操作的种类(0-3)
        _main_player_can_see:主视角可以看见其他玩家摸到的牌吗0/1
        """
        self._parent=_parent
        self._P=_prior_p
        self._main_player_id=_main_player_id
        self._now_player_id=_now_player_id
        self._now_player_opt=_now_player_opt
        self._now_player_opt_state=_now_player_opt_state
        self._main_player_can_see=_main_player_can_see
        self._children={}
        self._n_visits=0
        self._Q=[0,0,0,0]
        self._u=0
        
    def expand(self,action_priors,close_kong=0):
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self,prob,self._main_player_id,self._main_player_can_see,
                                                  MCT_Nex(self._now_player_id,self._now_player_opt,self._now_player_opt_state,action,close_kong)
    
    def select(self, c_puct):
        return max(self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        self._n_visits += 1
        for i in range(4):
            self._Q[i] += 1.0*(leaf_value[i] - self._Q[i]) / self._n_visits
    
    def update_recursive(self, leaf_value):
        if self._parent:
            self._parent.update_recursive(leaf_value)
        self.update(leaf_value)
    
    def get_value(self, c_puct):
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q[self._main_player_id] + self._u
    
    def is_leaf(self):
        return self._children == {}

    def is_root(self):
        return self._parent is None

class MCTS(object):
    """An implementation of Monte Carlo Tree Search."""

    def __init__(self, policy_value_fn, c_puct=5, n_playout, init_state):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._root = TreeNode(None, 1.0,init_state[0],init_state[1],init_state[2],init_state[3],init_state[4])
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self._root
        while(1):
            if node.is_leaf():
                break
            # Greedily select next move.
            action, node = node.select(self._c_puct)
            state.do_move(action)

        # Evaluate the leaf using a network which outputs a list of
        # (action, probability) tuples p and also a score v in [-1, 1]
        # for the current player.
        action_probs, leaf_value = self._policy(state)
        # Check for end of game.
        if node._now_player_opt<>7 :
            # not end
            node.expand(action_probs,state.opt_is_close_kong())
        # Update value and visit count of nodes in this traversal.
        node.update_recursive(leaf_value)

    def get_move_probs(self, state, temp=1e-3):
        """Run all playouts sequentially and return the available actions and
        their corresponding probabilities.
        state: the current game state
        temp: temperature parameter in (0, 1] controls the level of exploration
        """
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        # calc the move probabilities based on visit counts at the root node
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def update_with_move(self, last_move, close_kong):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0, self._root._main_player_id, self._root._main_player_can_see, 
                                  MCT_Nex(self._root._now_player_id,self._root._now_player_opt,self._root._now_player_opt_state,
                                          action,close_kong)

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    def __init__(self, policy_value_function,c_puct=5, n_playout, game_state, is_selfplay=0):
        self._is_selfplay = is_selfplay
        self._n_playout=n_playout
        self._c_puct=c_puct
        self._policy_value_function=policy_value_function
        self.game_state=game_state
    
    def reset_mcts(self):
        self.mcts = MCTS(self._policy_value_function, self._c_puct, self._n_playout, self.game_state.get_mct_state())

    def get_action(self, temp=1e-3, return_prob=0):
        sensible_moves = self.game_state.availables()
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move_probs = np.zeros(board.width*board.height)
        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(board, temp)
            move_probs[list(acts)] = probs
            if self._is_selfplay:
                # add Dirichlet Noise for exploration (needed for
                # self-play training)
                move = np.random.choice(
                    acts,
                    p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
                )
                # update the root node and reuse the search tree
                self.mcts.update_with_move(move)
            else:
                # with the default temp=1e-3, it is almost equivalent
                # to choosing the move with the highest prob
                move = np.random.choice(acts, p=probs)
                # reset the root node
                self.mcts.update_with_move(-1)
#                location = board.move_to_location(move)
#                print("AI move: %d,%d\n" % (location[0], location[1]))

            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("WARNING: the board is full")
    
    def do_action(self,action):
        

    def __str__(self):
        return "MCTS {}".format(self.player)
