# -*- coding: utf-8 -*-

import numpy as np
import copy

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs
"""
opt: 0,draw 1,dora 2,dsicard 3,ron 4,kong 5,pong 6,chow 7,reach 8,end
"""
def MCT_Nex(player,opt,state,choice,close_kong=0):
    """
    状态转移函数
    """
    if opt==0 :
        if state==0 :
            return player,3,0
        if state==1 :
            return player,3,4
    if opt==1 :
        if state==0 :
            return player,0,0
        if state==1 :
            return player,4,1
        if state==2 :
            return player,1,0
        if state==3 :
            return player,3,7
    if opt==2 :
        if state==0 :
            return player,3,1
        if state==1 :
            return player,3,2
        if state==2 :
            return player,3,5
        if state==3 :
            return player,3,6
    if opt==3 :
        if state==0 :
            if choice==0 :
                return player,4,0
            else :
                return player,8,0
        if state==1 :
            if choice==0 :
                return player,4,1
            else :
                return player,8,0
        if state==2 :
            if choice==0 :
                return player,4,1
            else :
                return player,8,0
        if state==3 :
            if choice==0 :
                return player,0,1
            else :
                return player,8,0
        if state==4 :
            if choice==0 :
                return player,4,2
            else :
                return player,8,0
        if state==5 :
            if choice==0 :
                return player,1,1
            else :
                return player,8,0
        if state==6 :
            if choice==0 :
                return player,1,1
            else :
                return player,8,0
        if state==7 :
            if choice==0 :
                return player,0,1
            else :
                return player,8,0
    if opt==4 :
        if state==0:
            if choice==0 :
                return player,7,0
            else :
                if close_kong==1:
                    return player,1,0
                else :
                    return player,3,3
        if state==1 :
            if choice==0 :
                return player,5,0
            else :
                return (player+choice-1)%4,0,1
        if state==2:
            if choice==0 :
                return player,7,1
            else :
                if close_kong==1:
                    return player,1,2
                else :
                    return player,1,3
    if opt==5 :
        if choice==0 :
            return player,6,0
        else :
            return (player+choice)%4,2,0
    if opt==6 :
        if choice==0 :
            return (player+1)%4,0,0
        else :
            return (player+1)%4,2,0
    if opt==7 :
        if state==0 :
            if choice==0 :
                return player,2,0
            else :
                return player,2,1
        if state==1 :
            if choice==0 :
                return player,2,2
            else :
                return player,2,3
    return -1,-1,-1

class MCT_TreeNode(object):
    def __init__(self, _parent, _prior_p, _main_player_id, _now_player_id, _now_player_opt, _now_player_opt_state):
        """
        _parent:父节点
        _prior_p:先验概率
        _main_player_id:主视角id(0-3)
        _now_player_id:当前玩家id(0-3)
        _now_player_opt:当前玩家正在进行的操作(0-6)
        _now_player_opt_state:当前玩家正在进行的操作的种类(0-3)
        """
        self._parent=_parent
        self._P=_prior_p
        self._main_player_id=_main_player_id
        self._now_player_id=_now_player_id
        self._now_player_opt=_now_player_opt
        self._now_player_opt_state=_now_player_opt_state
        self._children={}
        self._n_visits=0
        self._Q=[0,0,0,0]
        self._u=0
        
    def expand(self,action_priors,close_kong=0):
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self,prob,self._main_player_id,MCT_Nex(self._now_player_id,self._now_player_opt,self._now_player_opt_state,action,close_kong))
    
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

    def __init__(self, policy_value_fn, c_puct, n_playout, init_state):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._root = TreeNode(None, 1.0,init_state[0],init_state[2],init_state[3],init_state[4])
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
            state.do_action(action)

        # Evaluate the leaf using a network which outputs a list of
        # (action, probability) tuples p and also a score v in [-1, 1]
        # for the current player.
        action_probs, leaf_value = self._policy(state)
        # Check for end of game.
        if node._now_player_opt!=7 :
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
    def get_move_probs_pong(self, temp=1e-3):
        if 0 in self._root._children:
            now_root=self._root._children[0]
            if not now_root.is_leaf():
                act_visits = [(act, node._n_visits) for act, node in now_root._children.items()]
                acts, visits = zip(*act_visits)
                act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))
                return acts, act_probs
        return [0],[0]
    
    def get_move_probs_chow(self, temp=1e-3):
        if 0 in self._root._children:
            now_root=self._root._children[0]
            if not now_root.is_leaf():
                if 0 in now_root._children:
                    now_root=now_root._children[0]
                    if not now_root.is_leaf():
                        act_visits = [(act, node._n_visits) for act, node in now_root._children.items()]
                        acts, visits = zip(*act_visits)
                        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))
                        return acts, act_probs
        return [0],[0]
        
    def update_with_move(self, last_move, close_kong):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0, self._root._main_player_id,MCT_Nex(self._root._now_player_id,self._root._now_player_opt,self._root._now_player_opt_state,action,close_kong))

    def __str__(self):
        return "MCTS"

class Game_State(object):
    opt_num_move=[34,34,34,16,5,4,4,2,0]
    opt_name=['draw','dora','discard','ron','kong','pong','chow','reach','end']
    def __init__(self, init_message):
        #局面表示缺少完整的宝牌表示，牌河和宝牌指示器没有宝牌表示
        self.main_player_id=init_message['player']# 0-3
        self.see_opponent=init_message['opponent']# 0/1
        self.see_mountion=init_message['mountion']# 0/1
        self.now_player_id=init_message['now_id']# 0-3
        self.now_player_opt=init_message['now_opt']# 0-8
        self.now_player_opt_state=init_message['now_opt_state']# 0-7
        
        self.score=init_message['score']# 4*int 4*34*1
        self.combo=init_message['combo']# int 34*1
        self.reach=init_message['reach']# int 34*1
        self.time=init_message['time']# int 34*1
        #now_id 34*1
        #min_id 34*1
        self.wind=init_message['wind']# 34*1
        self.selfwind=init_message['selfwind']# 34*1
        self.dora=init_message['dora']# 34*1
        self.closehand=init_message['closehand']#4*34*17
        self.openhand=init_message['openhand']#4*34*32((5+3)*4)
        self.river=init_message['river']#4*34*34
        self.reach=init_message['reach']#4*34*1
        # 向听数 4*34*1
        # 有效进章 4*34*34
        # 得点与速度 4*34*72(4*18)
        self.mountion=init_message['mountion']
        self.mountion_detail=init_message['mountion_detail']
        
    def do_action(self,action):
        #change state
        if self.now_player_opt==0 :
            if self.see_opponent :
                
            else :
                
        if self.now_player_opt==6 :
            
        if self.now_player_opt==3 :
            if self.now_player_id==self.main_player_id or self.see_opponent :
                
            else : 
                
        if self.now_player_opt==1 :
            if self.now_player_opt_state==0 or self.now_player_opt_state==2 :
                
            else :
                if self.see_opponent :
                    
                else :
                    
        if self.now_player_opt==2 :
            if self.now_player_opt_state==0 or self.now_player_opt_state==2 :
                
            else :
                if self.see_opponent :
                    
                else :
                    
            
        if self.now_player_opt==4 :
            if self.see_opponent :
                    
            else :
                
        if self.now_player_opt==5 :
            if self.see_opponent :
                    
            else :
                
    def do_action_with_dora(self,action,dora):
        self.do_action(action)
        #处理红宝牌
        if self.now_player_opt==0 :
            #自己抽到了红5
        if self.now_player_opt==2 :
            #带红5的杠
        if self.now_player_opt==4 :
            #带红5的碰
        if self.now_player_opt==5 :
            #带红5的吃
        
    def get_num_move(self):
        return opt_num_move[self.now_player_opt]
        
    def opt_is_close_kong(self):
        if self.now_player_opt!=2 :
            return 0
        if self.now_player_opt_state==1 :
            return 0
        if self. :
            # 检查是否为加杠
            return 0
        return 1
    def get_mct_state(self):
        return self.main_player_id,self.now_player_id,self.now_player_opt,self.now_player_opt_state
    
    def get_score(self):
        
    
    def get_xt(self,player):
        
    def get_jz(self,player):
        
    def get_gain_speed(self,player):
        
    def get_state_920_34_1(self):
        now_state=np.array()
        now_state.appned(self.get_score())
        now_state.appned([1 if i==self.combo else 0 for i in range(34)])
        now_state.appned([1 if i==self.reach else 0 for i in range(34)])
        now_state.appned([1 if i==self.time else 0 for i in range(34)])
        now_state.appned([1 if i==self.now_player_id else 0 for i in range(34)])
        now_state.appned([1 if i==self.main_player_id else 0 for i in range(34)])
        
        return now_state
    def get_legal_actions(self):
        legal_actions=[]
        if self.now_player_opt==0 or self.now_player_opt==6 :
            
        if self.now_player_opt==3 :
            if self.now_player_id==self.main_player_id or self.see_opponent :
                # 立直/没立直
            else : 
                
        if self.now_player_opt==1 :
            if self.now_player_opt_state==0 or self.now_player_opt_state==2 :
                
            else :
                if self.see_opponent :
                    
                else :
                    legal_actions=[0,2,4,6,8,10,12,14]
        if self.now_player_opt==2 :
            if self.now_player_opt_state==0 or self.now_player_opt_state==2 :
                
            else :
                if self.see_opponent :
                    
                else :
                    
            
        if self.now_player_opt==4 :
            if self.see_opponent :
                    
            else :
                
        if self.now_player_opt==5 :
            if self.see_opponent :
                    
            else :
        return legal_actions
    def current_state(self):       
        return self.get_current_state(),opt_name[self.now_player_opt],self.get_legal_actions()
    
class MCTSPlayer(object):
    def __init__(self, policy_value_function, n_playout,c_puct=5,is_selfplay=0):
        self._is_selfplay = is_selfplay
        self._n_playout=n_playout
        self._c_puct=c_puct
        self._policy_value_function=policy_value_function
        
    def set_game_state(round_init_message):
        self.game_state=Game_State(round_init_message)
        self.mcts = MCTS(self._policy_value_function, self._c_puct, self._n_playout, self.game_state.get_mct_state())
        
    def get_action(self, temp=1e-3, return_prob=0):
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        mp,np,nop,nops=self.game_state.get_mct_state()
        if nop==4 and np!=0 :
            move=np.arry([0,0,0])
            move_probs=np.arry([np.zeros(5),np.zeros(4),np.zeros(4)])
            acts, probs = self.mcts.get_move_probs(self.game_state, temp)
            move_probs[0][list(acts)] = probs
            if self._is_selfplay:
                _p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
                move[0] = np.random.choice(acts,p=_p)
            else:
                move[0] = np.random.choice(acts,p=_p)
            if move[0]==0 :
                acts, probs = self.mcts.get_move_probs_pong(temp)
                move_probs[1][list(acts)] = probs
                if self._is_selfplay:
                    _p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
                    move[1] = np.random.choice(acts,p=_p)
                else:
                    move[1] = np.random.choice(acts,p=_p)
                if move[1]==0 :
                    acts, probs = self.mcts.get_move_probs_chow(temp)
                    move_probs[2][list(acts)] = probs
                    if self._is_selfplay:
                        _p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
                        move[2] = np.random.choice(acts,p=_p)
                    else:
                        move[2] = np.random.choice(acts,p=_p)
            if return_prob:
                return move, move_probs
            else:
                return move
        else :
            move_probs = np.zeros(self.game_state.get_num_move())
            acts, probs = self.mcts.get_move_probs(self.game_state, temp)
            move_probs[list(acts)] = probs
            if nop==3 :
                tmp=np.array([0,0])
                for i in range(16):
                    tmp[i%2]+=move_probs[i]
                _p=tmp
                acts=np.array([0,1])
            if self._is_selfplay:
                _p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
                move = np.random.choice(acts,p=_p)
            else:
                move = np.random.choice(acts,p=_p)
            if return_prob:
                return np.arry([move]), np.arry([move_probs])
            else:
                return move
                                  
    def do_action(self,action):
        if self._is_selfplay :
            self.mcts.update_with_move(action,self.game_state.opt_is_close_kong())
            self.game_state.do_action(action)
        else :
            self.game_state.do_action(action)
            self.mcts = MCTS(self._policy_value_function, self._c_puct, self._n_playout, self.game_state.get_mct_state())
    def do_action_with_dora(self,action,dora):
        if self._is_selfplay :
            self.mcts.update_with_move(action,self.game_state.opt_is_close_kong())
            self.game_state.do_action_with_dora(action,dora)
        else :
            self.game_state.do_action_with_dora(action,dora)
            self.mcts = MCTS(self._policy_value_function, self._c_puct, self._n_playout, self.game_state.get_mct_state())

    def __str__(self):
        return "MCTS {}".format(self.player)
