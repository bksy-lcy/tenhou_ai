# 1.train GRP model
## 1.1 GRP_NET: Global_Reward_Prediction.py
## 1.2 transform log to data: log_to_GRP.py
## 1.3 train: Global_Reward_Prediction.py
# 2.train player
## 2.1 self_play_server: game.py
## 2.2 player_net for (action,prob): discard_net.py and opts_net.py
## 2.3 player_mcts: mcts_lcy.py
## 2.4 tarin(self_play): train.py
# 3.inline test
## 3.1 online_play_server: game.py
## 3.2 online_game_start: game_start.py

# 调用
train->game->mcts_lcy->discard_net.py and opts_net.py
game_start->game->...
log_to_GRP
Global_Reward_Prediction

# 局面状态
## 公用状态
1.点数 34，4 （二进制）
2.本场 34，1
3.立直棒 34，1
4.巡目 34，1
5.场风 34，1
6.自风 34，1
7.宝牌 34，5
8.自.手牌 34，17 14+3（红宝牌）
9.自.副露 34，32 32=（（5+3）* 4） 5=4（副露牌）+1（哪张来自其他人） 3（红宝牌）
10.自.牌河 34，34
11.自.立直 34， 1 （巡目）
12.自.先验.向听 34，1
13.自.先验.有效进章 34，34 （打i时的有效进章i=0...33）
14.自.先验.得点与速度 34，（4 * 18) （打i能否在j向听内听到k * 100的牌，i=0...33,j=0...3,k=1...12,16,18,24,32,36,48） 
15.下.副露 34，32 
16.下.牌河 34，34
17.下.立直 34， 1
18.对.副露 34，32
19.对.牌河 34，34
20.对.立直 34， 1
21.上.副露 34，32
22.上.牌河 34，34
23.上.立直 34， 1
24.牌山 34，4 （自己视角）
## 鸣牌的额外状态（立直（其实不能算鸣牌）、吃、碰、杠）
25.当前操作 34，8
26.先验.向听 34，1
27.先验.有效进章 34，34 （打i时的有效进章i=0...33）
28.先验.得点与速度 34，（4 * 18) （打i能否在j向听内听到k * 100的牌，i=0...33,j=0...3,k=1...12,16,18,24,32,36,48） 
