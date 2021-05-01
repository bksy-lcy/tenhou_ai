# -*- coding: utf-8 -*-
"""
用于预测终局后的最终得分。
用于辅助计算每轮游戏结束后得分来进行弃牌和鸣牌模型的训练

@author:Cahoyang Li
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class GRP_Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        
    def forward(self, state_input):
        
class GRP():
    def __init__(self, model_file=None):
        self.grp_net=GRP_Net().cuda()
        self.l2_const = 1e-4
        self.optimizer = optim.Adam(self.grp_net.parameters(),weight_decay=self.l2_const)
        if model_file:
            net_params = torch.load(model_file)
            self.grp_net.load_state_dict(net_params)
        
    def get_reward(self, state_input):
        
    
    def train_step(self, state_batch, target_batch, lr):
        # wrap in Variable
        state_batch = Variable(torch.FloatTensor(state_batch).cuda())
        target_batch = Variable(torch.FloatTensor(target_batch).cuda())

        # zero the parameter gradients
        self.optimizer.zero_grad()
        # set learning rate
        set_learning_rate(self.optimizer, lr)
        
        # forward
        log_predicts = self.grp_net(state_batch)
        # define the loss = (z - v)^2 + c||theta||^2
        # Note: the L2 penalty is incorporated in optimizer
        #loss = F.mse_loss(value.view(-1), winner_batch)
        #policy_loss = -torch.mean(torch.sum(mcts_probs*log_act_probs, 1))
        loss = 0
        # backward and optimize
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def get_param(self):
        net_params = self.grp_net.state_dict()
        return net_params
    
    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_param()  # get model params
        torch.save(net_params, model_file)
    
if __name__ == '__main__':
    
