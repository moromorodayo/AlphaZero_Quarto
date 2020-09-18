# -*- coding: utf-8 -*-
"""
An implementation of the policyValueNet in PyTorch
Tested in PyTorch 0.2.0 and 0.3.0

@author: Junxiao Song
"""

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


class Net(nn.Module):
    """policy-value network module"""
    def __init__(self, board_width, board_height):
        super(Net, self).__init__()

        self.board_width = board_width
        self.board_height = board_height
        # common layers
        # 第一引数は入力層のdim= 0と一致
        self.conv1 = nn.Conv2d(7, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # action policy layers
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4*board_width*board_height,
                                 board_width*board_height)
        # state value layers
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2*board_width*board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)

        # action koma layers
        self.koma_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.koma_fc1 = nn.Linear(2*board_width*board_height, 64)
        self.koma_fc2 = nn.Linear(64, 16)
        

    def forward(self, state_input):
        '-> torch.Size([1, 16]), torch.Size([1, 1])'

        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # action policy layers
        x_act = F.relu(self.act_conv1(x))
        # ここの4も入力層のdim= 0と一致
        x_act = x_act.view(-1, 4*self.board_width*self.board_height)
        #UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
        x_act = F.log_softmax(self.act_fc1(x_act), dim=1)
        # state value layers
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2*self.board_width*self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = torch.tanh(self.val_fc2(x_val))
        # action koma layers
        x_koma = F.relu(self.koma_conv1(x))
        x_koma = x_koma.view(-1, 2*self.board_width*self.board_height)
        x_koma = F.relu(self.koma_fc1(x_koma))
        x_koma = F.log_softmax(self.koma_fc2(x_koma),dim=1)
        return x_act, x_koma, x_val


class PolicyValueNet():
    """policy-value network """
    def __init__(self, board_width, board_height,
                 model_file=None, use_gpu=False):
        self.use_gpu = use_gpu
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4  # coef of l2 penalty
        # the policy value net module
        if self.use_gpu:
            self.policy_value_net = Net(board_width, board_height).cuda()
        else:
            self.policy_value_net = Net(board_width, board_height)
        self.optimizer = optim.Adam(self.policy_value_net.parameters(),
                                    weight_decay=self.l2_const)

        if model_file:
            net_params = torch.load(model_file)
            self.policy_value_net.load_state_dict(net_params)

    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            log_act_probs,log_koma_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.cpu().numpy())
            koma_probs = np.exp(log_koma_probs.data.cpu().numpy()) 
            return act_probs,koma_probs, value.data.cpu().numpy()
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            log_act_probs,log_koma_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.numpy())
            koma_probs = np.exp(log_koma_probs.data.numpy())
            return act_probs, koma_probs, value.data.numpy()

    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        legal_field_positions = board.availables_field
        #一致していなかった。
        legal_koma_positions = board.availables_koma_int
        current_state = np.ascontiguousarray(board.current_state().reshape(
                -1, 7, self.board_width, self.board_height))
        if self.use_gpu:
            log_act_probs,log_koma_probs, value = self.policy_value_net(
                    Variable(torch.from_numpy(current_state)).cuda().float())
            act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())
            koma_probs = np.exp(log_koma_probs.data.cpu().numpy().flatten())
        else:
            log_act_probs,log_koma_probs, value = self.policy_value_net(
                    Variable(torch.from_numpy(current_state)).float())
            act_probs = np.exp(log_act_probs.data.numpy().flatten())
            koma_probs = np.exp(log_koma_probs.data.numpy().flatten())

        act_probs = zip(legal_field_positions, act_probs[legal_field_positions])
        koma_probs = zip(legal_koma_positions, koma_probs[legal_koma_positions])
        value = value.data[0][0]
        united_act_probs = self.concat_action_and_koma_probs(act_probs,koma_probs)
        return united_act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """perform a training step"""
        # wrap in Variable
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            mcts_probs = Variable(torch.FloatTensor(mcts_probs).cuda())
            winner_batch = Variable(torch.FloatTensor(winner_batch).cuda())
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            mcts_probs = Variable(torch.FloatTensor(mcts_probs))
            winner_batch = Variable(torch.FloatTensor(winner_batch))

        # zero the parameter gradients
        self.optimizer.zero_grad()
        # set learning rate
        set_learning_rate(self.optimizer, lr)

        # forward
        log_act_probs, log_koma_probs, value = self.policy_value_net(state_batch)

        #tmp1 = torch.zeros(len(state_batch), len(log_act_probs), len(log_koma_probs))
        # tmp2 = torch.zeros(len(state_batch), len(log_koma_probs),len(log_act_probs))
        tmp1 = torch.cat([log_act_probs.unsqueeze(2) for _ in range(len(log_koma_probs[0]))], dim=2)
        tmp2 = torch.cat([log_koma_probs.unsqueeze(1) for _ in range(len(log_act_probs[0]))], dim=1)
        #tmp1[:,:] = log_koma_probs
        #tmp2[:,:] = log_act_probs
        #log_united_probs = torch.reshape(tmp1 + tmp2.permute(0,2,1),(len(state_batch),-1))
        log_united_probs = torch.reshape(tmp1 + tmp2,(len(state_batch),-1))

        #torch.bmm(torch.unsqueeze(log_act_probs, 2), torch.unsqueeze(log_koma_probs, 1))
        # log_united_probs = [log_act_prob + log_koma_prob for log_act_prob in log_act_probs for log_koma_prob in log_koma_probs]
        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # Note: the L2 penalty is incorporated in optimizer
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs*log_united_probs, 1))
        loss = value_loss + policy_loss
        # backward and optimize
        loss.backward()
        self.optimizer.step()
        # calc policy entropy, for monitoring only
        entropy = -torch.mean(
                torch.sum(torch.exp(log_united_probs) * log_united_probs, 1)
                )
        # return loss.data[0], entropy.data[0]
        #for pytorch version >= 0.5 please use the following line instead.
        return loss.item(), entropy.item()

    def concat_action_and_koma_probs(self, action_probs, koma_probs):
        'action_probs = [(0, 0.00), (1, 0.3)...]'
        'koma_probs = [(0, 0.00), (1, 0.3)...]'
        '->[((0,0),0),((0,1),0),((1,0)0),((1,1)0.09)...]'
        return [((act[0],koma_int[0]),act[1]*koma_int[1]) for koma_int in koma_probs for act in action_probs ]


    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()  # get model params
        torch.save(net_params, model_file)