import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from environment import gameEnv

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"


class BackboneTrainer:
    def __init__(self, NN):
        self.step_model = NN()
        self.step_model.to(device)


class PredictionTrainer:
    def __init__(self, PredictionNN, backbone, env, learning_rate=0.01):
        self.step_model = PredictionNN()
        self.step_model.to(device)
        self.backbone = backbone
        self.env = env
        self.learning_rate = learning_rate

        self.value_criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.step_model.parameters(), lr=self.learning_rate)

    def getBack(self, var_grad_fn):
        # print(var_grad_fn)
        for n in var_grad_fn.next_functions:
            if n[0]:
                try:
                    tensor = getattr(n[0], 'variable')
                    print("paisi", n[0], tensor.grad.size())
                    # print('Tensor with grad found:', tensor.size())
                    print(' - gradient:', tensor.grad)
                    # print()
                except AttributeError as e:
                    self.getBack(n[0])

    def train(self, states, search_pis, returns, moves):
        self.optimizer.zero_grad()

        logits = []
        policy = []
        rewards = []

        for i, state in enumerate(states):
            feat_mat = self.backbone.step_model(torch.FloatTensor(state).to(device))
            # ~~~ Come back after figuring out move incorporation
            logit, y, z = self.step_model(feat_mat, moves[i])
            logits.append(logit)
            policy.append(y)
            rewards.append(z)

        logits = torch.stack(logits)
        policy = torch.stack(policy)
        # ~~~ Check out what to do with rewards
        rewards = torch.stack(rewards)

        logsoftmax = nn.LogSoftmax(dim=1)

        search_pis = torch.FloatTensor(search_pis).to(device)
        returns = torch.FloatTensor(returns).to(device)
        loss_policy = torch.mean(torch.sum(-search_pis * logsoftmax(logits), dim=1))
        # ~~~ Again, check out after reward
        loss_reward = self.value_criterion(rewards, returns.view(returns.size()[0], 1))

        total_loss = loss_policy + loss_reward
        total_loss.backward()
        self.optimizer.step()

        # self.getBack(loss_grad_fn)

        return loss_policy, loss_reward
