import copy
import datetime

import math
import numpy as np
import random as rd


# exploration constants
c_PUCT = 1.38
# Dirch noise
D_NOISE_ALPHA = 0.1
# Stochastic move number
MOVE_THRESHOLD = 0



class MCTSNode:
    def __init__(self, state, env, parent=None, predicted_moves=None, predicted_Q=None, action_space_size=None):
        self.env = env
        if parent is None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1
        self.parent = parent
        self.state = state
        self.is_expanded = False
        self.predicted_moves = predicted_moves
        self.predicted_Q = predicted_Q
        self.actions = {}
        self.children = {}
        self.N = 0.0
        self.W = 0.0

        self.action_idx = {}
        self.idx_action = {}

        self.child_N = {}
        self.child_W = {}
        self.original_prior = {}
        self.child_prior = {}

    @property
    def Q(self):
        """
        Returns the current action value of the node.
        """
        return self.W / (1 + self.N)

    @property
    def child_Q(self):
        for key in self.child_N:
            self.child_N[key] = self.children[key].N
            self.child_W[key] = self.children[key].W
        return np.array(list(self.child_W.values())) / (1 + np.array(list(self.child_N.values())))

    @property
    def child_U(self):
        for key in self.child_N:
            self.child_N[key] = self.children[key].N
        return (c_PUCT * math.sqrt(1 + self.N)) * np.array(list(self.child_prior.values())) / (
                1 + np.array(list(self.child_N.values())))

    @property
    def child_action_score(self):
        """
        Action_Score(s, a) = Q(s, a) + U(s, a) as in paper. A high value
        means the node should be traversed.
        """
        return self.child_Q + self.child_U

    def is_done(self, agentType, idx):
        if agentType == "Prey" and self.state[1][idx] == -1:
            return True
        return self.env.is_done_state(self.state, self.depth)

    def select_moves_from_pi(self):

        predator_moves = []
        prey_moves = []
        for i in range(len(self.predicted_moves[0])):
            pi = self.predicted_moves[0][i]
            move_idx = np.argmax(pi)
            predator_moves.append(self.env.actions[self.state[0][i]][move_idx])
        for i in range(len(self.state[1])):
            if self.state[1][i] == -1:
                prey_moves.append((-1, -1))
                continue
            pi = self.predicted_moves[1][i]
            move_idx = np.argmax(pi)
            prey_moves.append(self.env.actions[self.state[1][i]][move_idx])
        return [predator_moves, prey_moves]

    def create_child(self, actions, idx=None):
        new_state = self.env.next_state(self.state, actions)

        if idx is None:
            action_immutable_str = str(actions)
            if action_immutable_str in self.children:
                return self.children[action_immutable_str]
            self.children[action_immutable_str] = MCTSNode(new_state, self.env, parent=self)
            self.child_N[action_immutable_str] = 0
            self.child_W[action_immutable_str] = 0
            idx = len(self.action_idx)
            self.action_idx[action_immutable_str] = idx
            self.idx_action[idx] = action_immutable_str
            self.child_prior[action_immutable_str] = 1.0
            return self.children[action_immutable_str]
        else:
            action_immutable_str = actions[1][idx]
            if action_immutable_str in self.children:
                return self.children[action_immutable_str]
            self.children[action_immutable_str] = MCTSNode(new_state, self.env, parent=self)
            return self.children[action_immutable_str]

    def backup_value(self, value, up_to):
        self.W += value
        if self.parent is None or self is up_to:
            return
        # self.parent.backup_value(value, up_to)
        # self.parent.backup_value(value*1.003, up_to)
        self.parent.backup_value(value * 0.95, up_to)


class C_MCTS:
    def __init__(self, predator_netw, prey_netw, backbone, env, agentType, idx, ex_fac):
        self.predator_netw = predator_netw
        self.prey_netw = prey_netw
        self.backbone = backbone
        self.env = env
        self.agentType = agentType
        self.idx = idx
        self.ex_fac = ex_fac
        self.root = None

    def tree_search(self):
        failsafe = 0
        while failsafe < 5:
            failsafe += 1

            current = self.root
            while True:
                current.N += 1
                if not current.is_expanded:
                    break
                if current.is_done(self.agentType, self.idx):
                    break
                best_move = np.argmax(current.child_action_score)
                current = current.children[current.idx_action[best_move]]

            # leaf paoar por
            leaf = current
            if leaf.is_done(self.agentType, self.idx):
                if self.agentType == "Predator":
                    score = self.env.get_return(leaf.state, self.agentType, self.idx, self.root.state)
                else:
                    score = self.env.get_return(leaf.state, self.agentType, self.idx)
                # print("bhauu ", score, leaf.depth)
                leaf.backup_value(score, up_to=self.root)
                continue
            # new node
            if self.agentType == "Predator":
                leaf_my_moves, Q_val = self.find_gibbs_moves(leaf.state, self.agentType, self.ex_fac * 3)
                leaf_opposition_moves, _ = self.find_gibbs_moves(leaf.state, "Prey", self.ex_fac)
            else:
                leaf_my_moves, Q_val = self.find_gibbs_moves(leaf.state, self.agentType, self.ex_fac * 3)
                leaf_opposition_moves, _ = self.find_gibbs_moves(leaf.state, "Predator", self.ex_fac)

            leaf.action_space_size = (len(leaf_my_moves), len(leaf_opposition_moves))
            leaf.backup_value(Q_val, up_to=self.root)

            for move1 in leaf_my_moves:
                for move2 in leaf_opposition_moves:
                    actions = [move1, move2]
                    leaf.create_child(actions)

            leaf.is_expanded = True
            break
        # print("pore", len(self.root.children), leaf.depth)

    def one_hot_state(self, state):
        temp_state = np.zeros([self.env.n_nodes, len(state[0]) + len(state[1])], dtype=float)
        for i in range(len(state[0])):
            if state[0][i] != -1:
                temp_state[state[0][i]][i] = 1.0
        for i in range(len(state[1])):
            if state[1][i] != -1:
                temp_state[state[1][i]][len(state[0]) + i] = 1.0
        return temp_state

    def find_gibbs_moves(self, state, active, limit):
        qval = 0.0
        todo = 'boltzman'
        gorge = {}

        actions = [[], []]
        list_actions = []
        for i in range(len(state[0])):
            actions[0].append((state[0][i], state[0][i]))
        for i in range(len(state[1])):
            if state[1][i] == -1:
                actions[1].append((-1, -1))
            else:
                actions[1].append((state[1][i], state[0][i]))

        if active == "Predator":
            for lim in range(limit):
                for i in range(len(state[0])):
                    actions[0][i] = (state[0][i], state[0][i])
                    mon = self.env.next_state(state, actions)
                    feat_mat = self.backbone.step_model.step(self.one_hot_state(mon))
                    probs, val = self.predator_netw[mon[0][i]].step_model.step(feat_mat, self.root.depth)
                    if lim == 0 and i == 0:
                        qval = val.data.cpu().numpy()
                    probs = probs.data.cpu().numpy()
                    if todo == "boltzman":
                        for j in range(len(probs)):
                            if j == 0:
                                continue
                            probs[j] = probs[j - 1] + probs[j]
                        selection = rd.random()
                        what = probs.searchsorted(selection)
                        if what >= len(self.env.actions[mon[0][i]]):
                            print("I MESSED UP", what, len(self.env.actions[mon[0][i]]), probs)
                        move = self.env.actions[mon[0][i]][what]
                        actions[0][i] = move
                    elif todo == "greedy":
                        what = np.argmax(probs)
                        move = self.env.actions[mon[0][i]][what]
                        actions[0][i] = move
                if str(actions) not in gorge:
                    list_actions.append(actions[0])
                    gorge[str(actions)] = 1.0
            return list_actions, qval

        else:
            for lim in range(limit):
                for i in range(len(state[1])):
                    if state[1][i] == -1:
                        continue
                    actions[1][i] = (state[1][i], state[1][i])
                    mon = self.env.next_state(state, actions)
                    feat_mat = self.backbone.step_model.step(self.one_hot_state(mon))
                    probs, val = self.prey_netw[mon[1][i]].step_model.step(feat_mat, self.root.depth)
                    if lim == 0 and i == 0:
                        qval = val.data.cpu().numpy()
                    probs = probs.data.cpu().numpy()
                    if todo == "boltzman":
                        for j in range(len(probs)):
                            if j == 0:
                                continue
                            probs[j] = probs[j - 1] + probs[j]
                        selection = rd.random()
                        what = probs.searchsorted(selection)
                        move = self.env.actions[mon[1][i]][what]
                        actions[1][i] = move
                    elif todo == "greedy":
                        what = np.argmax(probs)
                        move = self.env.actions[mon[1][i]][what]
                        actions[1][i] = move
                if str(actions) not in gorge:
                    list_actions.append(actions[1])
                    gorge[str(actions)] = 1.0
            return list_actions, qval

    def pick_action(self, idx):
        # the idea is to pick the best 'move' after the search and send it over
        for key in self.root.child_N:
            self.root.child_N[key] = self.root.children[key].N

        compressed = [0.0] * self.root.action_space_size[0]
        pre = list(self.root.child_N.values())
        for i in range(len(pre)):
            compressed[int(i / self.root.action_space_size[1])] = compressed[int(i / self.root.action_space_size[1])] + pre[i]
        child_number = np.argmax(np.array(compressed))
        child_number = child_number * self.root.action_space_size[1]
        child = self.root.children[self.root.idx_action[child_number]]

        # amar state
        probs = [0.0] * self.env.degree[self.root.state[0][idx]]
        move_number = 0
        for i in range(self.env.degree[self.root.state[0][idx]]):
            if self.env.actions[self.root.state[0][idx]][i] == (self.root.state[0][idx], child.state[0][idx]):
                move_number = i
                break
        probs[move_number] = 1.0

        state = [np.array(child.state[0]), np.array(child.state[1])]
        state[0][idx] = self.root.state[0][idx]

        return state, (self.root.state[0][idx], child.state[0][idx]), probs


class DC_MCTS:
    def __init__(self, predator_netw, prey_netw, backbone, env, agentType, idx, ex_fac):
        self.predator_netw = predator_netw
        self.prey_netw = prey_netw
        self.backbone = backbone
        self.env = env
        self.agentType = agentType
        self.idx = idx
        self.ex_fac = ex_fac
        self.root = None

    def tree_search(self):
        failsafe = 0
        while failsafe < 5:
            failsafe += 1

            current = self.root
            while True:
                current.N += 1
                if not current.is_expanded:
                    break
                if current.is_done(self.agentType, self.idx):
                    break
                best_move = np.argmax(current.child_action_score)
                current = current.children[current.idx_action[best_move]]

            # leaf paoar por
            leaf = current
            if leaf.is_done(self.agentType, self.idx):
                if self.agentType == "Predator":
                    score = self.env.get_return(leaf.state, self.agentType, self.idx, self.root.state)
                else:
                    score = self.env.get_return(leaf.state, self.agentType, self.idx)
                leaf.backup_value(score, up_to=self.root)
                continue
            # new node
            predicted_moves, Q_val = self.predict_agent_moves(leaf.state, leaf.depth,
                                                              self.agentType, self.idx)
            if self.agentType == "Predator":
                leaf_opposition_moves, _ = self.find_gibbs_moves(leaf.state, "Prey", self.ex_fac)
            else:
                leaf_opposition_moves, _ = self.find_gibbs_moves(leaf.state, "Predator", self.ex_fac)

            for move1 in predicted_moves:
                for move2 in leaf_opposition_moves:
                    leaf.create_child([move2, move1])

            leaf.action_space_size = (len(predicted_moves), len(leaf_opposition_moves))

            leaf.backup_value(Q_val, up_to=self.root)
            leaf.is_expanded = True
            break

    def predict_agent_moves(self, state, depth, agentType, idx):
        feat_mat = self.backbone.step_model.step(self.one_hot_state(state))
        actions = []
        list_actions = []
        qval = 0.0
        if agentType == "Predator":
            for i in range(len(state[0])):
                pi, v = self.predator_netw[state[0][i]].step_model.step(feat_mat, depth)
                pi = pi.data.cpu().numpy()
                move_idx = np.argmax(pi)
                actions.append(self.env.actions[state[0][i]][move_idx])
                if i == idx:
                    qval = v.data.cpu().numpy()

            for act in enumerate(self.env.actions[state[0][idx]]):
                actions[idx] = act
                list_actions.append(list(actions))
            return list_actions, qval
        else:
            for i in range(len(state[1])):
                pi, v = self.prey_netw[state[1][i]].step_model.step(feat_mat, depth)
                pi = pi.data.cpu().numpy()
                move_idx = np.argmax(pi)
                actions.append(self.env.actions[state[1][i]][move_idx])
                if i == idx:
                    qval = v.data.cpu().numpy()
            for act in self.env.actions[state[1][idx]]:
                actions[idx] = act
                list_actions.append(list(actions))
            return list_actions, qval

    def one_hot_state(self, state):
        temp_state = np.zeros([self.env.n_nodes, len(state[0]) + len(state[1])], dtype=float)
        for i in range(len(state[0])):
            if state[0][i] != -1:
                temp_state[state[0][i]][i] = 1.0
        for i in range(len(state[1])):
            if state[1][i] != -1:
                temp_state[state[1][i]][len(state[0]) + i] = 1.0
        return temp_state

    def find_gibbs_moves(self, state, active, limit):
        qval = 0.0
        todo = 'boltzman'
        gorge = {}

        actions = [[], []]
        list_actions = []
        for i in range(len(state[0])):
            actions[0].append((state[0][i], state[0][i]))
        for i in range(len(state[1])):
            if state[1][i] == -1:
                actions[1].append((-1, -1))
            else:
                actions[1].append((state[1][i], state[0][i]))

        if active == "Predator":
            for lim in range(limit):
                for i in range(len(state[0])):
                    actions[0][i] = (state[0][i], state[0][i])
                    mon = self.env.next_state(state, actions)
                    feat_mat = self.backbone.step_model.step(self.one_hot_state(mon))
                    probs, val = self.predator_netw[mon[0][i]].step_model.step(feat_mat, self.root.depth)
                    if lim == 0 and i == 0:
                        qval = val.data.cpu().numpy()
                    probs = probs.data.cpu().numpy()
                    if todo == "boltzman":
                        for j in range(len(probs)):
                            if j == 0:
                                continue
                            probs[j] = probs[j - 1] + probs[j]
                        selection = rd.random()
                        what = probs.searchsorted(selection)
                        if what >= len(self.env.actions[mon[0][i]]):
                            print("I MESSED UP", what, len(self.env.actions[mon[0][i]]), probs)
                        move = self.env.actions[mon[0][i]][what]
                        actions[0][i] = move
                    elif todo == "greedy":
                        what = np.argmax(probs)
                        move = self.env.actions[mon[0][i]][what]
                        actions[0][i] = move
                if str(actions) not in gorge:
                    list_actions.append(actions[0])
                    gorge[str(actions)] = 1.0
            return list_actions, qval

        else:
            for lim in range(limit):
                for i in range(len(state[1])):
                    if state[1][i] == -1:
                        continue
                    actions[1][i] = (state[1][i], state[1][i])
                    mon = self.env.next_state(state, actions)
                    feat_mat = self.backbone.step_model.step(self.one_hot_state(mon))
                    probs, val = self.prey_netw[mon[1][i]].step_model.step(feat_mat, self.root.depth)
                    if lim == 0 and i == 0:
                        qval = val.data.cpu().numpy()
                    probs = probs.data.cpu().numpy()
                    if todo == "boltzman":
                        for j in range(len(probs)):
                            if j == 0:
                                continue
                            probs[j] = probs[j - 1] + probs[j]
                        selection = rd.random()
                        what = probs.searchsorted(selection)
                        move = self.env.actions[mon[1][i]][what]
                        actions[1][i] = move
                    elif todo == "greedy":
                        what = np.argmax(probs)
                        move = self.env.actions[mon[1][i]][what]
                        actions[1][i] = move
                if str(actions) not in gorge:
                    list_actions.append(actions[1])
                    gorge[str(actions)] = 1.0
            return list_actions, qval

    def pick_action(self, idx):
        # the idea is to pick the best 'move' after the search and send it over
        for key in self.root.child_N:
            self.root.child_N[key] = self.root.children[key].N

        compressed = [0.0] * self.root.action_space_size[0]
        pre = list(self.root.child_N.values())
        for i in range(len(pre)):
            compressed[int(i / self.root.action_space_size[1])] += pre[i]
        compressed = np.array(compressed)
        child_number = np.argmax(compressed)
        child_number = child_number * self.root.action_space_size[1]
        child = self.root.children[self.root.idx_action[child_number]]

        # amar state
        probs = compressed / np.sum(compressed)

        if self.agentType == "Predator":
            return (self.root.state[0][idx], child.state[0][idx]), probs
        else:
            return (self.root.state[1][idx], child.state[1][idx]), probs


class SuperMCTS:
    def __init__(self, predator_netw, prey_netw, backbone, env, num_simulations):
        self.predator_netw = predator_netw
        self.prey_netw = prey_netw
        self.backbone = backbone
        self.env = env
        self.num_simulations = num_simulations
        self.move_threshold = None
        self.root = None
        self.mcts_predator = None
        self.mcts_list_prey = []
        self.searches_pi_predator = []
        self.searches_pi_prey = []
        self.sts_predator = []
        self.sts_prey = []
        self.z_val_predator = []
        self.z_val_prey = []
        self.moves_curr_predator = []
        self.moves_curr_prey = []
        self.idx_prey = []
        for i in range(self.env.n_nodes):
            self.searches_pi_predator.append([])
            self.searches_pi_prey.append([])
            self.sts_predator.append([])
            self.sts_prey.append([])
            self.z_val_predator.append([])
            self.z_val_prey.append([])
            self.moves_curr_predator.append([])
            self.moves_curr_prey.append([])
            self.idx_prey.append([])

    def initialize_search(self):
        # ~~~ initialize root
        self.root = MCTSNode(self.env.init_state, self.env)

        self.mcts_predator = C_MCTS(self.predator_netw, self.prey_netw, self.backbone, self.env, "Predator", -1, 3)
        for i in range(len(self.env.init_state[1])):
            self.mcts_list_prey.append(
                DC_MCTS(self.predator_netw, self.prey_netw, self.backbone, self.env, "Prey", i, 3))
        self.move_threshold = MOVE_THRESHOLD

    def progress(self, move_limit):
        progression = [self.root.state]
        poch = 0
        while not self.root.is_done("SuperMCTS", -1) and poch < move_limit:
            poch = poch + 1
            # ~~~ dirch noise left
            self.mcts_predator.root = MCTSNode(self.root.state, self.env)
            self.mcts_predator.root.depth = self.root.depth
            for i in range(len(self.mcts_list_prey)):
                if self.root.state[1][i] != -1:
                    self.mcts_list_prey[i].root = MCTSNode(self.root.state, self.env)
                    self.mcts_list_prey[i].root.depth = self.root.depth

            # chalaite thako sim bar
            x = datetime.datetime.now()
            for sim in range(self.num_simulations):
                for mul in range(3):
                    self.mcts_predator.tree_search()
                    # print("dot")
                for i in range(len(self.mcts_list_prey)):
                    if self.root.state[1][i] != -1:
                        self.mcts_list_prey[i].tree_search()
                        # print("dot")
                # y = datetime.datetime.now() - x
                # print(sim, y.microseconds)

            y = datetime.datetime.now() - x
            print("-> ", poch, y.seconds)
            # action niye kochlakochli
            new_predator_actions = []
            # print(self.mcts_predator.root.child_action_score)
            # print(len(self.mcts_predator.root.child_N))
            # print(self.mcts_predator.root.child_N.values())
            for i in range(len(self.root.state[0])):
                state, move, probs = self.mcts_predator.pick_action(i)
                new_predator_actions.append(move)
                # print("areh bhai", move)
                self.sts_predator[self.root.state[0][i]].append(self.mcts_predator.one_hot_state(state))
                self.searches_pi_predator[self.root.state[0][i]].append(probs)
                self.moves_curr_predator[self.root.state[0][i]].append(self.root.depth)
            new_prey_actions = []
            for i in range(len(self.mcts_list_prey)):
                if self.root.state[1][i] != -1:
                    # print("pailam ", self.root.state[1][i], self.mcts_list_prey[i].root.child_action_score,
                    #       self.mcts_list_prey[i].root.child_N)
                    move, probs = self.mcts_list_prey[i].pick_action(i)
                    new_prey_actions.append(move)
                    # if poch == 1:
                    #     print("pr move ", i, self.root.state[1][i], probs,
                    #           np.array(list(self.mcts_list_prey[i].root.child_prior.values())))
                    self.sts_prey[self.root.state[1][i]].append(self.mcts_list_prey[i].one_hot_state(self.root.state))
                    self.searches_pi_prey[self.root.state[1][i]].append(probs)
                    self.moves_curr_prey[self.root.state[1][i]].append(self.root.depth)
                    self.idx_prey[self.root.state[1][i]].append(i)
            new_state = self.env.next_state(self.root.state, [new_predator_actions, new_prey_actions])

            self.root = MCTSNode(new_state, self.env, parent=self.root)
            progression.append(self.root.state)

        for node in range(self.env.n_nodes):
            for state in self.sts_predator[node]:
                self.z_val_predator[node].append(self.env.get_return(self.root.state, "Predator", 0, state))
            for i, state in enumerate(self.sts_prey[node]):
                self.z_val_prey[node].append(
                    self.env.get_return(self.root.state, "Prey", self.idx_prey[node][i], state))
        return progression


def execute_episode(predator_netw, prey_netw, backbone, num_simulations, env, move_limit=15):
    mcts = SuperMCTS(predator_netw, prey_netw, backbone, env, num_simulations)

    mcts.initialize_search()
    progression = mcts.progress(move_limit)

    return mcts.searches_pi_predator, mcts.searches_pi_prey, mcts.sts_predator, mcts.sts_prey, mcts.z_val_predator, mcts.z_val_prey, mcts.moves_curr_predator, mcts.moves_curr_prey, progression
