import math
import numpy as np

# exploration constants
c_PUCT = 1.38
# Dirch noise
D_NOISE_ALPHA = 0.1
# Stochastic move number
MOVE_THRESHOLD = 0


class MCTSNode:
    def __init__(self, state, env, actions=None, prev_action=None, parent=None, predicted_moves=None, predicted_Q=None):
        self.env = env
        if parent is None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1
        self.parent = parent
        self.prev_action = prev_action
        self.state = state
        self.is_expanded = False
        self.predicted_moves = predicted_moves
        self.predicted_Q = predicted_Q
        self.actions = actions
        self.children = {}
        self.N = 0.0
        self.W = 0.0

        self.action_idx = {}
        self.idx_action = {}

        if self.actions is None:
            self.child_N = None
            self.child_W = None
            self.original_prior = None
            self.child_prior = None
        else:
            self.child_N = np.zeros([len(self.actions)], dtype=np.float32)
            self.child_W = np.zeros([len(self.actions)], dtype=np.float32)
            self.original_prior = np.zeros([len(self.actions)], dtype=np.float32)
            self.child_prior = np.zeros([len(self.actions)], dtype=np.float32)
            for i in range(len(self.actions)):
                self.action_idx[self.actions[i]] = i
                self.idx_action[i] = self.actions[i]

    @property
    def Q(self):
        """
        Returns the current action value of the node.
        """
        return self.W / (1 + self.N)

    @property
    def child_Q(self):
        for i in range(len(self.actions)):
            if self.actions[i] in self.children:
                self.child_N[i] = self.children[self.actions[i]].N
                self.child_W[i] = self.children[self.actions[i]].W
        return self.child_W / (1 + self.child_N)

    @property
    def child_U(self):
        for i in range(len(self.actions)):
            if self.actions[i] in self.children:
                self.child_N[i] = self.children[self.actions[i]].N
        return (c_PUCT * math.sqrt(1 + self.N) *
                self.child_prior / (1 + self.child_N))

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

    def select_leaf(self, agentType, idx):
        current = self
        while True:
            current.N += 1
            if not current.is_expanded:
                break
            if current.is_done(agentType, idx):
                break
            best_move = np.argmax(current.child_action_score)
            if current.idx_action[best_move] not in current.children:
                temp_moves = self.select_moves_from_pi(current.predicted_moves, current.state)
                if agentType == "Predator":
                    temp_moves[0][idx] = current.idx_action[best_move]
                else:
                    temp_moves[1][idx] = current.idx_action[best_move]
                current = current.create_child(agentType, idx, temp_moves)
            else:
                current = current.children[current.idx_action[best_move]]
        return current

    def select_moves_from_pi(self, predicted_moves, state):

        predator_moves = []
        prey_moves = []
        for i in range(len(predicted_moves[0])):
            pi = predicted_moves[0][i]
            move_idx = np.argmax(pi)
            predator_moves.append(self.env.actions[state[0][i]][move_idx])
        for i in range(len(state[1])):
            if state[1][i] == -1:
                prey_moves.append((-1, -1))
                continue
            pi = predicted_moves[1][i]
            move_idx = np.argmax(pi)
            prey_moves.append(self.env.actions[state[1][i]][move_idx])
        return [predator_moves, prey_moves]

    def create_child(self, agentType, idx, actions):
        new_state = self.env.next_state(self.state, actions)

        if agentType == "Predator":
            self.children[actions[0][idx]] = MCTSNode(new_state, self.env, self.env.actions[actions[0][idx][1]],
                                                      prev_action=self.action_idx[actions[0][idx]], parent=self)
            return self.children[actions[0][idx]]
        else:
            self.children[actions[1][idx]] = MCTSNode(new_state, self.env, self.env.actions[actions[1][idx][1]],
                                                      prev_action=self.action_idx[actions[1][idx]], parent=self)
            return self.children[actions[1][idx]]

    def backup_value(self, value, up_to):
        self.W += value
        if self.parent is None or self is up_to:
            return
        self.parent.backup_value(value, up_to)
        # self.parent.backup_value(value*1.003, up_to)
        # self.parent.backup_value(value*0.997, up_to)


class MCTS:
    def __init__(self, predator_netw, prey_netw, backbone, env, agentType, idx):
        self.predator_netw = predator_netw
        self.prey_netw = prey_netw
        self.backbone = backbone
        self.env = env
        self.agentType = agentType
        self.idx = idx
        self.root = None

    def tree_search(self):
        failsafe = 0
        while failsafe < 10:
            failsafe += 1
            leaf = self.root.select_leaf(self.agentType, self.idx)

            # leaf paoar por
            if leaf.is_done(self.agentType, self.idx):
                if self.agentType == "Predator":
                    score = self.env.get_return(leaf.state, self.agentType, self.idx, self.root.state)
                else:
                    score = self.env.get_return(leaf.state, self.agentType, self.idx)
                leaf.backup_value(score, up_to=self.root)
                # while self.root.parent is not None:
                #     self.root = self.root.parent
                continue
            # new node
            leaf.predicted_moves, Q_val = self.predict_agent_moves(leaf.state, leaf.depth,
                                                                   self.agentType, self.idx)
            if self.agentType == "Predator":
                leaf.child_prior = leaf.predicted_moves[0][self.idx]
            else:
                leaf.child_prior = leaf.predicted_moves[1][self.idx]
            leaf.backup_value(Q_val, up_to=self.root)
            leaf.is_expanded = True
            # while self.root.parent is not None:
            #     self.root = self.root.parent
            break

    def pick_action(self):
        # the idea is to pick the best 'move' after the search and send it over
        for i in range(len(self.root.actions)):
            if self.root.actions[i] in self.root.children:
                self.root.child_N[i] = self.root.children[self.root.actions[i]].N

        move_idx = np.argmax(self.root.child_N)
        probs = self.root.child_N
        probs = probs / np.sum(probs)
        return self.root.idx_action[move_idx], probs

    def predict_agent_moves(self, state, depth, agentType, idx):
        feat_mat = self.backbone.step_model.step(self.one_hot_state(state))
        predator_moves = []
        prey_moves = []
        Q_val = None
        for i in range(len(state[0])):
            pi, v = self.predator_netw[state[0][i]].step_model.step(feat_mat, depth)
            pi = pi.data.numpy()
            if agentType == "Predator" and i == idx:
                Q_val = v.data.numpy()
            predator_moves.append(pi)
        for i in range(len(state[1])):
            if state[1][i] == -1:
                prey_moves.append([])
                continue
            pi, v = self.prey_netw[state[1][i]].step_model.step(feat_mat, depth)
            pi = pi.data.numpy()
            if agentType == "Prey" and i == idx:
                Q_val = v.data.numpy()
            prey_moves.append(pi)
        return [predator_moves, prey_moves], Q_val

    def one_hot_state(self, state):
        temp_state = np.zeros([self.env.n_nodes, len(state[0]) + len(state[1])], dtype=float)
        for i in range(len(state[0])):
            if state[0][i] != -1:
                temp_state[state[0][i]][i] = 1.0
        for i in range(len(state[1])):
            if state[1][i] != -1:
                temp_state[state[1][i]][len(state[0]) + i] = 1.0
        return temp_state


class SuperMCTS:
    def __init__(self, predator_netw, prey_netw, backbone, env, num_simulations):
        self.predator_netw = predator_netw
        self.prey_netw = prey_netw
        self.backbone = backbone
        self.env = env
        self.num_simulations = num_simulations
        self.move_threshold = None
        self.root = None
        self.mcts_list_predator = []
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

        for i in range(len(self.env.init_state[0])):
            self.mcts_list_predator.append(
                MCTS(self.predator_netw, self.prey_netw, self.backbone, self.env, "Predator", i))
        for i in range(len(self.env.init_state[1])):
            self.mcts_list_prey.append(MCTS(self.predator_netw, self.prey_netw, self.backbone, self.env, "Prey", i))
        self.move_threshold = MOVE_THRESHOLD

    def progress(self, move_limit):
        progression = [self.root.state]
        poch = 0
        while not self.root.is_done("Super", -1) and poch<move_limit:
            poch = poch + 1
            # ~~~ dirch noise left
            for i in range(len(self.mcts_list_predator)):
                self.mcts_list_predator[i].root = MCTSNode(self.root.state, self.env,
                                                           self.env.actions[self.root.state[0][i]])
            for i in range(len(self.mcts_list_prey)):
                if self.root.state[1][i] != -1:
                    self.mcts_list_prey[i].root = MCTSNode(self.root.state, self.env,
                                                           self.env.actions[self.root.state[1][i]])

            # chalaite thako sim bar
            for sim in range(self.num_simulations):
                for i in range(len(self.mcts_list_predator)):
                    temp_root = self.mcts_list_predator[i].root
                    self.mcts_list_predator[i].tree_search()
                    self.mcts_list_predator[i].root = temp_root
                for i in range(len(self.mcts_list_prey)):
                    if self.root.state[1][i] != -1:
                        temp_root = self.mcts_list_prey[i].root
                        self.mcts_list_prey[i].tree_search()
                        self.mcts_list_prey[i].root = temp_root

            # action niye kochlakochli
            new_predator_actions = []
            for i in range(len(self.mcts_list_predator)):
                move, probs = self.mcts_list_predator[i].pick_action()
                new_predator_actions.append(move)
                if poch == 1:
                    print("pd move ", i, self.root.state[0][i], probs, self.mcts_list_predator[i].root.child_prior)
                self.sts_predator[self.root.state[0][i]].append(self.mcts_list_predator[i].one_hot_state(self.root.state))
                self.searches_pi_predator[self.root.state[0][i]].append(probs)
                self.moves_curr_predator[self.root.state[0][i]].append(self.root.depth)
            new_prey_actions = []
            for i in range(len(self.mcts_list_prey)):
                if self.root.state[1][i] != -1:
                    move, probs = self.mcts_list_prey[i].pick_action()
                    new_prey_actions.append(move)
                    if poch == 1:
                        print("pr move ", i, self.root.state[1][i], probs,
                              self.mcts_list_prey[i].root.child_prior)
                    self.sts_prey[self.root.state[1][i]].append(self.mcts_list_prey[i].one_hot_state(self.root.state))
                    self.searches_pi_prey[self.root.state[1][i]].append(probs)
                    self.moves_curr_prey[self.root.state[1][i]].append(self.root.depth)
                    self.idx_prey[self.root.state[1][i]].append(i)
            new_state = self.env.next_state(self.root.state, [new_predator_actions, new_prey_actions])

            self.root = MCTSNode(new_state, self.env)
            progression.append(self.root.state)

        for node in range(self.env.n_nodes):
            for state in self.sts_predator[node]:
                self.z_val_predator[node].append(self.env.get_return(self.root.state, "Predator", 0, state))
            for i, state in enumerate(self.sts_prey[node]):
                self.z_val_prey[node].append(
                    self.env.get_return(self.root.state, "Prey", self.idx_prey[node][i], state))
        return progression


def execute_episode(predator_netw, prey_netw, backbone, num_simulations, env, move_limit = 10):
    mcts = SuperMCTS(predator_netw, prey_netw, backbone, env, num_simulations)

    mcts.initialize_search()
    progression = mcts.progress(move_limit)

    return mcts.searches_pi_predator, mcts.searches_pi_prey, mcts.sts_predator, mcts.sts_prey, mcts.z_val_predator, mcts.z_val_prey, mcts.moves_curr_predator, mcts.moves_curr_prey, progression
