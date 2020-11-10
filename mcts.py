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
        self.child_N = np.zeros([len(self.actions)], dtype=np.float32)
        self.child_W = np.zeros([len(self.actions)], dtype=np.float32)
        self.original_prior = np.zeros([len(self.actions)], dtype=np.float32)
        self.child_prior = np.zeros([len(self.actions)], dtype=np.float32)
        self.children = {}
        self.N = 0.0
        self.W = 0.0

        self.action_idx = {}
        self.idx_action = {}
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
        # print("-", self.type, self.child_prior, self.child_N, self.state, self)
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
        # print(self.child_Q, self.child_U)
        return self.child_Q + self.child_U

    def is_done(self):
        return self.env.is_done_state(self.state, self.depth)

    def create_child(self, agentType, idx, actions):

        new_state = self.state
        for action in actions[1]:
            new_state = self.env.next_state(new_state, "Prey", idx, action)
        for action in actions[0]:
            new_state = self.env.next_state(new_state, "Predator", idx, action)

        if agentType == "Predator":
            self.children[actions[0][idx]] = MCTSNode(new_state, self.env, self.env.actions[actions[0][idx][1]],
                                                      prev_action=self.action_idx[actions[0][idx][1]], parent=self)
            return self.children[actions[0][idx]]
        else:
            self.children[actions[1][idx]] = MCTSNode(new_state, self.env, self.env.actions[actions[1][idx][1]],
                                                      prev_action=self.action_idx[actions[1][idx][1]], parent=self)
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

            current = self.root
            while True:
                current.N += 1
                if not current.is_expanded:
                    break
                if current.is_done():
                    break
                best_move = np.argmax(current.child_action_score)
                if current.idx_action[best_move] not in current.children:
                    temp_moves = current.predicted_moves
                    if self.agentType == "Predator":
                        temp_moves[0][self.idx] = current.idx_action[best_move]
                    else:
                        temp_moves[1][self.idx] = current.idx_action[best_move]

                    current = current.create_child(temp_moves)
                else:
                    current = current.children[current.idx_action[best_move]]

            # leaf paoar por
            if current.is_done():
                if self.agentType == "Predator":
                    score = self.env.get_return(current.state, self.agentType, self.idx, self.root.state)
                else:
                    score = self.env.get_return(current.state, self.agentType, self.idx)
                current.backup_value(score, up_to=self.root)
                continue
            # new node
            current.predicted_moves, Q_val = self.predict_agent_moves(current.state, current.depth,
                                                                      self.agentType, self.idx)
            if self.agentType == "Predator":
                current.original_prior = current.predicted_moves[0][self.idx]
            else:
                current.original_prior = current.predicted_moves[1][self.idx]
            current.backup_value(Q_val, up_to=self.root)
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
        feat_mat = self.backbone.step(self.one_hot_state(state))
        predator_moves = []
        prey_moves = []
        Q_val = None
        for i in range(len(state[0])):
            pi, v = self.predator_netw[state[0][i]].step(feat_mat, depth)
            pi = pi.data.numpy()
            if agentType == "Predator" and i == idx:
                Q_val = v.data.numpy()
            move_idx = np.argmax(pi)
            predator_moves.append(self.env.actions[state[0][i]][move_idx])
        for i in range(len(state[1])):
            pi, v = self.prey_netw[state[1][i]].step(feat_mat, depth)
            pi = pi.data.numpy()
            if agentType == "Prey" and i == idx:
                Q_val = v.data.numpy()
            move_idx = np.argmax(pi)
            prey_moves.append(self.env.actions[state[1][i]][move_idx])
        return [predator_moves, prey_moves], Q_val

    def one_hot_state(self, state):
        temp_state = np.zeors([self.env.n_nodes, len(state[0]) + len(state[1])], dtype=float)
        for i in range(len(state[0])):
            temp_state[state[0][i]][i] = 1.0
        for i in range(len(state[1])):
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

    def progress(self):
        while not self.root.is_done():
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
                    self.mcts_list_predator[i].search_tree()
                for i in range(len(self.mcts_list_prey)):
                    if self.root.state[1][i] != -1:
                        self.mcts_list_prey[i].search_tree()

            # action niye kochlakochli
            new_state = self.root.state
            for i in range(len(self.mcts_list_prey)):
                if self.root.state[1][i] != -1:
                    move, probs = self.mcts_list_prey[i].pich_action()
                    new_state = self.env.next_state(new_state, "Prey", i, move)
                    self.sts_prey[self.root.state[1][i]].append(self.root.state)
                    self.searches_pi_prey[self.root.state[1][i]].append(probs)
                    self.moves_curr_prey[self.root.state[1][i]].append(self.root.depth)
                    self.idx_prey[self.root.state[1][i]].append(i)
            for i in range(len(self.mcts_list_predator)):
                move, probs = self.mcts_list_predator[i].pick_action()
                new_state = self.env.next_state(new_state, "Predator", i, move)
                self.sts_predator[self.root.state[0][i]].append(self.root.state)
                self.searches_pi_predator[self.root.state[0][i]].append(probs)
                self.moves_curr_predator[self.root.state[0][i]].append(self.root.depth)

            self.root = MCTSNode(new_state, self.env)

        for node in range(self.env.n_nodes):
            for state in self.sts_predator[node]:
                self.z_val_predator[node].append(self.env.get_return(self.root.state, "Predator", 0, state))
            for state, i in enumerate(self.sts_prey[node]):
                self.z_val_prey[node].append(
                    self.env.get_return(self.root.state, "Prey", self.idx_prey[node][i], state))


def execute_episode(predator_netw, prey_netw, backbone, num_simulations, env):
    mcts = SuperMCTS(predator_netw, prey_netw, backbone, env, num_simulations)

    mcts.initialize_search()
    mcts.progress()

    return mcts.searches_pi_predator, mcts.searches_pi_prey, mcts.sts_predator, mcts.sts_prey, mcts.z_val_predator, mcts.z_val_prey, mcts.moves_curr_predator, mcts.moves_curr_prey
