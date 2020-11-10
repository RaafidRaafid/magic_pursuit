import numpy as np
from static_env import staticEnv


class gameEnv(staticEnv):

    def __init__(self, id):
        self.adj, self.init_state = self.readFile("data/" + str(id))
        self.n_nodes = self.adj.shape[0]
        self.degree = np.zeros(self.n_nodes, dtype=int)
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                self.degree[i] += (self.adj[i][j] > 0)
        self.actions = []
        for i in range(self.n_nodes):
            temp = []
            for j in range(self.n_nodes):
                if self.adj[i][j] > 0.0:
                    temp.append((i, j))

            self.actions.append(temp)

    @staticmethod
    def next_state(state, agentType, idx, action):
        if agentType == 'Predator':
            state[0][idx] = action[1]
            death = np.where(state[1] == action[1])
            state[1][death[0]] = -1.0
        elif agentType == 'Prey':
            death = np.where(state[0] == action[1])
            if len(death[0]) > 0:
                state[1][idx] = -1.0
            else:
                state[1][idx] = action[1]
        return state

    @staticmethod
    def is_done_state(state, step_idx):
        if np.sum(state[1]) == (-1)*len(state[1]):
            return True
        return step_idx >= 20

    @staticmethod
    def get_return(state, agentType, idx, root_state=None):
        if agentType == "Predator":
            return np.count_nonzero(state[1] == -1.0) - np.count_nonzero(root_state[1] == -1.0)
        elif agentType == "Prey":
            if state[1][idx] == -1.0:
                return -1.0
            else:
                return 1.0

    def readFile(self, filename):
        f = open("data/adj.txt")
        adj = []
        for line in f:
            arr = []
            line = line.split()
            for v in line:
                arr.append(float(v))
            adj.append(arr)
        f.close()
        f = open(filename + "pd.txt")
        pd = []
        for line in f:
            line = line.split()
            for v in line:
                pd.append(float(v))
        f.close()
        f = open(filename + "pr.txt")
        pr = []
        for line in f:
            line = line.split()
            for v in line:
                pr.append(float(v))
        f.close()
        return np.array(adj), [np.array(pd), np.array(pr)]
