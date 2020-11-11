from environment import gameEnv
import numpy as np
from trainer import *
from NN import *
from replay_memory import ReplayMemory
from mcts import *

import networkx as nx
import matplotlib.pyplot as plt

if __name__ == '__main__':

    def make_graph(adj):
        G = nx.Graph()
        G.add_nodes_from(list(range(adj.shape[0])))
        for i in range(adj.shape[0]):
            for j in range(adj.shape[0]):
                if adj[i][j] > 0.0:
                    G.add_edge(i, j)

        return G


    def show_graph(graph,state, i):
        color_map = ["yellow"] * graph.number_of_nodes()
        for idx in state[0]:
            color_map[idx] = "red"
        for idx in state[1]:
            if idx != -1:
                color_map[idx] = "green"

        nx.draw_networkx(graph, node_color=color_map)
        plt.savefig('step' + str(i) + '.png')
        plt.show()


    def one_hot_state(env, state):
        temp_state = np.zeors([env.n_nodes, len(state[0]) + len(state[1])], dtype=float)
        for i in range(len(state[0])):
            temp_state[state[0][i]][i] = 1.0
        for i in range(len(state[1])):
            temp_state[state[1][i]][len(state[0]) + i] = 1.0
        return temp_state


    env = gameEnv(0)
    nPass = 64
    inputDim = len(env.init_state[0]) + len(env.init_state[1])

    backbone = BackboneTrainer(lambda: BackboneNN(inputDim, nPass, env.adj))
    trainer_predator = []
    trainer_prey = []
    for i in range(env.n_nodes):
        trainer_predator.append(PredictionTrainer(lambda: PredictionNN(nPass, env.degree[i]), backbone, env))
        trainer_prey.append(PredictionTrainer(lambda: PredictionNN(nPass, env.degree[i]), backbone, env))

    searches_pi_predator, searches_pi_prey, sts_predator, sts_prey, z_val_predator, z_val_prey, moves_curr_predator, moves_curr_prey, progression = execute_episode(
        trainer_predator, trainer_prey, backbone, 500, env)

    graph = make_graph(env.adj)
    for i, val in enumerate(progression):
        print(val)
        show_graph(graph, val, i)

    print("shofol")
