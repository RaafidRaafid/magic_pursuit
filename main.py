from environment import gameEnv
import numpy as np
from trainer import *
from NN import *
from replay_memory import ReplayMemory
from mcts import *
import random

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


    def show_graph(graph, state, i):
        color_map = ["yellow"] * graph.number_of_nodes()
        for idx in state[0]:
            color_map[idx] = "red"
        for idx in state[1]:
            if idx != -1:
                color_map[idx] = "green"

        nx.draw_networkx(graph, node_color=color_map)
        plt.savefig('step' + str(i) + '.png')
        plt.show()


    env = gameEnv(0)
    nPass = 64
    inputDim = len(env.init_state[0]) + len(env.init_state[1])

    backbone = BackboneTrainer(lambda: BackboneNN(inputDim, nPass, env.adj))
    trainer_predator = []
    trainer_prey = []
    for i in range(env.n_nodes):
        trainer_predator.append(PredictionTrainer(lambda: PredictionNN(nPass, env.degree[i]), backbone, env))
        trainer_prey.append(PredictionTrainer(lambda: PredictionNN(nPass, env.degree[i]), backbone, env))

    mem_predator = []
    mem_prey = []
    for i in range(env.n_nodes):
        mem_predator.append(ReplayMemory(200, {"sts": [env.n_nodes, len(env.init_state[0]) + len(env.init_state[1])],
                                               "pi": [env.degree[i]], "z": [], "moves_left": []}))
        mem_prey.append(ReplayMemory(200, {"sts": [env.n_nodes, len(env.init_state[0]) + len(env.init_state[1])],
                                           "pi": [env.degree[i]], "z": [], "moves_left": []}))

    # log = open("log.txt", "w+")
    for epoch in range(20):
        env_id = random.randint(0, 9)
        # env_id = 4
        env = gameEnv(env_id)
        print("epoch env", epoch, env_id)

        searches_pi_predator, searches_pi_prey, sts_predator, sts_prey, z_val_predator, z_val_prey, moves_curr_predator, moves_curr_prey, progression = execute_episode(
            trainer_predator, trainer_prey, backbone, 100, env)

        if epoch >= 0:
            # graph = make_graph(env.adj)
            for i, val in enumerate(progression):
                print(val)
                # show_graph(graph, val, i)

        # save and traing
        for i in range(env.n_nodes):
            mem_predator[i].add_all({"sts": sts_predator[i], "pi": searches_pi_predator[i], "z": z_val_predator[i],
                                     "moves_left": moves_curr_predator[i]})
            mem_prey[i].add_all(
                {"sts": sts_prey[i], "pi": searches_pi_prey[i], "z": z_val_prey[i], "moves_left": moves_curr_prey[i]})

            print("count ", i, mem_predator[i].count, mem_prey[i].count)
            # log.write(" cscount " + str(i) + " " + str(mem_predator[i].count) + " " + str(mem_prey[i].count) + "\n")

            if mem_predator[i].count > 4:
                batch = mem_predator[i].get_minibatch()
                lossP, lossV = trainer_predator[i].train(batch["sts"], batch["pi"], batch["z"], batch["moves_left"])
                print("predator ====>", lossP.cpu(), lossV.cpu())
                # log.write("predator ====>" + str(lossP) + str(lossV) + "\n")

            if mem_prey[i].count > 4:
                batch = mem_prey[i].get_minibatch()
                lossP, lossV = trainer_prey[i].train(batch["sts"], batch["pi"], batch["z"], batch["moves_left"])
                print("preya ====>", lossP.cpu(), lossV.cpu())
                # log.write("prey ====>" + str(lossP) + str(lossV) + "\n")

    # log.close()
    print("shofol")
