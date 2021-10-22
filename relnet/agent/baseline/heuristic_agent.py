from abc import abstractmethod, ABC

import numpy as np

from relnet.agent.baseline.baseline_agent import BaselineAgent
from relnet.state.graph_state import S2VGraph


class HeuristicAgent(BaselineAgent, ABC):
    def __init__(self, environment):
        super().__init__(environment)

    def pick_actions_using_strategy(self, t, i):
        if t == 0:
            g = self.environment.g_list[i]
            strategy_profiles, _ = self.run_heuristic_strategy(g, i)
            self.stored_actions[i] = strategy_profiles
            return self.stored_actions[i][0]
        else:
            if t < len(self.stored_actions[i]):
                return self.stored_actions[i][t]
            return -1

    def post_env_setup(self):
        self.stored_actions = [None] * len(self.environment.g_list)

    def find_unsatisfied_nodes(self, g, query_nodes):
        unsatisfied_nodes = []
        efforts = []
        for node in query_nodes:
            satisfied, current_effort = self.is_node_satisfied(g, node)
            if not satisfied:
                unsatisfied_nodes.append(node)
                efforts.append(current_effort)
        return unsatisfied_nodes, efforts

    def is_node_satisfied(self, g, node):
        node_neighbours = list(g.neighbors[node])
        satisfied = not (self.all_nb_0_player_0(g, node, node_neighbours)
                     or self.nb_1_player_1(g, node, node_neighbours))

        node_effort = g.effort_levels[node]
        return satisfied, node_effort

    def all_nb_0_player_0(self, g, node, neighbours):
        neighbour_efforts = g.effort_levels[neighbours]
        node_effort = g.effort_levels[node]

        all_nb_0 = np.all(neighbour_efforts == 0)
        return node_effort == 0 and all_nb_0

    def all_nb_0_player_1(self, g, node, neighbours):
        neighbour_efforts = g.effort_levels[neighbours]
        node_effort = g.effort_levels[node]

        all_nb_0 = np.all(neighbour_efforts == 0)
        return node_effort == 1 and all_nb_0

    def nb_1_player_1(self, g, node, neighbours):
        neighbour_efforts = g.effort_levels[neighbours]
        node_effort = g.effort_levels[node]

        nb_playing_1 = np.any(neighbour_efforts == 1)
        return node_effort == 1 and nb_playing_1

    def any_nb_1_player_0(self, g, node, neighbours):
        neighbour_efforts = g.effort_levels[neighbours]
        node_effort = g.effort_levels[node]

        nb_playing_1 = np.any(neighbour_efforts == 1)
        return node_effort == 0 and nb_playing_1

    def one_nb_1_player_0(self, g, node, neighbours):
        neighbour_efforts = g.effort_levels[neighbours]
        node_effort = g.effort_levels[node]

        #print(f"neighbor effort sum is {np.sum(neighbour_efforts)}")
        nb_exactly_1 = (np.sum(neighbour_efforts) == 1)
        return node_effort == 0 and nb_exactly_1


    @abstractmethod
    def run_heuristic_strategy(self, original_g, i):
        pass

    def is_mis(self, current_g, effort_levels):
        try:
            effort_nodes = current_g.node_labels[effort_levels == 1]
            test_g = S2VGraph(current_g.to_networkx())
            for node in effort_nodes:
                test_g.select_node(node)

            n_not_covered = len(test_g.nodes_not_covered)
            is_mis = (n_not_covered == 0)
            return is_mis
        except BaseException:
            return False



