from copy import deepcopy

import numpy as np

from relnet.agent.baseline.heuristic_agent import HeuristicAgent
from relnet.evaluation.eval_utils import eval_on_dataset
from relnet.state.graph_state import S2VGraph


class BestResponseAgent(HeuristicAgent):
    algorithm_name = 'best_response'
    is_deterministic = False

    def __init__(self, environment):
        super().__init__(environment)

    def post_env_setup(self):
        super().post_env_setup()
        self.player_orders = []
        for g in self.environment.g_list:
            player_labels = list(g.node_labels)
            self.local_random.shuffle(player_labels)
            self.player_orders.append(player_labels)

    def run_heuristic_strategy(self, original_g, i):
        g = original_g.copy()

        player_list = g.node_labels
        player_orders = self.player_orders[i]
        N = len(player_orders)
        random_effort_profile = np.random.randint(0, 2, size=N)
        g.effort_levels = random_effort_profile

        if not self.is_mis(g, g.effort_levels):
            br_output, final_g = self.run_best_response(N, g, player_list, player_orders)
        else:
            br_output = list(g.node_labels[g.effort_levels == 1])
            final_g = g
        return br_output, final_g

    def run_best_response(self, N, g, player_list, player_orders):
        effort_nodes = None
        t = 0
        player_turn = 0
        while True:
            unsatisfied_nodes, _ = self.find_unsatisfied_nodes(g, player_list)

            if len(unsatisfied_nodes) == 0:
                effort_nodes = g.node_labels[g.effort_levels == 1]
                break
            else:
                player = player_orders[player_turn]
                satisfied, current_action = self.is_node_satisfied(g, player)
                if not satisfied:
                    g.effort_levels[player] = int(not current_action)

                player_turn += 1
                if player_turn == N:
                    player_turn = 0

                t += 1
        assert self.is_mis(g, g.effort_levels)
        return list(effort_nodes), g

    def setup(self, options, hyperparams):
        super().setup(options, hyperparams)

    @classmethod
    def get_default_hyperparameters(cls):
        return {}


class PayoffTransferAgent(BestResponseAgent):
    algorithm_name = 'payoff_transfer'
    is_deterministic = False

    def __init__(self, environment):
        super().__init__(environment)

    def eval(self, g_list,
             initial_obj_values=None,
             validation=False,
             make_action_kwargs=None):

        # need to explicitly override evaluation of method,
        # since equilibria in this setting are not necessarily MIS.
        eval_nets = [deepcopy(g) for g in g_list]
        if self.environment.use_zero_initial_obj():
            initial_obj_values = np.zeros(len(eval_nets))

        if initial_obj_values is None:
            obj_values = self.environment.get_objective_function_values(g_list)
        else:
            obj_values = initial_obj_values

        self.environment.setup(g_list, obj_values, training=False)
        self.post_env_setup()

        final_gs = []

        for i in range(len(self.environment.g_list)):
            g = self.environment.g_list[i]
            strategy_profiles, final_g = self.run_heuristic_strategy(g, i)
            final_gs.append(final_g)

        final_obj_values = self.environment.get_objective_function_values(final_gs)
        return eval_on_dataset(initial_obj_values, final_obj_values)

    def post_env_setup(self):
        super().post_env_setup()

    def run_best_response(self, N, g, player_list, player_orders):
        t = 0
        player_turn = 0
        steps_no_change = 0

        while True:
            player = player_orders[player_turn]
            player_effort_cost = g.effort_costs[player]
            neighbors = list(g.neighbors[player])

            change_made = self.on_strategy_select(g, neighbors, player, player_effort_cost)
            if change_made:
                steps_no_change = 0
            else:
                steps_no_change += 1

            if steps_no_change == N:
                unsatisfied_nodes, _ = self.find_unsatisfied_nodes(g, player_list)
                break

            player_turn += 1
            if player_turn == N:
                player_turn = 0
            t += 1

        effort_nodes = g.node_labels[g.effort_levels == 1]
        return list(effort_nodes), g

    def on_strategy_select(self, g, neighbors, player, player_effort_cost):
        change_made = False

        f_0 = self.all_nb_0_player_0(g, player, neighbors)
        t_1 = self.nb_1_player_1(g, player, neighbors)
        if f_0:
            g.effort_levels[player] = 1
            change_made = True
        elif t_1:
            # send strategy change to all neighbors and wait for replies
            payoffs = np.zeros(len(neighbors), dtype=np.float32)
            for i, neighbor in enumerate(neighbors):
                payoff = self.send_strategy_change(g, player, neighbor)  # , transfers)
                payoffs[i] = payoff

            payoff_sum = np.sum(payoffs)

            if player_effort_cost > payoff_sum:
                g.effort_levels[player] = 0
                change_made = True

        return change_made

    def send_strategy_change(self, g, src, dest):#, transfers):
        neighbors = list(g.neighbors[dest])
        one_nb_1_player_0 = self.one_nb_1_player_0(g, dest, neighbors)
        if one_nb_1_player_0:
            l_i = g.effort_costs[dest]
        else:
            l_i = 0

        return l_i

    @classmethod
    def get_default_hyperparameters(cls):
        return {}