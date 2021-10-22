from abc import abstractmethod, ABC

import numpy as np

from relnet.agent.base_agent import Agent


class BaselineAgent(Agent, ABC):
    is_trainable = False

    def make_actions(self, t, **kwargs):
        actions = []
        for i in range(len(self.environment.g_list)):
            action = self.pick_actions_using_strategy(t, i)
            actions.append(action)
        return actions

    def post_env_setup(self):
        pass

    def finalize(self):
        pass

    @abstractmethod
    def pick_actions_using_strategy(self, t, i):
        pass

class RandomAgent(BaselineAgent):
    algorithm_name = 'random'
    is_deterministic = False

    def pick_actions_using_strategy(self, t, i):
        return self.pick_random_actions(i)

    def say_hello(self):
        print("Hello world from Random Agent!")

class TargetHubsAgent(BaselineAgent):
    algorithm_name = 'target_hub'
    is_deterministic = True

    def pick_actions_using_strategy(self, t, i):
        g = self.environment.g_list[i]
        valid_acts = list(self.environment.get_valid_actions(g))

        if len(valid_acts) == 0:
            return -1

        degrees = g.get_degrees()
        act_degrees = np.array([degrees[node] for node in valid_acts])
        chosen_act = valid_acts[np.argmax(act_degrees)]
        return chosen_act

class TargetMinCostAgent(BaselineAgent):
    algorithm_name = 'target_min_cost'
    is_deterministic = True

    def pick_actions_using_strategy(self, t, i):
        g = self.environment.g_list[i]
        valid_acts = list(self.environment.get_valid_actions(g))

        if len(valid_acts) == 0:
            return -1

        effort_costs = g.effort_costs
        act_costs = np.array([effort_costs[node] for node in valid_acts])
        chosen_act = valid_acts[np.argmin(act_costs)]
        return chosen_act


class ExhaustiveSearchAgent(BaselineAgent):
    algorithm_name = 'exhaustive'
    is_deterministic = True

    def __init__(self, environment):
        super().__init__(environment)
        self.future_actions = {}

    def pick_actions_using_strategy(self, t, i):
        if t == 0:
            g_actions = self.run_search_for_graph(i)
            self.future_actions[i] = g_actions

        if t >= len(self.future_actions[i]):
            return -1

        return self.future_actions[i][t]

    def run_search_for_graph(self, i):
        best_acts = None

        g = self.environment.g_list[i]
        initial_value = self.environment.objective_function_values[0, i]
        best_difference = float("-inf")

        n = g.num_nodes
        for effort_levels in self.get_all_effort_seqs(n):
            g_copy = g.copy()
            nodes_to_select = g_copy.node_labels[(effort_levels)]

            valid_mis = True
            if len(nodes_to_select) == 0:
                valid_mis = False
            else:
                for node in nodes_to_select:
                    try:
                        g_copy = g_copy.select_node(node, in_place=True)
                    except BaseException:
                        valid_mis = False
                        break

            if not self.environment.is_state_terminal(g_copy):
                valid_mis = False

            if valid_mis:
                next_value = self.environment.get_objective_function_value(g_copy)
                diff = next_value - initial_value
                if diff > best_difference:
                    best_difference = diff
                    best_acts = nodes_to_select

        return list(best_acts)

    def get_all_effort_seqs(self, n):
        total_seqs = 2**n
        for seq_num in range(total_seqs):
            yield self.gen_effort_levels_sequence(seq_num, n)

    def gen_effort_levels_sequence(self, seq_num, n):
        return [bool(seq_num & (1 << offset)) for offset in range(n)]



