import numpy as np
from copy import deepcopy


class GraphMISEnv(object):
    def __init__(self, objective_function, objective_function_kwargs, heterogenous_cost=False):
        self.objective_function = objective_function
        self.original_objective_function_kwargs = objective_function_kwargs
        self.objective_function_kwargs = deepcopy(self.original_objective_function_kwargs)

        self.reward_eps = 1e-4
        self.reward_scale_multiplier = 100

        self.heterogenous_cost = heterogenous_cost

    def setup(self, g_list, initial_objective_function_values, training=False):
        self.g_list = g_list
        self.n_steps = 0

        self.exhausted_budgets = np.zeros(len(g_list), dtype=np.bool)
        self.training = training

        self.objective_function_values = np.zeros((2, len(self.g_list)), dtype=np.float)
        self.objective_function_values[0, :] = initial_objective_function_values
        self.objective_function_kwargs = deepcopy(self.original_objective_function_kwargs)
        self.rewards = np.zeros(len(g_list), dtype=np.float)

        if self.training:
            self.objective_function_values[0, :] = np.multiply(self.objective_function_values[0, :], self.reward_scale_multiplier)

    def assign_env_specific_properties(self, g_list):
        for g in g_list:
            g.assign_costs(self.heterogenous_cost)

    def pass_logger_instance(self, logger):
        self.logger_instance = logger

    def get_final_values(self):
        return self.objective_function_values[-1, :]

    def get_objective_function_value(self, s2v_graph):
        obj_function_value = self.objective_function.compute(s2v_graph, **self.objective_function_kwargs)
        return obj_function_value

    def get_objective_function_values(self, s2v_graphs):
        return np.array([self.get_objective_function_value(g) for g in s2v_graphs])

    @staticmethod
    def from_env_instance(env_instance):
        new_instance = GraphMISEnv(deepcopy(env_instance.objective_function),
                                   deepcopy(env_instance.objective_function_kwargs),
                                   deepcopy(env_instance.heterogenous_cost)
                                   )
        return new_instance

    @staticmethod
    def get_valid_actions(g):
        return g.nodes_not_covered

    @staticmethod
    def apply_action(g, action, in_place=True):
        return g.select_node(action, in_place=in_place)

    def exploratory_actions(self, agent_exploration_policy):
        act_list = []
        for i in range(len(self.g_list)):
            act = agent_exploration_policy(i)
            act_list.append(act)

        return act_list

    def step(self, actions):
        for i in range(len(self.g_list)):
            if not self.exhausted_budgets[i]:
                if actions[i] == -1:
                    if self.logger_instance is not None:
                        self.logger_instance.warn("budget not exhausted but trying to apply dummy action!")

                self.g_list[i] = self.apply_action(self.g_list[i], actions[i])
                g = self.g_list[i]

                if self.is_state_terminal(g):
                    self.exhausted_budgets[i] = True
                    objective_function_value = self.get_objective_function_value(self.g_list[i])
                    if self.training:
                        objective_function_value = objective_function_value * self.reward_scale_multiplier
                    self.objective_function_values[-1, i] = objective_function_value
                    reward = objective_function_value - self.objective_function_values[0, i]
                    if abs(reward) < self.reward_eps:
                        reward = 0.

                    self.rewards[i] = reward

        self.n_steps += 1

    def is_state_terminal(self, g):
        return len(g.nodes_not_covered) == 0

    def is_terminal(self):
        return np.all(self.exhausted_budgets)

    def get_num_mdp_substeps(self):
        return 1

    def get_num_node_feats(self):
        return 3

    def get_num_edge_feats(self):
        return 0

    def use_zero_initial_obj(self):
        return True

    def mark_exhausted(self, i):
        self.exhausted_budgets[i] = True

    # used in DQN
    def get_state_ref(self):
        return self.g_list

    # used in DQN
    def clone_state(self, indices=None):
        if indices is None:
            return [deepcopy(g) for g in self.g_list]
        else:
            cp_g_list = []
            for i in indices:
                cp_g_list.append(deepcopy(self.g_list[i]))
            return cp_g_list