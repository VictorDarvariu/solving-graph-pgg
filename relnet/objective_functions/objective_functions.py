import numpy as np


class SocialWelfare(object):
    name = "social_welfare"
    upper_limit = 1.

    @staticmethod
    def compute(s2v_graph, **kwargs):
        utilities = extract_utilities(s2v_graph)
        return np.mean(utilities)


class MeanCost(object):
    name = "mean_cost"
    upper_limit = 1.

    @staticmethod
    def compute(s2v_graph, **kwargs):
        all_costs = np.zeros(s2v_graph.num_nodes, dtype=np.float32)
        effort_nodes = (s2v_graph.effort_levels == 1)
        all_costs[effort_nodes] = s2v_graph.effort_costs[effort_nodes]
        return np.mean(all_costs)


class Fairness(object):
    name = "fairness"
    upper_limit = 1.

    @staticmethod
    def compute(s2v_graph, **kwargs):
        utilities = extract_utilities(s2v_graph)
        gini = gini_coefficient(utilities)
        return 1 - gini

def extract_utilities(s2v_graph):
    utilities = np.zeros(s2v_graph.num_nodes, dtype=np.float32)
    utilities[s2v_graph.effort_levels == 0] = 1
    effort_nodes = (s2v_graph.effort_levels == 1)
    corresponding_costs = s2v_graph.effort_costs[effort_nodes]
    utilities[effort_nodes] = 1 - corresponding_costs
    return utilities


def gini_coefficient(xs):
    diffsum = 0
    for i, xi in enumerate(xs[:-1], 1):
        diffsum += np.sum(np.abs(xi - xs[i:]))
    return diffsum / (len(xs) ** 2 * np.mean(xs))