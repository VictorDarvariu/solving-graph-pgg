import numpy as np
from relnet.utils.config_utils import local_np_seed

"""
Simple tree node representation.
"""
class MCTSTreeNode(object):
    def __init__(self, state, parent_node, parent_action, valid_actions, depth=-1):
        self.state = state
        self.parent_node = parent_node
        self.parent_action = parent_action

        if self.parent_node is None:
            self.depth = depth
        else:
            self.depth = self.parent_node.depth + 1

        self.children = {}

        self.N = 0.0
        self.Q = 0.0

        self.valid_actions = np.array(valid_actions)
        self.num_valid_actions = len(valid_actions)
        self.actions_arr_index = {self.valid_actions[i]: i for i in range(self.num_valid_actions)}
        self.visited_actions = np.full(self.num_valid_actions, False, dtype=np.bool)

    def choose_action(self, random_seed):
        remaining_actions = self.valid_actions[~self.visited_actions]

        with local_np_seed(random_seed):
            chosen_action = np.random.choice(remaining_actions)

        chosen_idx = self.actions_arr_index[chosen_action]
        self.visited_actions[chosen_idx] = True
        return chosen_action

    def update_estimates(self, R):
        self.N += 1
        self.Q = self.Q + ((R - self.Q) / self.N)


    def __str__(self):
        return f"Node at {hex(id(self))} with Q={self.Q:.3f}, N={self.N}"

    def __repr__(self):
        return f"Node at {hex(id(self))} with Q={self.Q:.3f}, N={self.N}"
