import warnings
from copy import deepcopy

import networkx as nx
import numpy as np
import xxhash

from relnet.utils.config_utils import local_np_seed

class S2VGraph(object):
    UNIFORM_COST_VALUE = 0.5

    def __init__(self, g):
        self.num_nodes = g.number_of_nodes()
        self.node_labels = np.arange(self.num_nodes)
        self.all_nodes_set = set(self.node_labels)

        x, y = zip(*(sorted(g.edges())))
        self.num_edges = len(x)
        self.edge_pairs = np.ndarray(shape=(self.num_edges, 2), dtype=np.int32)
        self.edge_pairs[:, 0] = x
        self.edge_pairs[:, 1] = y
        self.edge_pairs = np.ravel(self.edge_pairs)

        self.effort_levels = np.zeros(self.num_nodes, dtype=np.int32)

        self.neighbors = {l: set(g[l]) for l in self.node_labels}
        self.nodes_not_covered = set(self.node_labels)
        self.selected_nodes = set()

    def select_node(self, node, in_place=True):
        if not in_place:
            g = self.copy()
        else:
            g = self

        g.effort_levels[node] = 1
        g.selected_nodes.add(node)
        g.nodes_not_covered.remove(node)
        node_neighbors = g.neighbors[node]
        g.nodes_not_covered.difference_update(node_neighbors)

        return g

    def assign_costs(self, heterogenous_cost):
        if heterogenous_cost:
            g_seed = get_graph_hash(self, size=32, include_selected=False)
            with local_np_seed(g_seed):
                self.effort_costs = np.random.uniform(low=0.0, high=1.0, size=self.num_nodes)
        else:
            self.effort_costs = np.full(self.num_nodes, self.UNIFORM_COST_VALUE, dtype=np.float32)

    def get_number_effort_nodes(self):
        return sum(self.effort_levels)

    def to_networkx(self):
        edges = self.convert_edges()
        g = nx.Graph()
        g.add_edges_from(edges)
        return g

    def convert_edges(self):
        return np.reshape(self.edge_pairs, (self.num_edges, 2))

    def get_degrees(self):
        g = self.to_networkx()
        return np.array([deg for (node, deg) in sorted(g.degree(), key=lambda deg_pair: deg_pair[0])])

    def display(self, ax=None):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            nx_graph = self.to_networkx()
            node_colors = []
            for n in self.node_labels:
                node_colors.append(
                    'g' if self.effort_levels[n] == 1 else ('r' if n in self.nodes_not_covered else 'b'))
            nx.draw_shell(nx_graph, node_color=node_colors, with_labels=True, ax=ax)


    def draw_to_file(self, filename):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_size_length = self.num_nodes / 5
        figsize = (fig_size_length, fig_size_length)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        self.display(ax=ax)
        fig.savefig(filename)
        plt.close()

    def get_adjacency_matrix(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            nx_graph = self.to_networkx()
            adj_matrix = np.asarray(nx.convert_matrix.to_numpy_matrix(nx_graph, nodelist=self.node_labels))

        return adj_matrix

    def copy(self):
        return deepcopy(self)

    def __repr__(self):
        gh = get_graph_hash(self, size=32, include_selected=False)
        return f"Graph State with hash {gh}"


def get_graph_hash(g, size=32, include_selected=True):
    if size == 32:
        hash_instance = xxhash.xxh32()
    elif size == 64:
        hash_instance = xxhash.xxh64()
    else:
        raise ValueError("only 32 or 64-bit hashes supported.")

    if include_selected:
        hash_instance.update(g.effort_levels)
    else:
        hash_instance.update(np.zeros(g.num_nodes))

    hash_instance.update(g.edge_pairs)
    graph_hash = hash_instance.intdigest()
    return graph_hash























































































