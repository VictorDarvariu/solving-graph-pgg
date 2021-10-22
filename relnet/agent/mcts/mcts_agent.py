import math
import warnings
from copy import copy
from math import sqrt, log
from pathlib import Path
from random import Random
import gc
import matplotlib.pyplot as plt
import network2tikz

import networkx as nx
from billiard.pool import Pool
from networkx.drawing.nx_agraph import graphviz_layout
from relnet.environment.graph_mis_env import GraphMISEnv

from relnet.agent.base_agent import Agent
from relnet.agent.mcts.mcts_tree_node import MCTSTreeNode
from relnet.agent.mcts.simulation_policies import RandomSimulationPolicy
from relnet.evaluation.eval_utils import *
from relnet.state.graph_state import get_graph_hash
from relnet.state.network_generators import NetworkGenerator

class MonteCarloTreeSearchAgent(Agent):
    algorithm_name = 'uct'

    is_deterministic = False
    is_trainable = False

    def __init__(self, environment):
        super().__init__(environment)
        self.draw_trees = False
        self.root_nodes = None

        self.final_action_strategies = {'max_child': self.pick_max_child,
                                        'robust_child': self.pick_robust_child,
                                        }

    def init_root_information(self, t):
        self.root_nodes = []
        self.node_expansion_budgets = []

        for i in range(len(self.environment.g_list)):
            start_state = self.environment.g_list[i]

            root_node = self.initialize_tree_node(None, start_state, None, with_depth=t)
            exp_budget = int(start_state.num_nodes * self.expansion_budget_modifier)

            self.root_nodes.append(root_node)
            self.node_expansion_budgets.append(exp_budget)


    def eval(self, g_list,
             initial_obj_values=None,
             validation=False,
             make_action_kwargs=None):

        orig_env = self.environment
        env_ref = GraphMISEnv.from_env_instance(self.environment)

        trajectories = []
        for i, g in enumerate(g_list):
            starting_graph = g.copy()
            starting_graph_initial_obj_value = 0.

            env_ref.setup([starting_graph],
                                    [starting_graph_initial_obj_value],
                                    training=False)

            self.environment = env_ref
            trajectory = self.run_trajectory_collection()
            trajectories.append(trajectory)

        self.environment = orig_env
        rewards = [t[1] for t in trajectories]
        return float(np.mean(rewards))

    def run_search_for_g_list(self, t, force_init=False):
        if t == 0:
            self.moves_so_far = []
            self.C_ps = []
            self.dummy_moves_reached = []
            self.dummy_Qs = []

            for i in range(len(self.environment.g_list)):
                g = self.environment.g_list[i]
                self.moves_so_far.append([])
                self.C_ps.append(self.hyperparams['C_p'])

                self.dummy_moves_reached.append(False)
                self.dummy_Qs.append(-1.)

            if self.hyperparams['adjust_C_p']:
                self.init_root_information(t)
                for i in range(len(self.root_nodes)):
                    self.execute_search_step(i, t)

        if t == 0 or force_init:
            self.init_root_information(t)
        for i in range(len(self.root_nodes)):
            self.execute_search_step(i, t)

    def execute_search_step(self, i, t):
        #print(f"executing steppe {t}")
        root_node = self.root_nodes[i]
        node_expansion_budget = self.node_expansion_budgets[i]

        if self.dummy_moves_reached[i]:
            hit_terminal_depth = True

            dummy_N = math.floor(float(node_expansion_budget) / float(len(root_node.valid_actions)))
            for valid_act in root_node.valid_actions:
                next_state = self.environment.apply_action(root_node.state, valid_act, in_place=False)
                next_node = self.initialize_tree_node(root_node, next_state, valid_act)
                next_node.Q = self.dummy_Qs[i]
                next_node.N = dummy_N
                root_node.children[valid_act] = next_node

            root_node.N = dummy_N * len(root_node.valid_actions)
            root_node.Q = self.dummy_Qs[i]

        else:
            hit_terminal_depth = False
            while True:
                # follow tree policy to reach a certain node
                tree_nodes, tree_actions = self.follow_tree_policy(root_node, i)
                if len(tree_actions) == 0:
                    hit_terminal_depth = True

                v_l = tree_nodes[-1]
                simulation_results = self.execute_simulation_policy(v_l, root_node, i)
                self.backup_values(tree_nodes, tree_actions, simulation_results)

                if root_node.N >= node_expansion_budget:
                    root_Q = root_node.Q
                    if self.hyperparams['adjust_C_p']:
                        self.C_ps[i] = self.hyperparams['C_p'] * root_Q
                    break
            # print(f"picked action {action} using strategy {self.final_action_strategy}.")
            # print(f"prior to executing, we had {root_node.state.moves_budget} moves left.")

            self.dummy_moves_reached[i], self.dummy_Qs[i] = self.check_dummy_reached(root_node)

        if self.draw_trees:
            if hit_terminal_depth:
                self.draw_search_tree_with_values(i, root_node, t, max_breadth=20, max_depth=1, drawing_type=self.drawing_type)
            else:
                self.draw_search_tree_with_values(i, root_node, t, max_breadth=20, drawing_type=self.drawing_type)

    def check_dummy_reached(self, root_node):
        BW_TOL = 1e-6
        child_vals = np.array([node.Q for node in root_node.children.values()], dtype=np.float64)
        bw_diff = np.max(child_vals) - np.min(child_vals)

        # print(f"child vals are {child_vals}")
        # print(f"bw_diff is {bw_diff}!")

        vals_almost_equal = (bw_diff <= BW_TOL)
        if vals_almost_equal:
            Q = child_vals[0]
        else:
            Q = -1.
        return vals_almost_equal, Q

    def pick_children(self):
        actions = []
        for i in range(len(self.root_nodes)):
            root_node = self.root_nodes[i]
            if len(root_node.children) > 0:
                action, selected_child = self.final_action_strategy(root_node)
                self.root_nodes[i] = selected_child
                self.root_nodes[i].parent_node = None
                del root_node
                gc.collect()
            else:
                self.environment.mark_exhausted(i)
                action = -1

            self.moves_so_far[i].append(action)
            actions.append(action)
        return actions

    def pick_robust_child(self, root_node):
        # print(f"there are {len(root_node.children)} possible actions")
        # print(f"there have been {root_node.state.number_actions_taken} actions so far.")
        # print(f"move budget is {root_node.state.moves_budget}.")
        return sorted(root_node.children.items(), key=lambda x: (x[1].N, x[1].Q), reverse=True)[0]


    def pick_max_child(self, root_node):
        return sorted(root_node.children.items(), key=lambda x: (x[1].Q, x[1].N), reverse=True)[0]

    def follow_tree_policy(self, node, i):
        traversed_nodes = []
        actions_taken = []

        if node.num_valid_actions == 0:
            traversed_nodes.append(node)
            return traversed_nodes, actions_taken

        while True:
            traversed_nodes.append(node)
            state = node.state

            if len(node.children) < node.num_valid_actions:
                if hasattr(self, 'step'):
                    global_step = self.step
                else:
                    global_step = 1

                chosen_action = node.choose_action(int(self.random_seed * node.N * global_step))
                next_state = self.environment.apply_action(state, chosen_action, in_place=False)

                next_node = self.initialize_tree_node(node, next_state, chosen_action)

                node.children[chosen_action] = next_node
                actions_taken.append(chosen_action)
                traversed_nodes.append(next_node)

                break
            else:
                if node.num_valid_actions == 0:
                    break
                else:
                    highest_ucb, ucb_action, ucb_node = self.pick_best_child(node, i, self.C_ps[i])
                    node = ucb_node
                    actions_taken.append(ucb_action)
                    continue

        return traversed_nodes, actions_taken

    def initialize_tree_node(self, parent_node, node_state, chosen_action, with_depth=-1):
        next_node_actions = self.environment.get_valid_actions(node_state)
        next_node_actions = list(next_node_actions)

        depth = parent_node.depth + 1 if with_depth == -1 else with_depth
        next_node = MCTSTreeNode(node_state, parent_node, chosen_action, next_node_actions, depth=depth)
        return next_node


    def pick_best_child(self, node, i, c, print_vals=False):
        highest_value = float("-inf")
        best_node = None
        best_action = None

        child_values = {action: self.node_selection_strategy(node, i, action, child_node)
                        for action, child_node in node.children.items()}


        if print_vals:
            print(f"child values were {child_values}")

        for action, value in child_values.items():
            if value > highest_value:
                highest_value = value
                best_node = node.children[action]
                best_action = action
        return highest_value, best_action, best_node

    def node_selection_strategy(self, parent_node, i, action, child_node):
        # based on UCB1 for default UCT.
        Q, parent_N, child_N = child_node.Q, parent_node.N, child_node.N
        node_value = self.get_ucb1_term(Q, parent_N, child_N, self.C_ps[i])

        if math.isnan(node_value):
            node_value = 0.0

        return node_value

    def get_ucb1_term(self, Q, parent_N, child_N, C_p):
        ci_term = sqrt((2 * log(parent_N)) / child_N)
        ucb1_value = Q + C_p * ci_term
        # print(f"UCB1 value is {ucb1_value}")
        return ucb1_value

    def execute_simulation_policy(self, node, root_node, i):
        if self.sim_policy == 'random':
            sim_policy_class = RandomSimulationPolicy
            sim_policy_kwargs = {}
        else:
            raise ValueError(f"sim policy {self.sim_policy} not recognised!")

        obj_fun_computation = self.environment.objective_function.compute
        obj_fun_kwargs = self.environment.objective_function_kwargs
        if node.num_valid_actions == 0:
            return [([], self.get_final_node_val(node.state, obj_fun_computation, obj_fun_kwargs), None, 0)]

        valid_actions_finder = self.environment.get_valid_actions

        action_applier = self.environment.apply_action
        simulation_results = []
        for sim_number in range(self.num_simulations):
            # print(f"yeah, doing a sim!")
            out_of_tree_acts, R, post_random_state, sim_number = self.sim_policy_episode(sim_policy_class,
                                                                                                sim_policy_kwargs,
                                                                                                node,
                                                                                                root_node,
                                                                                                sim_number,
                                                                                                valid_actions_finder,
                                                                                                action_applier,
                                                                                                obj_fun_computation,
                                                                                                obj_fun_kwargs,
                                                                                                self.local_random.getstate(),
                                                                                                )
            simulation_results.append((out_of_tree_acts, R, post_random_state, sim_number))
            self.local_random.setstate(post_random_state)
        return simulation_results

    @staticmethod
    def sim_policy_episode(sim_policy_class,
                                  sim_policy_kwargs,
                                  node,
                                  root_node,
                                  sim_number,
                                  valid_actions_finder,
                                  action_applier,
                                  obj_fun_computation,
                                  obj_fun_kwargs,
                                  random_state,
                                  ):

        """
        when do we want to stop?
            A: when total used budget from _current root_ exceeds the rollout limit
            alternatively, since env needs remaining budget:
                - subtract from root remaining budget the budget of the current node (that's how much was used up
                from root to here)
                - then, subtract from rollout limit this value: that's your remaining budget
                - take the minimum between this value and rem budget at starting node; this will be hit when search
                is nearly completed
        """

        state = node.state.copy()
        out_of_tree_actions = []
        sim_policy = sim_policy_class(random_state, **sim_policy_kwargs)

        while True:
            possible_actions = valid_actions_finder(state)
            if len(possible_actions) == 0:
                break
                # raise ValueError("Shouldn't run simulation from node when no actions are available!")

            chosen_action = sim_policy.choose_action(state, possible_actions)
            out_of_tree_actions.append(chosen_action)
            action_applier(state, chosen_action, in_place=True)
            # print(f"yay, adding an action from simulation policy! budget updated from {orig_budget} to {rem_budget}")

        final_state = state
        node_val = MonteCarloTreeSearchAgent.get_final_node_val(final_state, obj_fun_computation,
                                                                obj_fun_kwargs)
        post_random_state = sim_policy.get_random_state()
        return out_of_tree_actions, node_val, post_random_state, sim_number


    @staticmethod
    def get_final_node_val(final_state, obj_fun_computation, obj_fun_kwargs):
        final_value = obj_fun_computation(final_state, **obj_fun_kwargs)
        node_val = final_value
        return node_val


    def backup_values(self, tree_nodes, tree_actions, simulation_results):
        for tree_node in tree_nodes:
            for _, R, _, _ in simulation_results:
                tree_node.update_estimates(R)

    def run_trajectory_collection(self):
        acts = []

        t = 0
        while not self.environment.is_terminal():
            # if self.log_progress:
            #     self.logger.info(f"executing inner search step {t}")
            #     self.logger.info(f"{get_memory_usage_str()}")

            self.run_search_for_g_list(t, force_init=True)
            list_at = self.pick_children()
            acts.append(list_at[0])
            self.environment.step(list_at)
            t += 1


        best_acts = self.moves_so_far[0]
        best_F = self.environment.get_final_values()[0]

        if math.isnan(best_F):
            best_F = 0.
        return best_acts, best_F

    def setup(self, options, hyperparams):
        super().setup(options, hyperparams)
        if "draw_trees" in options:
            self.draw_trees = options['draw_trees']
            if self.draw_trees:
                self.tree_illustration_path = Path(options['tree_illustration_path'])
                self.tree_illustration_path.mkdir(parents=True, exist_ok=True)
                self.drawing_type = options['drawing_type']
        else:
            self.draw_trees = False

        if 'num_simulations' in options:
            self.num_simulations = options['num_simulations']
        else:
            self.num_simulations = 1

        self.C_p = hyperparams['C_p']
        self.expansion_budget_modifier = hyperparams['expansion_budget_modifier']

        if 'sim_policy' in hyperparams:
            self.sim_policy = hyperparams['sim_policy']
        else:
            self.sim_policy = 'random'

        if 'final_action_strategy' in hyperparams:
            self.final_action_strategy = self.final_action_strategies[hyperparams['final_action_strategy']]
        else:
            self.final_action_strategy = self.pick_robust_child

    @classmethod
    def get_default_hyperparameters(cls):
        default_hyperparams = {
            'C_p': 0.1,
            'adjust_C_p': True,
            'final_action_strategy': 'max_child',
            'expansion_budget_modifier': 20,
            'sim_policy': 'random',
        }
        return default_hyperparams

    def draw_search_tree_with_values(self, graph_index, root_node, t, max_depth=1, max_breadth=20, drawing_type="mpl", write_dot=False):
        with warnings.catch_warnings():

            G = nx.DiGraph()
            root_node_label = self.construct_node_label(root_node, drawing_type)
            G.add_node(root_node_label, value=root_node.Q)
            self.add_mcts_edges_recursively(G, root_node, root_node_label, drawing_type, depth=0, max_depth=max_depth,
                                            max_breadth=max_breadth)


            if drawing_type == "mpl":
                pos = graphviz_layout(G, prog='dot')
                dot_file = self.tree_illustration_path / f"mcts_tree_g{graph_index}_{self.random_seed}_{t}.dot"
                dot_filename = str(dot_file)
                png_filename = str(self.tree_illustration_path / f"mcts_tree_g{graph_index}_{self.random_seed}_{t}.png")

                plt.figure(figsize=(20, 10))
                plt.title('Monte Carlo Search Tree')

                node_data = list(G.nodes())
                node_colours = [float(x.split("\n")[0].split(":")[1]) for x in node_data]

                labels = {}
                for u, v, data in G.edges(data=True):
                    labels[(u, v)] = data['action']

                nx.draw_networkx(G, pos, with_labels=True, arrows=True, node_size=1500, font_size=8, alpha=0.95, vmin=0,
                                 vmax=1,
                                 cmap="Reds", node_color=node_colours)
                nx.draw_networkx_edges(G, pos, edge_color='g', width=5)



                nx.draw_networkx_edge_labels(G, pos, font_size=8, edge_labels=labels)
                plt.axis('off')
                plt.savefig(png_filename, bbox_inches="tight")
                plt.close()

                # subprocess.Popen(["dot", "-Tpng", dot_file, "-o", png_file])

                if write_dot:
                    A = nx.nx_agraph.to_agraph(G)
                    A.write(dot_filename)

            elif drawing_type == "tikz":
                pos = graphviz_layout(G, prog='dot', args='-Gnodesep=7.5 -Granksep=75')

                tikz_filename = str(self.tree_illustration_path / f"tikz_mcts_tree_g{graph_index}_{self.random_seed}_{t}.tex")
                my_style = {}

                lightblue = (171, 215, 230)
                my_style["canvas"] = (12, 2.5)
                my_style["edge_color"] = ['darkgray' for edge in G.edges]
                my_style['node_color'] = [lightblue for n in G.nodes]
                my_style["node_label"] = [n for n in G.nodes()]
                my_style["node_label_size"] = [0 for n in G.nodes()]

                my_style["node_size"] = [0.28 for n in G.nodes()]
                my_style["edge_width"] = [0.85 for e in G.edges()]
                my_style["edge_style"] = ["-" for e in G.edges()]

                my_style["edge_label_size"] = [0 for e in G.edges()]
                network2tikz.plot(G, tikz_filename, **my_style, layout=pos, standalone=False)



    def add_mcts_edges_recursively(self, G, parent_node, parent_node_label, drawing_type, depth, max_depth, max_breadth):
        if max_depth == 0 or depth == max_depth:
            return

        children_items = parent_node.children.items()
        visited_children = [entry for entry in children_items]
        children = sorted(visited_children, key=lambda x: x[1].Q, reverse=True)[0:max_breadth]

        for action, child_node in children:
            child_node_label = self.construct_node_label(child_node, drawing_type)
            G.add_node(child_node_label, value=child_node.Q)
            G.add_edge(parent_node_label, child_node_label, action=f"{action}")
            self.add_mcts_edges_recursively(G, child_node, child_node_label, drawing_type, depth + 1, max_depth, max_breadth)

    def construct_node_label(self, node, drawing_type):
        state_id = get_graph_hash(node.state, size=64)
        if drawing_type == "mpl":


            node_label = f"Q: {node.Q: .3f}\n" \
                         f"N: {node.N}\n" \
                         f"A: {node.parent_action}\n" \
                         f"id: {state_id}"
            return node_label
        else:
            return state_id

    def finalize(self):
        self.root_nodes = None
        self.node_expansion_budgets = None

    def make_actions(self, t, **kwargs):
        raise ValueError("should not call make_actions directly, but use overriden eval method!")

    def post_env_setup(self):
        pass
