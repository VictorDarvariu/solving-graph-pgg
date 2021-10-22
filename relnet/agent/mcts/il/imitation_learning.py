import itertools
import math
from copy import copy, deepcopy

from billiard.pool import Pool
from psutil import cpu_count

import numpy as np
import torch

from relnet.agent.mcts.il.move_dataset import SearchMove, MoveDataset
from relnet.agent.mcts.il.policy_net import PolicyNet
from relnet.environment.graph_mis_env import GraphMISEnv

from torch import optim
import torch.nn.functional as F

from tqdm import tqdm

from relnet.agent.mcts.mcts_agent import MonteCarloTreeSearchAgent
from relnet.agent.pytorch_agent import PyTorchAgent
from relnet.evaluation.eval_utils import get_values_for_g_list, eval_on_dataset
from relnet.state.graph_state import get_graph_hash
from relnet.utils.config_utils import get_device_placement



class ImitationLearningAgent(MonteCarloTreeSearchAgent, PyTorchAgent):
    algorithm_name = MonteCarloTreeSearchAgent.algorithm_name + '_il'

    is_deterministic = False
    is_trainable = True

    def train(self, train_g_list, validation_g_list, max_steps, **kwargs):
        if "ds_start_index" in self.options:
            start_index = self.options["ds_start_index"]
            end_index = self.options["ds_end_index"]

            train_g_list = train_g_list[start_index:end_index]
            print(f"reduced train g list to indices{start_index}:{end_index}")

        self.setup_graphs(train_g_list, validation_g_list)


        if self.il_part == "collect":
            self.dataset_size = self.get_dataset_size(train_g_list)
            self.setup_move_dataset(self.dataset_size, self.move_dataset_storage_root, self.move_dataset_prefix)

            self.collect_search_moves_pool()
            self.move_dataset.save()

        elif self.il_part == "imitate":
            if self.hyperparams['il_procedure'] == 'separate':
                self.load_relevant_data_separate(train_g_list, validation_g_list)
            elif self.hyperparams['il_procedure'] == 'mixed':
                self.load_relevant_data_mixed(train_g_list, validation_g_list)
            elif self.hyperparams['il_procedure'] == 'curriculum':
                self.load_relevant_data_curriculum(train_g_list, validation_g_list)
            else:
                raise ValueError(f"il_procedure must be one of [separate, mixed, curriculum]!")

            self.setup_step_metrics()
            self.setup_histories_file()

            # adjusting number of steps based on received batch size...
            adjusted_max_steps = math.floor((self.DEFAULT_BATCH_SIZE / self.batch_size) * max_steps)
            print(f"batch size {self.batch_size}: adjusting max steps from {max_steps} to {adjusted_max_steps}.")
            max_steps = adjusted_max_steps

            adjusted_check_interval = math.floor((self.DEFAULT_BATCH_SIZE / self.batch_size) * self.validation_check_interval)
            print(f"batch size {self.batch_size}: adjusting check interval from {self.validation_check_interval} to {adjusted_check_interval}.")
            self.validation_check_interval = adjusted_check_interval

            self.setup_training_parameters(max_steps)

            self.setup_policy_net()
            self.save_model_checkpoints()

            self.save_model_checkpoints(model_suffix='latest')
            with torch.no_grad():
                self.check_validation_loss_if_req(self.step, max_steps, save_model_if_better=True)

            optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
            pbar = tqdm(range(1, max_steps + 1), unit='steps', disable=None, desc='imitation learning loop')

            ds_switch_steps = [1 + math.floor(i * (max_steps / len(self.train_datasets))) for i in range(len(self.train_datasets))]
            curr_ds_num = 0

            for self.step in pbar:
                # print(f"executing step {self.step}")
                if self.step in ds_switch_steps:
                    train_ds = self.train_datasets[curr_ds_num]
                    min_ds_size = train_ds.get_min_subdataset_size()
                    self.setup_sample_idxes(min_ds_size)
                    curr_ds_num += 1
                    print(f"switched dataset at step {self.step}!")

                selected_idx = self.advance_pos_and_sample_indices()
                #print(f"working with idxes {selected_idx}")
                net_losses = []

                list_states, list_actions, list_graph_sizes, list_visit_counts, list_is_dummy, list_timesteps = train_ds.sample_with_indices(selected_idx, 0)
                vc_sums = [sum(vc) for vc in list_visit_counts]
                list_probabilities = [[float(list_visit_counts[i][j]) / vc_sums[i] for j in range(len(list_visit_counts[i]))] for i in range(len(list_visit_counts))]

                _, log_probs, _ = self.net(list_states, list_actions)
                targets = torch.tensor(list(itertools.chain(*list_probabilities)), dtype=torch.float32)

                if get_device_placement() == 'GPU':
                    targets = targets.cuda()
                loss = F.kl_div(log_probs, targets)
                net_losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                self.save_model_checkpoints(model_suffix='latest')

                with torch.no_grad():
                    self.check_validation_loss_if_req(self.step, max_steps, save_model_if_better=True)

            self.restore_model_from_checkpoint()

    def setup_policy_net(self):
        self.num_node_feats = self.environment.get_num_node_feats()
        self.num_edge_feats = self.environment.get_num_edge_feats()

        self.net = PolicyNet(self.hyperparams, self.num_node_feats, self.num_edge_feats)

        if self.restore_model:
            print(f"restore model flag true; restoring!")
            self.restore_model_from_checkpoint()

        if get_device_placement() == 'GPU':
            self.net = self.net.cuda()


    def eval(self, g_list,
             initial_obj_values=None,
             validation=False,
             make_action_kwargs=None):

        env_ref = GraphMISEnv.from_env_instance(self.environment)
        step_dep_params = self.dump_step_dependent_params()

        iter_rewards = []
        for i, g in enumerate(g_list):
            iter_gs = [deepcopy(g_list[i])]
            iter_initial = [0.]
            opts_copy = copy(self.options)
            opts_copy['collect_exp_locally'] = True
            opts_copy['log_tf_summaries'] = False

            my_class = self.__class__
            self_cp = my_class(env_ref)

            self_cp.setup(opts_copy, self.hyperparams)
            self_cp.restore_step_dependent_params(self.step, step_dep_params)

            self_cp.setup_policy_net()
            self_cp.restore_model_from_checkpoint(model_suffix='latest')

            with torch.no_grad():
                starting_obj_values, final_obj_values = get_values_for_g_list(self_cp, iter_gs, iter_initial, validation,
                                                                              make_action_kwargs)

                iter_perf = eval_on_dataset(starting_obj_values, final_obj_values)
                iter_rewards.append(iter_perf)

            self_cp.finalize()
            del self_cp

        return np.mean(iter_rewards)

    def collect_search_moves_pool(self):
        num_collected_moves = 0
        parallel_tasks = []
        env_ref = GraphMISEnv.from_env_instance(self.environment)
        step_dep_params = self.dump_step_dependent_params()
        graph_dataset_idx = list(range(0, len(self.train_g_list)))
        self.local_random.shuffle(graph_dataset_idx)

        for graph_idx in graph_dataset_idx:
            starting_graphs, starting_graphs_initial_obj_values = [self.train_g_list[graph_idx]], [self.train_initial_obj_values[graph_idx]]
            opts_copy = copy(self.options)
            opts_copy['collect_exp_locally'] = True
            opts_copy['log_tf_summaries'] = False
            opts_copy['random_seed'] = (graph_idx + 1) * self.random_seed

            parallel_tasks.append((self.__class__,
                                   env_ref,
                                   self.step,
                                   step_dep_params,
                                   self.hyperparams,
                                   opts_copy,
                                   starting_graphs,
                                   starting_graphs_initial_obj_values,
                                   ))

        for local_search_moves in self.worker_pool.starmap(self.get_search_moves_for_graphs, parallel_tasks):
            for move, mdp_substep  in local_search_moves:
                # print(f"move returned was {move}")
                self.move_dataset.add(move, mdp_substep)
                num_collected_moves += 1

        return num_collected_moves


    @staticmethod
    def get_search_moves_for_graphs(agent_class, environment, step, step_dep_params, hyperparams, options, starting_graphs,
                                    starting_graphs_initial_obj_values):

        agent = agent_class(environment)
        agent.setup(options, hyperparams)
        agent.restore_step_dependent_params(step, step_dep_params)
        agent.environment.setup(starting_graphs,
                                starting_graphs_initial_obj_values,
                                training=True)

        search_moves = []
        t = 0
        while not agent.environment.is_terminal():
            # root_graph = agent.environment.g_list[0]
            # if get_graph_hash(root_graph) == 857399136:
            #     print("got it!")
            #     pass

            agent.run_search_for_g_list(t)
            for i in range(len(agent.root_nodes)):
                root_node = agent.root_nodes[i]
                root_N = root_node.N
                state = root_node.state.copy()
                actions = root_node.valid_actions

                graph_size = root_node.state.num_nodes
                visit_counts = [root_node.children[action].N for action in actions]
                is_dummy = agent.dummy_moves_reached[i]
                timestep = t

                search_move = (SearchMove(state, actions, graph_size, visit_counts, is_dummy, timestep), 0)
                search_moves.append(search_move)

            list_at = agent.pick_children()
            environment.step(list_at)
            t += 1
        return search_moves

    def dump_step_dependent_params(self):
        param_names = ["best_validation_changed_steps", "best_validation_losses", "step", "max_steps"]
        agent_fields = vars(self)
        size_dep_params = {f: agent_fields[f] for f in param_names if f in agent_fields}
        return size_dep_params

    def restore_step_dependent_params(self, step, step_params):
        for f, v in step_params.items():
            setattr(self, f, v)

    def make_actions(self, t, **kwargs):
        states = self.environment.g_list
        allowed_actions = []
        for state in states:
            valid_acts = self.environment.get_valid_actions(state)
            allowed_actions.append(list(valid_acts))
        return self.net.get_greedy_actions(states, allowed_actions)

    def get_dataset_size(self, g_list):
        avg_moves_per_iteration = len(g_list) * g_list[0].num_nodes
        dataset_size = int(avg_moves_per_iteration)
        return dataset_size

    def setup_move_dataset(self, dataset_size, move_dataset_storage_root=None, move_dataset_prefix=None):
        self.move_dataset = MoveDataset(dataset_size, self.environment.get_num_mdp_substeps(),
                                        storage_root=move_dataset_storage_root,
                                        prefix = move_dataset_prefix)
        print(f"memory set to track {dataset_size} search moves.")

    def setup_worker_pool(self):
        self.worker_pool = Pool(processes=self.pool_size)

    def setup_training_parameters(self, max_steps):
        self.max_steps = max_steps
        self.learning_rate = self.hyperparams['learning_rate']

    def setup(self, options, hyperparams):
        super().setup(options, hyperparams)

        if 'il_part' in options:
            self.il_part = options['il_part']
        else:
            self.il_part = None

        if 'collect_exp_locally' in options and options['collect_exp_locally']:
            pass
        else:
            if 'worker_pool_size' in options:
                self.pool_size = options['worker_pool_size']
            else:
                self.pool_size = cpu_count(logical=True)
            self.setup_worker_pool()

        if "move_dataset_storage_root" in options:
            self.move_dataset_storage_root = options['move_dataset_storage_root']
        else:
            self.move_dataset_storage_root = None

        if "move_dataset_storage_roots" in options:
            self.move_dataset_storage_roots = options['move_dataset_storage_roots']
        else:
            self.move_dataset_storage_roots = None

        if "target_n" in options:
            self.target_n = options['target_n']
        else:
            self.target_n = None

        if "move_dataset_prefix" in options:
            self.move_dataset_prefix = options['move_dataset_prefix']
        else:
            self.move_dataset_prefix = None

    def get_default_hyperparameters(self):
        mcts_params = super().get_default_hyperparameters()
        il_params = {
            'il_procedure': 'curriculum',
            'learning_rate': 0.001,
            'latent_dim': 64,
            'embedding_method': 'mean_field',
            'max_lv': 5,
            'softmax_temp': 10.,
            'batch_size': self.DEFAULT_BATCH_SIZE
        }

        mcts_params.update(il_params)
        return mcts_params

    def finalize(self):
        try:
            if hasattr(self, 'worker_pool'):
                if self.worker_pool is not None:
                    self.worker_pool.close()
        except BaseException:
            pass

    def read_data_from_root(self, target_ds_root, move_dataset_prefix):
        prefix_root = "-".join(move_dataset_prefix.split("-")[0:3])
        all_data = []
        for data_file in target_ds_root.glob(prefix_root + "*"):
            ds_chunk_prefix = "-".join(str(data_file.name).split("-")[0:4])

            # print(f"got data file {data_file.name}")
            chunk_ds = MoveDataset(0, self.environment.get_num_mdp_substeps(),
                                   storage_root=target_ds_root,
                                   prefix=ds_chunk_prefix)
            chunk_ds.load()

            all_chunk_idxes = range(0, len(chunk_ds))
            chunk_data = chunk_ds.sample_with_indices_zip(all_chunk_idxes, 0)
            all_data.extend(chunk_data)
        return all_data

    def load_relevant_data_separate(self, train_g_list, validation_g_list):
        base_ds_size = self.get_dataset_size(train_g_list)
        full_ds = MoveDataset(base_ds_size, self.environment.get_num_mdp_substeps())
        target_ds_root = self.move_dataset_storage_roots[self.target_n]

        all_data = self.read_data_from_root(target_ds_root, self.move_dataset_prefix)
        full_ds.add_list(all_data, 0)

        self.train_datasets = [full_ds]
        dataset_size = len(full_ds)
        print(f"{self.algorithm_name}:at end, full ds has size {dataset_size}")


    def load_relevant_data_mixed(self, train_g_list, validation_g_list):
        base_ds_size = self.get_dataset_size(train_g_list) * len(self.move_dataset_storage_roots)
        full_ds = MoveDataset(base_ds_size, self.environment.get_num_mdp_substeps())

        for graph_n, ds_root in self.move_dataset_storage_roots.items():
            n_data = self.read_data_from_root(ds_root, self.move_dataset_prefix)
            full_ds.add_list(n_data, 0)
            print(f"{self.algorithm_name}: read {len(n_data)} moves from path {ds_root}")

        self.train_datasets = [full_ds]
        dataset_size = len(full_ds)
        print(f"{self.algorithm_name}: at end, full ds has size {dataset_size}")


    def load_relevant_data_curriculum(self, train_g_list, validation_g_list):
        self.train_datasets = []
        base_ds_size = self.get_dataset_size(train_g_list)

        #print(f"{self.algorithm_name}: changed {self.move_dataset_prefix} to {ds_prefix}.")

        for graph_n, ds_root in self.move_dataset_storage_roots.items():
            n_ds = MoveDataset(base_ds_size, self.environment.get_num_mdp_substeps())

            n_data = self.read_data_from_root(ds_root, self.move_dataset_prefix)
            n_ds.add_list(n_data, 0)
            print(f"{self.algorithm_name}: read {len(n_data)} moves from path {ds_root}")
            self.train_datasets.append(n_ds)











































































