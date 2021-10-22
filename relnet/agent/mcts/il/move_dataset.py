import math
import pickle
from collections import namedtuple
import numpy as np

move_data = ['state', 'actions', 'graph_size', 'visit_counts', 'is_dummy', 'timestep']
SearchMove = namedtuple('SearchMove', move_data)

class SubStepMoveDataset(object):
    def __init__(self, dataset_size):
        self.dataset_size = dataset_size

        self.states = [None] * self.dataset_size
        self.actions = [None] * self.dataset_size

        self.graph_sizes = [None] * self.dataset_size
        self.visit_counts = [None] * self.dataset_size
        self.is_dummy = [None] * self.dataset_size
        self.timesteps = [None] * self.dataset_size

        self.count = 0
        self.current = 0

    def __len__(self):
        return self.count

    def add(self, search_move):
        self.states[self.current] = search_move.state
        self.actions[self.current] = search_move.actions

        self.graph_sizes[self.current] = search_move.graph_size
        self.visit_counts[self.current] = search_move.visit_counts
        self.is_dummy[self.current] = search_move.is_dummy
        self.timesteps[self.current] = search_move.timestep

        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.dataset_size

    def add_list(self, search_moves):
        for search_move in search_moves:
            self.add(search_move)

    def sample_with_indices(self, idx):
        #print(f"count is {self.count}, indices of length {len(idx)}")
        assert self.count >= len(idx)

        list_states = []
        list_actions = []

        list_graph_sizes = []
        list_visit_counts = []
        list_is_dummy = []
        list_timesteps = []

        for selected_index in idx:
            list_states.append(self.states[selected_index])
            list_actions.append(self.actions[selected_index])

            list_graph_sizes.append(self.graph_sizes[selected_index])
            list_visit_counts.append(self.visit_counts[selected_index])
            list_is_dummy.append(self.is_dummy[selected_index])
            list_timesteps.append(self.timesteps[selected_index])

        return list_states, list_actions, list_graph_sizes, list_visit_counts, list_is_dummy, list_timesteps

    def sample_with_indices_zip(self, idx):
        list_states, list_actions, list_graph_sizes, list_visit_counts, list_is_dummy, list_timesteps = self.sample_with_indices(idx)
        zipped_moves = zip(list_states, list_actions, list_graph_sizes, list_visit_counts, list_is_dummy, list_timesteps)
        return [SearchMove(*move) for move in zipped_moves]

    def sample_most_recent(self, num_samples):
        assert self.count >= num_samples

        list_states = self.states[self.count-num_samples:]
        list_actions = self.actions[self.count - num_samples:]

        list_graph_sizes = self.graph_sizes[self.count-num_samples:]
        list_visit_counts = self.visit_counts[self.count-num_samples:]
        list_is_dummy = self.is_dummy[self.count-num_samples:]
        list_timesteps = self.timesteps[self.count-num_samples:]

        return list_states, list_actions, list_graph_sizes, list_visit_counts, list_is_dummy, list_timesteps

    def sample_most_recent_zip(self, num_samples):
        list_states, list_actions, list_graph_sizes, list_visit_counts, list_is_dummy, list_timesteps = self.sample_most_recent(num_samples)
        zipped_moves = zip(list_states, list_actions, list_graph_sizes, list_visit_counts, list_is_dummy, list_timesteps)
        return [SearchMove(*move) for move in zipped_moves]

class MoveDataset(object):

    def __init__(self, dataset_size, num_mdp_substeps, storage_root=None, prefix=None):
        self.dataset_size = dataset_size
        self.num_mdp_substeps = num_mdp_substeps

        self.max_subdataset_size = math.floor(dataset_size / num_mdp_substeps)
        self.storage_root = storage_root
        self.prefix = prefix

        self.sub_step_datasets = []
        for _ in range(self.num_mdp_substeps):
            self.sub_step_datasets.append(SubStepMoveDataset(self.max_subdataset_size))

    def __len__(self):
        total_count = 0
        for sub_step_ds in self.sub_step_datasets:
            total_count += len(sub_step_ds)
        return total_count

    def add(self, search_move, t):
        self.sub_step_datasets[t].add(search_move)

    def add_list(self, search_moves, t):
        self.sub_step_datasets[t].add_list(search_moves)

    def sample_with_indices(self, idx, t):
        return self.sub_step_datasets[t].sample_with_indices(idx)

    def sample_with_indices_zip(self, idx, t):
        return self.sub_step_datasets[t].sample_with_indices_zip(idx)

    def sample_most_recent(self, num_samples, t):
        return self.sub_step_datasets[t].sample_most_recent(num_samples)

    def sample_most_recent_zip(self, num_samples, t):
        return self.sub_step_datasets[t].sample_most_recent_zip(num_samples)

    def get_min_subdataset_size(self):
        return min([len(d) for d in self.sub_step_datasets])

    def save(self):
        if self.storage_root is None: raise FileNotFoundError("Storage not configured!")
        for i in range(self.num_mdp_substeps):
            pickle_out = open(self.get_subdataset_path(i), "wb")
            pickle.dump(self.sub_step_datasets[i], pickle_out)
            pickle_out.close()

    def load(self):
        if self.storage_root is None: raise FileNotFoundError("Storage not configured!")
        for i in range(self.num_mdp_substeps):
            pickle_in = open(self.get_subdataset_path(i), "rb")
            self.sub_step_datasets[i] = pickle.load(pickle_in)
            pickle_in.close()

    def get_subdataset_path(self, i):
        filename = f"{self.prefix}-{i}-move_data.pickle"
        return str((self.storage_root / filename).absolute())



