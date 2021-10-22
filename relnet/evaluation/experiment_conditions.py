import os
from copy import deepcopy

from relnet.agent.mcts.il.imitation_learning import ImitationLearningAgent
from relnet.agent.mcts.mcts_agent import MonteCarloTreeSearchAgent

from relnet.agent.baseline.best_response_agent import BestResponseAgent, PayoffTransferAgent

from relnet.agent.baseline.baseline_agent import *
from relnet.agent.baseline.simulated_annealing import SimulatedAnnealingAgent
from relnet.agent.pytorch_agent import PyTorchAgent
from relnet.evaluation.file_paths import FilePaths
from relnet.objective_functions.objective_functions import *
from relnet.state.network_generators import NetworkGenerator, GNMNetworkGenerator, BANetworkGenerator, \
    WattsStrogatzNetworkGenerator, RegularNetworkGenerator


class ExperimentConditions(object):
    def __init__(self, base_n, train_individually, heterogenous_cost):
        self.gen_params = {}

        #self.all_ns = [15, 25]
        self.all_ns = [15, 25, 50, 75, 100]

        self.base_n = base_n
        self.train_individually = train_individually
        self.heterogenous_cost = heterogenous_cost

        self.gen_params = get_default_gen_params(self.base_n)

        self.size_multipliers = [1]
        self.exhaustive_search_threshold = 15

        self.objective_functions = [
            SocialWelfare,
            Fairness
        ]

        self.network_generators = [
            GNMNetworkGenerator,
            BANetworkGenerator,
            WattsStrogatzNetworkGenerator,
        ]

        self.experiment_params = {'train_graphs': 1000,
                                  'validation_graphs': 100,
                                  'test_graphs': 100,
                                  'num_runs': 10}

        self.experiment_params['model_seeds'] = [self.get_model_seed(run_num) for run_num in
                                                 range(self.experiment_params['num_runs'])]

        self.model_seeds_to_skip = {
            # Can be used to skip some random seeds in case training failed.
        }

    def get_model_seed(self, run_number):
        return run_number * 42

    def get_run_number(self, model_seed):
        return model_seed / 42

    def update_size_dependant_params(self, multiplier):
        self.gen_params['n'] = int(self.base_n * multiplier)
        self.gen_params['m'] = NetworkGenerator.compute_number_edges(self.gen_params['n'], self.gen_params['m_percentage_er'])
        self.gen_params['size_multiplier'] = multiplier

    def set_generator_seeds(self):
        self.train_seeds, self.validation_seeds, self.test_seeds = NetworkGenerator.construct_network_seeds(
            self.experiment_params['train_graphs'],
            self.experiment_params['validation_graphs'],
            self.experiment_params['test_graphs'])

    def set_generator_seeds_individually(self, g_num, num_graphs):
        self.validation_seeds = [g_num]
        self.test_seeds = [g_num]

        # TODO: not compatible with configurable batch sizes! Need another mechanism to deal with this.
        self.train_seeds = [g_num + (i * num_graphs) for i in range(1, PyTorchAgent.DEFAULT_BATCH_SIZE+1)]


    def update_relevant_agents(self):
        relevant_agents = deepcopy(self.agents_models)
        self.relevant_agents = relevant_agents

    def extend_seeds_to_skip(self, run_num_start, run_num_end):
        for net in self.network_generators:
            for obj in self.objective_functions:
                for agent in self.relevant_agents:
                    setting = (net.name, obj.name, agent.algorithm_name)
                    if setting not in self.model_seeds_to_skip:
                        self.model_seeds_to_skip[setting] = []

                    for run_num_before in range(0, run_num_start):
                        self.model_seeds_to_skip[setting].append(self.get_model_seed(run_num_before))

                    for run_num_after in range(run_num_end + 1, self.experiment_params['num_runs']):
                        self.model_seeds_to_skip[setting].append(self.get_model_seed(run_num_after))

    def __str__(self):
        as_dict = deepcopy(self.__dict__)
        del as_dict["agents_models"]
        del as_dict["agents_baseline"]
        del as_dict["objective_functions"]
        del as_dict["network_generators"]
        return str(as_dict)

    def __repr__(self):
        return self.__str__()


class MainExperimentConditions(ExperimentConditions):
    def __init__(self, base_n, train_individually, heterogenous_cost):
        super().__init__(base_n, train_individually, heterogenous_cost)

        self.agents_models = [
            MonteCarloTreeSearchAgent,
        ]

        self.agents_baseline = {
            SocialWelfare.name: [
                RandomAgent,
                TargetHubsAgent,
                TargetMinCostAgent,
                ExhaustiveSearchAgent,
                BestResponseAgent,
                PayoffTransferAgent,
                SimulatedAnnealingAgent
            ],
            Fairness.name: [
                RandomAgent,
                TargetHubsAgent,
                TargetMinCostAgent,
                ExhaustiveSearchAgent,
                BestResponseAgent,
                PayoffTransferAgent,
                SimulatedAnnealingAgent
            ]
        }

        self.agent_budgets = {
            SocialWelfare.name: {
            },
            Fairness.name: {
            },
        }

        # Can be used to skip some parameter combinations by their int id
        self.parameter_combs_to_skip = {
            SocialWelfare.name: {
                MonteCarloTreeSearchAgent.algorithm_name: []
            },

            Fairness.name: {
                MonteCarloTreeSearchAgent.algorithm_name: []
            },
        }


        self.hyperparam_grids = self.create_hyperparam_grids()


    def create_hyperparam_grids(self):
        hyperparam_grid_base =  {
            MonteCarloTreeSearchAgent.algorithm_name: {
                "C_p": [0.05, 0.1, 0.25, 0.5, 1, 2.5],
                "adjust_C_p": [True],
                "expansion_budget_modifier": [20]
            },
        }
        hyperparam_grids = {}
        for f in self.objective_functions:
            hyperparam_grids[f.name] = deepcopy(hyperparam_grid_base)

        return hyperparam_grids


class ILCollectExperimentConditions(ExperimentConditions):
    def __init__(self, base_n, train_individually, heterogenous_cost):
        super().__init__(base_n, train_individually, heterogenous_cost)

        self.agents_models = [
            ImitationLearningAgent,
        ]

        self.agents_baseline = {
            SocialWelfare.name: [
            ],
            Fairness.name: [
            ]
        }

        self.agent_budgets = {
            SocialWelfare.name: {
                ImitationLearningAgent.algorithm_name: -1
            },
            Fairness.name: {
                ImitationLearningAgent.algorithm_name: -1
            },
        }

        # Can be used to skip some parameter combinations by their int id
        self.parameter_combs_to_skip = {
            SocialWelfare.name: {
                ImitationLearningAgent.algorithm_name: []
            },

            Fairness.name: {
                ImitationLearningAgent.algorithm_name: []
            },
        }

        self.hyperparam_grids = self.create_hyperparam_grids()


    def create_hyperparam_grids(self):
        hyperparam_grid_base = {
            ImitationLearningAgent.algorithm_name: {
                "dummy": [-1],
            },
        }
        hyperparam_grids = {}
        for f in self.objective_functions:
            hyperparam_grids[f.name] = deepcopy(hyperparam_grid_base)

        return hyperparam_grids


class ILImitateExperimentConditions(ExperimentConditions):
    def __init__(self, base_n, train_individually, heterogenous_cost):
        super().__init__(base_n, train_individually, heterogenous_cost)

        self.agents_models = [
            ImitationLearningAgent
        ]

        self.agents_baseline = {
            SocialWelfare.name: [
            ],
            Fairness.name: [
            ]
        }

        num_train_steps = 200

        self.agent_budgets = {
            SocialWelfare.name: {
                ImitationLearningAgent.algorithm_name: num_train_steps,
            },
            Fairness.name: {
                ImitationLearningAgent.algorithm_name: num_train_steps,
            },
        }

        # Can be used to skip some parameter combinations by their int id
        self.parameter_combs_to_skip = {
            SocialWelfare.name: {
                ImitationLearningAgent.algorithm_name: [],

            },

            Fairness.name: {
                ImitationLearningAgent.algorithm_name: [],
            },
        }

        self.hyperparam_grids = self.create_hyperparam_grids()


    def create_hyperparam_grids(self):
        hyperparam_grid_base = {
            ImitationLearningAgent.algorithm_name: {
                "learning_rate": [0.01, 0.001, 0.0001],
                "max_lv": [3, 4, 5],
                "latent_dim": [64],
                "embedding_method": ["mean_field"],
                "distance_metric": ["euclidean"],
                "softmax_temp": [10.],
                "batch_size": [5],
                "il_procedure": ["separate", "mixed", "curriculum"]
            },
        }
        hyperparam_grids = {}
        for f in self.objective_functions:
            hyperparam_grids[f.name] = deepcopy(hyperparam_grid_base)

        return hyperparam_grids

def get_conditions_for_experiment(which, base_n, train_individually, heterogenous_cost):
    if which == 'main':
        cond = MainExperimentConditions(base_n, train_individually, heterogenous_cost)
    elif which == 'il_collect':
        cond = ILCollectExperimentConditions(base_n, train_individually, heterogenous_cost)
    elif which == 'il_imitate':
        cond = ILImitateExperimentConditions(base_n, train_individually, heterogenous_cost)
    else:
        raise ValueError(f"experiment {which} not recognized!")
    return cond

def get_default_gen_params(n):
    gp = {}
    gp['n'] = n
    gp['m_ba'] = 1
    gp['m_percentage_er'] = 20
    gp['m_ws'] = 2
    gp['p_ws'] = 0.1
    gp['d_reg'] = 2
    gp['m'] = NetworkGenerator.compute_number_edges(gp['n'], gp['m_percentage_er'])

    return gp

def get_default_options(file_paths):
    options = {"log_progress": True,
               "log_filename": str(file_paths.construct_log_filepath()),
               "log_tf_summaries": True,
               "random_seed": 42,
               "models_path": file_paths.models_dir,
               "restore_model": False}
    return options

def get_default_file_paths(experiment_id = 'development'):
    parent_dir = '/experiment_data'
    file_paths = FilePaths(parent_dir, experiment_id)
    return file_paths
