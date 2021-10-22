import sys
sys.path.append('/relnet')

from copy import copy
from pathlib import Path

import numpy as np

from relnet.agent.mcts.il.imitation_learning import ImitationLearningAgent
from relnet.agent.baseline.baseline_agent import RandomAgent, TargetHubsAgent, TargetMinCostAgent
from relnet.environment.graph_mis_env import GraphMISEnv

from relnet.evaluation.experiment_conditions import get_default_gen_params
from relnet.objective_functions.objective_functions import SocialWelfare

from relnet.evaluation.file_paths import FilePaths
from relnet.state.network_generators import NetworkGenerator, BANetworkGenerator, WattsStrogatzNetworkGenerator, \
    GNMNetworkGenerator

use_dev = True

def get_options(file_paths, obj_fun, gen, agent, experiment_id):
    mcts_opts = {}
    mcts_opts['random_seed'] = 42

    mcts_opts['draw_trees'] = False

    # mcts_opts['draw_trees'] = True
    # mcts_opts['tree_illustration_path'] = '/relnet/tmp'
    # mcts_opts['drawing_type'] = 'mpl'

    mcts_opts['log_progress'] = True

    mcts_opts['log_filename'] = file_paths.construct_log_filepath()
    mcts_opts['models_path'] = file_paths.models_dir
    mcts_opts['model_identifier_prefix'] = file_paths.construct_model_identifier_prefix(agent.algorithm_name,
                                                                                        obj_fun.name,
                                                                                        gen.name,
                                                                                       0,
                                                                                       5,
                                                                                       graph_id=None)

    mcts_opts['log_tf_summaries'] = True

    train_ns = [15, 25]#, 50]
    target_n = 25 # 50

    move_dataset_storage_roots = {}
    for train_n in train_ns:
        collect_id = experiment_id.replace("ili", "ilc").replace(str(target_n), str(
            train_n))  # simple naming convention to know where to find move data
        fp_collect = FilePaths(file_paths.parent_dir, collect_id, setup_directories=False).move_data_dir
        move_dataset_storage_roots[train_n] = fp_collect

    mcts_opts['move_dataset_storage_roots'] = move_dataset_storage_roots
    mcts_opts['target_n'] = target_n

    mcts_opts['move_dataset_storage_root'] = file_paths.move_data_dir
    mcts_opts['move_dataset_prefix'] = file_paths.construct_move_dataset_prefix(agent.algorithm_name,
                                                                                        obj_fun.name,
                                                                                        gen.name,
                                                                                       0,
                                                                                       graph_id=None)

    mcts_opts['validation_check_interval'] = 5
    mcts_opts['worker_pool_size'] = 1

    return mcts_opts


def get_file_paths(exp_id):
    parent_dir = '/experiment_data'
    file_paths = FilePaths(parent_dir, exp_id, setup_directories=True)
    return file_paths


def print_baseline_perf(eval_graphs):
    randy = RandomAgent(env)
    randy.setup({}, options)
    rand_perf = randy.eval(eval_graphs)
    print(f"rand agent saw performance on test set of {rand_perf}.")
    hubs = TargetHubsAgent(env)
    hubs.setup({}, options)
    hubs_perf = hubs.eval(eval_graphs)
    print(f"target hubs agent saw performance on test set of {hubs_perf}.")
    mc = TargetMinCostAgent(env)
    mc.setup({}, options)
    mc_perf = mc.eval(eval_graphs)
    print(f"target min cost agent saw performance on test set of {mc_perf}.")


if __name__ == '__main__':
    n = 25

    obj_fun = SocialWelfare()
    is_hc = True
    exp_id_suff = 'hc' if is_hc else 'ic'

    exp_id = f'ggnn_ili_{n}_{exp_id_suff}'
    gp = get_default_gen_params(n)

    file_paths = get_file_paths(exp_id)
    num_train_graphs = 5
    num_validation_graphs = 100
    num_test_graphs = 100

    g_seed = 42
    num_training_steps = 200

    # graph_seeds = [g_seed] * num_train_graphs, [g_seed], [g_seed]
    # train_graph_seeds, validation_graph_seeds, test_graph_seeds = graph_seeds

    train_graph_seeds, validation_graph_seeds, test_graph_seeds = NetworkGenerator.construct_network_seeds(
        num_train_graphs,
        num_validation_graphs,
        num_test_graphs)

    storage_root = Path('/experiment_data/stored_graphs')
    kwargs = {'store_graphs': True, 'graph_storage_root': storage_root}

    #gen = BANetworkGenerator(**kwargs)
    gen = WattsStrogatzNetworkGenerator(**kwargs)
    #gen = GNMNetworkGenerator(**kwargs)

    train_graphs = gen.generate_many(gp, train_graph_seeds)
    validation_graphs = gen.generate_many(gp, validation_graph_seeds)
    eval_graphs = gen.generate_many(gp, test_graph_seeds)

    obj_fun_kwargs = {}

    env = GraphMISEnv(obj_fun, obj_fun_kwargs, heterogenous_cost=is_hc)

    env.assign_env_specific_properties(train_graphs)
    env.assign_env_specific_properties(validation_graphs)
    env.assign_env_specific_properties(eval_graphs)

    agent = ImitationLearningAgent(env)
    options = get_options(file_paths, obj_fun, gen, agent, exp_id)

    hyperparams = agent.get_default_hyperparameters()

    #hyperparams['distance_metric'] = 'cosine'
    hyperparams['distance_metric'] = 'euclidean'

    hyperparams['expansion_budget_modifier'] = 20

    #hyperparams['C_p'] = 0.1
    hyperparams['learning_rate'] = 0.001

    hyperparams['batch_size'] = 5

    opts = copy(options)
    #opts['il_part'] = 'collect'
    opts['il_part'] = 'imitate'

    #options['restore_model'] = True
    agent.setup(opts, hyperparams)
    agent.train(train_graphs, validation_graphs, num_training_steps)

    perf = agent.eval(eval_graphs)
    print(f"N={n}: imitation learned model saw performance on test set of {perf}.")
    agent.finalize()
    print_baseline_perf(eval_graphs)


