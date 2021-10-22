import sys
from pathlib import Path

sys.path.append('/relnet')
from relnet.agent.mcts.mcts_agent import MonteCarloTreeSearchAgent
from relnet.agent.baseline.simulated_annealing import SimulatedAnnealingAgent
from relnet.evaluation.experiment_conditions import get_default_gen_params, get_default_file_paths, get_default_options
from relnet.environment.graph_mis_env import GraphMISEnv
from relnet.objective_functions.objective_functions import SocialWelfare, Fairness
from relnet.state.network_generators import NetworkGenerator, BANetworkGenerator, WattsStrogatzNetworkGenerator, \
    GNMNetworkGenerator

if __name__ == '__main__':
    num_test_graphs = 1
    n = 75

    gen_params = get_default_gen_params(n)
    file_paths = get_default_file_paths()
    options = get_default_options(file_paths)
    #
    # options['draw_trees'] = True
    # options['drawing_type'] = 'mpl'
    # options['tree_illustration_path'] = '/relnet/tmp'

    agent_class = MonteCarloTreeSearchAgent
    hyperparams = agent_class.get_default_hyperparameters()
    hyperparams['C_p'] = 0.1
    #hyperparams['expansion_budget_modifier'] = 1

    # agent_class = ExhaustiveSearchAgent
    # hyperparams = {}

    # agent_class = PayoffTransferAgent
    # hyperparams = {}


    storage_root = Path('/experiment_data/stored_graphs')
    kwargs = {'store_graphs': True, 'graph_storage_root': storage_root}

    #gen = BANetworkGenerator(**kwargs)
    #gen = GNMNetworkGenerator(**kwargs)
    gen = WattsStrogatzNetworkGenerator(**kwargs)
    # graph_seeds = NetworkGenerator.construct_network_seeds(0, 0, num_test_graphs)
    # _, _, test_graph_seeds = graph_seeds

    test_graph_seeds = [126]

    test_graphs = gen.generate_many(gen_params, test_graph_seeds)

    env = GraphMISEnv(SocialWelfare(), {}, heterogenous_cost=True)
    #env = GraphMISEnv(SocialWelfare(), {}, heterogenous_cost=False)
    #env = GraphMISEnv(Fairness(), {})
    env.assign_env_specific_properties(test_graphs)

    agent = agent_class(env)
    agent.setup(options, hyperparams)
    avg_perf = agent.eval(test_graphs)
    print(f"final eval performance was {avg_perf}")