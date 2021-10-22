import sys
from pathlib import Path

from relnet.agent.baseline.baseline_agent import RandomAgent, ExhaustiveSearchAgent, TargetHubsAgent, TargetMinCostAgent
from relnet.agent.baseline.best_response_agent import BestResponseAgent, PayoffTransferAgent

sys.path.append('/relnet')

from relnet.agent.baseline.simulated_annealing import SimulatedAnnealingAgent
from relnet.evaluation.experiment_conditions import get_default_gen_params, get_default_file_paths, get_default_options
from relnet.environment.graph_mis_env import GraphMISEnv
from relnet.objective_functions.objective_functions import SocialWelfare, Fairness
from relnet.state.network_generators import NetworkGenerator, BANetworkGenerator

if __name__ == '__main__':

    num_train_graphs = 50
    num_validation_graphs = 10
    num_test_graphs = 10

    n = 10

    gen_params = get_default_gen_params(n)
    file_paths = get_default_file_paths()
    options = get_default_options(file_paths)

    agent_class = TargetMinCostAgent
    #agent_class = TargetHubsAgent
    #agent_class = RandomAgent
    #agent_class = ExhaustiveSearchAgent
    #agent_class = SimulatedAnnealingAgent
    #agent_class = BestResponseAgent
    #agent_class = PayoffTransferAgent

    hyperparams = {}
    # hyperparams = agent_class.get_default_hyperparameters()
    # hyperparams['eps'] = 10
    # hyperparams['c_threshold'] = 10 ** 4


    storage_root = Path('/experiment_data/stored_graphs')
    kwargs = {'store_graphs': True, 'graph_storage_root': storage_root}

    gen = BANetworkGenerator(**kwargs)
    graph_seeds = NetworkGenerator.construct_network_seeds(num_train_graphs, num_validation_graphs, num_test_graphs)
    _, _, test_graph_seeds = graph_seeds

    test_graphs = gen.generate_many(gen_params, test_graph_seeds)

    env = GraphMISEnv(SocialWelfare(), {}, heterogenous_cost=True)
    #env = GraphMISEnv(Fairness(), {})
    env.assign_env_specific_properties(test_graphs)

    agent = agent_class(env)
    agent.setup(options, hyperparams)
    avg_perf = agent.eval(test_graphs)
    print(f"final eval performance was {avg_perf}")