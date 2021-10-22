from copy import deepcopy
from pathlib import Path

import pandas as pd
from pymongo import MongoClient

import projectconfig
from relnet.evaluation.file_paths import FilePaths
from relnet.state.network_generators import get_graph_ids_to_iterate


class EvaluationStorage:
    MONGO_EXPERIMENT_COLLECTION = 'experiment_data'
    MONGO_EVALUATION_COLLECTION = 'evaluation_data'

    def __init__(self):
        config = projectconfig.get_project_config()
        self.mongo_client = MongoClient(config.BACKEND_URL)
        self.db = self.mongo_client[config.MONGODB_DATABASE_NAME]

    def find_latest_experiment_id(self):
        result = self.db[self.MONGO_EXPERIMENT_COLLECTION].find().sort([("started_millis", -1)]).limit(1)[0]["experiment_id"]
        return result

    def get_hyperparameter_optimisation_data(self,
                                     experiment_id,
                                     model_seeds_to_skip,
                                     train_individually):

        latest_experiment = self.get_experiment_details(experiment_id)
        file_paths = latest_experiment["file_paths"]
        experiment_conditions = latest_experiment["experiment_conditions"]

        hyperopt_data = []

        network_generators = latest_experiment["network_generators"]
        objective_functions = latest_experiment["objective_functions"]
        agent_names = latest_experiment["agents"]
        param_spaces = latest_experiment["parameter_search_spaces"]

        for objective_function in objective_functions:
            for agent_name in agent_names:
                agent_grid = param_spaces[objective_function][agent_name]
                search_space_keys = list(agent_grid.keys())

                for hyperparams_id in search_space_keys:
                    for seed in experiment_conditions['experiment_params']['model_seeds']:
                        for network_generator in network_generators:

                            graph_ids_to_iterate = get_graph_ids_to_iterate(train_individually, network_generator, file_paths)
                            for graph_id in graph_ids_to_iterate:

                                setting = (network_generator, objective_function, agent_name, graph_id)
                                if setting in model_seeds_to_skip:
                                    if seed in model_seeds_to_skip[setting]:
                                        print(f"Skipping seed {seed} when computing optimal hyperparams.")
                                        continue

                                model_prefix = FilePaths.construct_model_identifier_prefix(agent_name,
                                                                                       objective_function,
                                                                                       network_generator,
                                                                                       seed,
                                                                                       hyperparams_id,
                                                                                       graph_id=graph_id)
                                hyperopt_result_filename = FilePaths.construct_best_validation_file_name(model_prefix)

                                hyperopt_result_path = Path(file_paths['hyperopt_results_dir'], hyperopt_result_filename)
                                if hyperopt_result_path.exists():
                                    with hyperopt_result_path.open('r') as f:
                                        avg_eval_reward = float(f.readline())

                                        hyperopt_data_row = {"network_generator": network_generator,
                                                             "objective_function": objective_function,
                                                             "agent_name": agent_name,
                                                             "hyperparams_id": hyperparams_id,
                                                             "avg_reward": avg_eval_reward,
                                                             "graph_id": graph_id}

                                        hyperopt_data.append(hyperopt_data_row)

        return param_spaces, pd.DataFrame(hyperopt_data)

    def retrieve_optimal_hyperparams(self,
                                     experiment_id,
                                     model_seeds_to_skip,
                                     train_individually):

        param_spaces, df = self.get_hyperparameter_optimisation_data(experiment_id,
                                                                     model_seeds_to_skip,
                                                                     train_individually)

        if not train_individually:
            df = df.drop(columns='graph_id')
        avg_rewards_df = df.groupby(list(set(df.columns) - {"avg_reward"})).mean().reset_index()
        gb_cols = list(set(avg_rewards_df.columns) - {"avg_reward", "hyperparams_id"})
        avg_rewards_max = avg_rewards_df.loc[avg_rewards_df.groupby(gb_cols)["avg_reward"].idxmax()].reset_index(
            drop=True)

        optimal_hyperparams = {}

        for row in avg_rewards_max.itertuples():
            if not train_individually:
                setting = row.network_generator, row.objective_function, row.agent_name
            else:
                setting = row.network_generator, row.objective_function, row.agent_name, row.graph_id
            optimal_id = row.hyperparams_id
            optimal_hyperparams[setting] = param_spaces[row.objective_function][row.agent_name][optimal_id], optimal_id

        return optimal_hyperparams

    def get_evaluation_data(self, experiment_id):
        eval_data = self.db[self.MONGO_EVALUATION_COLLECTION].find({"experiment_id": experiment_id})
        all_results_rows = []
        for eval_item in eval_data:
            all_results_rows.extend(list(eval_item['results_rows']))
        return all_results_rows

    def remove_evaluation_data(self, experiment_id):
        self.db[self.MONGO_EVALUATION_COLLECTION].remove({"experiment_id": experiment_id})

    def insert_experiment_details(self,
                                    file_paths,
                                    experiment_conditions,
                                    started_str,
                                    started_millis,
                                    parameter_search_spaces,
                                    experiment_id):
        all_experiment_details = {}
        all_experiment_details['experiment_id'] = experiment_id
        all_experiment_details['started_datetime'] = started_str
        all_experiment_details['started_millis'] = started_millis
        all_experiment_details['file_paths'] = {k: str(v) for k, v in dict(vars(file_paths)).items()}

        conds = dict(vars(deepcopy(experiment_conditions)))
        del conds["agents_models"]
        del conds["agents_baseline"]
        del conds["relevant_agents"]

        del conds["objective_functions"]
        del conds["network_generators"]
        del conds["model_seeds_to_skip"]

        all_experiment_details['experiment_conditions'] = conds
        all_experiment_details['agents'] = [agent.algorithm_name for agent in experiment_conditions.relevant_agents]
        all_experiment_details['objective_functions'] = [obj.name for obj in experiment_conditions.objective_functions]
        all_experiment_details['network_generators'] = [network_generator.name for network_generator in experiment_conditions.network_generators]
        all_experiment_details['parameter_search_spaces'] = parameter_search_spaces

        import pprint
        pprint.pprint(all_experiment_details)

        self.db[self.MONGO_EXPERIMENT_COLLECTION].insert_one(all_experiment_details)

        return all_experiment_details

    def get_experiment_details(self, experiment_id):
        return self.db[self.MONGO_EXPERIMENT_COLLECTION].find(
            {"experiment_id": {"$eq": experiment_id}}).limit(1)[0]

    def update_with_hyperopt_results(self, experiment_id, optimisation_result):
        self.db[self.MONGO_EXPERIMENT_COLLECTION].update_one({"experiment_id": experiment_id},
                                                                   {"$set": {"optimisation_result": optimisation_result}})

    def insert_evaluation_results(self, experiment_id, results_rows):
        self.db[self.MONGO_EVALUATION_COLLECTION].insert_one({"experiment_id": experiment_id,
                                                                "results_rows": results_rows})


    def fetch_all_eval_curves(self, agent_name, hyperparams_id, file_paths, objective_functions, network_generators, model_seeds, train_individually):
        all_dfs = []
        for obj_fun_name in objective_functions:
            for net_gen_name in network_generators:
                all_dfs.append(self.fetch_eval_curves(agent_name, hyperparams_id, file_paths, obj_fun_name, net_gen_name, model_seeds, train_individually))
        return pd.concat(all_dfs)

    def fetch_eval_curves(self, agent_name, hyperparams_id, file_paths, objective_function, network_generator, model_seeds, train_individually):
        eval_histories_dir = file_paths.eval_histories_dir
        if len(list(eval_histories_dir.iterdir())) == 0:
            return pd.DataFrame()

        data_dfs = []

        for seed in model_seeds:
            graph_ids = get_graph_ids_to_iterate(train_individually, network_generator, file_paths)

            for idx, g_id in enumerate(graph_ids):
                model_identifier_prefix = file_paths.construct_model_identifier_prefix(agent_name, objective_function, network_generator, seed, hyperparams_id, graph_id=g_id)
                filename = file_paths.construct_history_file_name(model_identifier_prefix)
                data_file = eval_histories_dir / filename
                if data_file.exists():
                    eval_df = pd.read_csv(data_file, sep=",", header=None, names=['timestep', 'perf'], usecols=[0,2])

                    model_seed_col = [seed] * len(eval_df)

                    eval_df['model_seed'] = model_seed_col
                    eval_df['objective_function'] = [objective_function] * len(eval_df)
                    eval_df['network_generator'] = [network_generator] * len(eval_df)
                    if g_id is not None:
                        eval_df['graph_id'] = [g_id] * len(eval_df)

                    data_dfs.append(eval_df)
        all_data_df = pd.concat(data_dfs)
        return all_data_df

