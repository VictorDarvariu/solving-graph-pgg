import argparse
import math
import sys
import tqdm

sys.path.append('/relnet')

from relnet.evaluation.experiment_conditions import get_conditions_for_experiment
from relnet.evaluation.file_paths import FilePaths

from multiprocessing.pool import Pool
from psutil import cpu_count

def main():
    parser = argparse.ArgumentParser(description="Generate graph instances in parallel")
    parser.add_argument("--pool_size_multiplier", required=True, type=float,
                        help="Multiplier for worker pool size, applied to number of logical cores.",
                        )
    args = parser.parse_args()
    file_paths = FilePaths('/experiment_data', None, setup_directories=False)

    storage_root = file_paths.graph_storage_dir
    logs_file = str(file_paths.construct_log_filepath())
    kwargs = {'store_graphs': True, 'graph_storage_root': storage_root, 'logs_file': logs_file}

    num_procs = math.ceil(cpu_count(logical=True) * args.pool_size_multiplier)
    worker_pool = Pool(processes=num_procs)

    tasks = []
    for n in [15, 25, 50]:
        experiment_conditions = get_conditions_for_experiment('main', n, False, False)
        experiment_conditions.set_generator_seeds()

        gen_params = experiment_conditions.gen_params
        all_seeds = []
        all_seeds.extend(experiment_conditions.train_seeds)
        all_seeds.extend(experiment_conditions.validation_seeds)
        all_seeds.extend(experiment_conditions.test_seeds)


        for network_generator_class in experiment_conditions.network_generators:
            for net_seed in all_seeds:
                #print(f"size {n}: would add in task for seed {net_seed}!")
                tasks.append((network_generator_class, kwargs, gen_params, net_seed))

    #for net_seed in worker_pool.starmap(call_network_generator, tasks):
    for net_seed in worker_pool.starmap(call_network_generator, tqdm.tqdm(tasks, total=len(tasks)), chunksize=50):
        pass

    worker_pool.close()

def call_network_generator(network_generator_class, gen_kwargs, params, net_seed):
    gen_instance = network_generator_class(**gen_kwargs)
    gen_instance.generate(params, net_seed)
    return net_seed

if __name__ == "__main__":
    main()
