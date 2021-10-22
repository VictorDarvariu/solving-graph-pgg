#! /bin/bash

### Experiments for identical cost setting

docker exec -d relnet-manager /bin/bash -c "source activate relnet-cenv;  python run_experiments.py --experiment_part both --which il_imitate --base_n 15 --experiment_id ggnn_ili_15_ic --bootstrap_hyps_expid ggnn_main_15_ic --force_insert_details"

docker exec -d relnet-manager /bin/bash -c "source activate relnet-cenv;  python run_experiments.py --experiment_part both --which il_imitate --base_n 25 --experiment_id ggnn_ili_25_ic --bootstrap_hyps_expid ggnn_main_25_ic --force_insert_details"

docker exec -d relnet-manager /bin/bash -c "source activate relnet-cenv;  python run_experiments.py --experiment_part both --which il_imitate --base_n 50 --experiment_id ggnn_ili_50_ic --bootstrap_hyps_expid ggnn_main_50_ic --force_insert_details"

docker exec -d relnet-manager /bin/bash -c "source activate relnet-cenv;  python run_experiments.py --experiment_part both --which il_imitate --base_n 75 --experiment_id ggnn_ili_75_ic --bootstrap_hyps_expid ggnn_main_75_ic --force_insert_details"

docker exec -d relnet-manager /bin/bash -c "source activate relnet-cenv;  python run_experiments.py --experiment_part both --which il_imitate --base_n 100 --experiment_id ggnn_ili_100_ic --bootstrap_hyps_expid ggnn_main_100_ic --force_insert_details"

### Experiments for heterogenous cost setting

docker exec -d relnet-manager /bin/bash -c "source activate relnet-cenv;  python run_experiments.py --experiment_part both --which il_imitate --base_n 15 --heterogenous_cost --experiment_id ggnn_ili_15_hc --bootstrap_hyps_expid ggnn_main_15_hc --force_insert_details"

docker exec -d relnet-manager /bin/bash -c "source activate relnet-cenv;  python run_experiments.py --experiment_part both --which il_imitate --base_n 25 --heterogenous_cost --experiment_id ggnn_ili_25_hc --bootstrap_hyps_expid ggnn_main_25_hc --force_insert_details"

docker exec -d relnet-manager /bin/bash -c "source activate relnet-cenv;  python run_experiments.py --experiment_part both --which il_imitate --base_n 50 --heterogenous_cost --experiment_id ggnn_ili_50_hc --bootstrap_hyps_expid ggnn_main_50_hc --force_insert_details"

docker exec -d relnet-manager /bin/bash -c "source activate relnet-cenv;  python run_experiments.py --experiment_part both --which il_imitate --base_n 75 --heterogenous_cost --experiment_id ggnn_ili_75_hc --bootstrap_hyps_expid ggnn_main_75_hc --force_insert_details"

docker exec -d relnet-manager /bin/bash -c "source activate relnet-cenv;  python run_experiments.py --experiment_part both --which il_imitate --base_n 100 --heterogenous_cost --experiment_id ggnn_ili_100_hc --bootstrap_hyps_expid ggnn_main_100_hc --force_insert_details"
