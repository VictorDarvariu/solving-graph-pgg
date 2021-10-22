#! /bin/bash

source activate relnet-cenv

rabbitmq-server &
sleep 30

if ! [[ $(rabbitmqctl list_users -q | grep relnetlaborer) ]]; then
    rabbitmqctl add_user relnetlaborer $RN_LABORER_PW
    rabbitmqctl add_vhost relnetvhost
    rabbitmqctl set_user_tags relnetlaborer laborer
    rabbitmqctl set_permissions -p relnetvhost relnetlaborer ".*" ".*" ".*"
fi

if ! [[ $(rabbitmqctl list_users -q | grep relnetadmin) ]]; then
    rabbitmqctl add_user relnetadmin $RN_ADMIN_PW
    rabbitmqctl set_user_tags relnetadmin administrator
    rabbitmqctl set_permissions -p / relnetadmin ".*" ".*" ".*"
    rabbitmqctl set_permissions -p relnetvhost relnetadmin ".*" ".*" ".*"
fi

rabbitmqctl set_log_level debug
# tensorboard --logdir=/experiment_data/il_experiment/models/summaries &
jupyter notebook --no-browser --notebook-dir=/relnet --ip=0.0.0.0 &

sleep 10
flower -A tasks --port=5555 --persistent=True --db=/flower/flower &

tail -f /dev/null


