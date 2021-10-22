#! /bin/bash

if ! [[ $1 = "up" || $1 = "stop" || $1 = "down" ]]; then
	echo "illegal first argument. must be one of [up, stop, down]."
	exit 1
fi

compose_command=$1
if [[ $compose_command == "up" ]]; then
	compose_command="$compose_command -d"
fi

container_name="relnet-worker-gpu"
if [[ ${compose_command:0:2} == up ]]; then
  if [[ "$(docker ps -a | grep $container_name)" ]]; then
    if [[ "$(docker inspect -f '{{.State.Running}}' $container_name)" = "true" ]]; then
      docker container stop $container_name
    fi
    docker container rm $container_name
  fi
  docker run -d --network "host" --hostname $container_name --name $container_name --gpus all --memory 62g --volume $RN_EXPERIMENT_DATA_DIR:/experiment_data --volume $RN_SOURCE_DIR:/relnet --env RN_GID --env RN_GNAME --env RN_LABORER_PW --env RN_ADMIN_PW relnet/relnet-worker-gpu
else
  if [[ "$(docker ps -a | grep $container_name)" ]]; then
    if [[ "$(docker inspect -f '{{.State.Running}}' $container_name)" = "true" ]]; then
      docker container stop $container_name
    fi
    docker container rm $container_name
  fi

fi
exit