#! /bin/bash

docker exec -it relnet-manager /bin/bash -c "bash /relnet/docker/manager/purge_queue.sh"
manage_container.sh stop && manage_container.sh up && manage_container_gpu.sh stop && manage_container_gpu.sh up