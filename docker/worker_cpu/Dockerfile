FROM relnet/relnet
ARG RN_GID
ARG RN_GNAME

# GPU image and newer versions also available, check https://hub.docker.com/r/tensorflow/tensorflow.
# Can replace with 1.12.3-gpu-py3; but host needs to have https://github.com/NVIDIA/nvidia-docker set up.

RUN source activate relnet-cenv \
    && pip install torch==0.4.0 -f https://download.pytorch.org/whl/torch_stable.html \
    && pip install cffi==1.11.2

RUN if ! [[ $(getent group ${RN_GNAME}) ]]; \
    then groupadd -g ${RN_GID} ${RN_GNAME}; \
    fi

RUN if ! [[ $(getent passwd relnetworker) ]]; \
    then useradd -r -u 20627 --create-home --home-dir /home/relnetworker -g ${RN_GNAME} relnetworker; \
    fi

RUN chgrp -R ${RN_GNAME} /relnet
RUN chmod -R g+rwx /relnet

RUN source activate relnet-cenv \
    && pip install geopy==2.0.0 \
    && pip install pyproj==2.6.1

USER relnetworker

ENV PATH /opt/conda/envs/relnet-cenv/bin:$PATH
ENV RELNET_DEVICE_PLACEMENT CPU

CMD ["bash", "/relnet/docker/worker_cpu/start_worker.sh"]