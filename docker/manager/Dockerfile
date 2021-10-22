FROM relnet/relnet
ARG RN_GID
ARG RN_GNAME

RUN source activate relnet-cenv \
    && pip install torch==0.4.0 -f https://download.pytorch.org/whl/torch_stable.html \
    && pip install cffi==1.11.2

RUN if ! [[ $(getent group ${RN_GNAME}) ]]; \
    then groupadd -g ${RN_GID} ${RN_GNAME}; \
    fi

RUN if ! [[ $(getent passwd relnetmanager) ]]; \
    then useradd -r -u 12954 --create-home --home-dir /home/relnetmanager -g ${RN_GNAME} relnetmanager; \
    fi

RUN chgrp -R ${RN_GNAME} /relnet
RUN chmod -R g+rwx /relnet

# See https://askubuntu.com/questions/909277/avoiding-user-interaction-with-tzdata-when-installing-certbot-in-a-docker-contai
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y dvipng texlive-latex-extra texlive-fonts-recommended texlive-fonts-extra

RUN source activate relnet-cenv; python -m ipykernel install --user --name relnet --display-name "py3-relnet"

RUN wget https://raw.githubusercontent.com/rabbitmq/rabbitmq-management/v3.7.15/bin/rabbitmqadmin -P /usr/bin && \
    chgrp ${RN_GNAME} /usr/bin/rabbitmqadmin && \
    chmod ug+x /usr/bin/rabbitmqadmin

RUN chgrp -R ${RN_GNAME} /var/lib/rabbitmq && \
    chmod g+rwx /var/lib/rabbitmq

RUN source activate relnet-cenv \
    && pip install geopy==2.0.0 \
    && pip install pyproj==2.6.1

RUN apt-get update && apt-get install -y libgeos-dev
RUN source activate relnet-cenv \
    && pip install git+https://github.com/matplotlib/basemap.git@3076ec9470cf7dba523bc94ebe5ae9a990e34d08 \
    && pip install pillow==7.2.0


USER relnetmanager
ENV PATH /opt/conda/envs/relnet-cenv/bin:$PATH
ENV RELNET_DEVICE_PLACEMENT CPU

RUN rabbitmq-plugins enable rabbitmq_management
ENV RABBITMQ_LOGS /tmp/rabbitmq-logs

# ports for flower, TF, jupyter, rabbitmq management
EXPOSE 5555 6006 8888 15672
CMD ["bash", "/relnet/docker/manager/start_manager.sh"]