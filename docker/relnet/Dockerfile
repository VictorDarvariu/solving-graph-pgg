FROM relnet/base
SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y build-essential cmake sqlite3 libsqlite3-dev zip unzip \
    libboost-dev libboost-system-dev libboost-thread-dev libboost-serialization-dev \
    libboost-python-dev libboost-regex-dev libboost-numpy-dev

RUN apt-get update && apt-get install -y graphviz libgraphviz-dev graphviz-dev pkg-config

WORKDIR /relnet
ADD ./environment.yml /relnet/environment.yml
RUN conda env update --file environment.yml

ADD ./packaged_deps /relnet/packaged_deps
RUN cp -r /relnet/packaged_deps/pytorch_structure2vec /usr/lib

RUN cd /usr/lib \
    && cd pytorch_structure2vec \
    && cd s2v_lib \
    && make -j4

RUN chmod -R 777 /usr/lib/pytorch_structure2vec

ENV PYTHONPATH="/usr/lib/pytorch_structure2vec/s2v_lib:${PYTHONPATH}"

CMD ["/bin/bash"]