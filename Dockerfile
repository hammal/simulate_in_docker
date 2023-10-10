# A dockerfile describing a container with all necessary dependencies
# to use the full cbadc feature suit.

FROM ubuntu:latest

LABEL maintainer="Hampus Malmberg <hampus.malmberg88@gmail.com>"

USER root

# install ngspice dependencies
RUN apt-get update && apt-get install -y \
    bison \
    flex \
    build-essential \
    autoconf \
    automake \
    libtool \
    libxaw7-dev \
    libreadline-dev \
    git

# clone ngspice repo
RUN git clone git://git.code.sf.net/p/ngspice/ngspice
# configure and install ngspice
RUN cd ngspice && \
    ./autogen.sh && \
    ./configure --enable-xspice --enable-cider --disable-debug --with-readline=yes --enable-openmp && \
    make clean && \ 
    make && \
    make install && \
    cd .. && \
    rm -rf ngspice


# install rust
RUN apt-get update && apt-get install -y \
    curl

# install rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | bash -s -- -y

# put cargo into path
ENV PATH="${HOME}/.cargo/bin:${PATH}"

# clone calib
RUN git clone https://github.com/hammal/calib.git

# install calib
RUN cd calib && \
    RUSTFLAGS="-C target-cpu=native" ${HOME}/.cargo/bin/cargo install --path .

# install python
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip

# install cbadc
RUN git clone https://github.com/hammal/cbadc.git
RUN cd cbadc && \
    python3 -m pip install --upgrade pip black pytest && \ 
    python3 -m pip install -e .