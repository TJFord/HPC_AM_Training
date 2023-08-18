wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.1.tar.gz

gunzip -c openmpi-4.0.1.tar.gz | tar xf -

cd openmpi-4.0.1

./configure --prefix=/home/$USER/.local # Install in the shared dir on the head-node

make all install

