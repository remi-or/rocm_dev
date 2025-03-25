apt-get update
apt-get install libnuma-dev
apt-get install libopenmpi-dev

cd /tmp
git clone https://github.com/microsoft/mscclpp.git
mkdir -p mscclpp/build && cd mscclpp/build

# This might not be needed
CXX=`which hipcc` cmake -DCMAKE_BUILD_TYPE=Release ..
make -j

# This is needed
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local/mscclpp -DMSCCLPP_BUILD_PYTHON_BINDINGS=OFF ..
make -j mscclpp mscclpp_static
sudo make install/fast
