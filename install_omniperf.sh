cd /
mkdr omniperf
cd omniperf

apt-get update
apt-get -y install wget
wget https://github.com/ROCm/rocprofiler-compute/releases/download/rocm-6.3.2/rocprofiler-compute-rocm-6.3.2.tar.gz

tar xfz rocprofiler-compute-rocm-6.3.2.tar.gz
cd rocprofiler-compute-3.0.0

# define top-level install path
export INSTALL_DIR=/omniperf

# install python deps
python3 -m pip install -t ${INSTALL_DIR}/python-libs -r requirements.txt

# configure Omniperf for shared install
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}/2.0.1 \
        -DPYTHON_DEPS=${INSTALL_DIR}/python-libs \
        -DMOD_INSTALL_PATH=${INSTALL_DIR}/modulefiles/omniperf ..

# install
make install

export PATH=$INSTALL_DIR/2.0.1/bin:$PATH
export PYTHONPATH=$INSTALL_DIR/python-libs
