cd Vc
cd build
rm -rf *
mkdir local

cmake -DCMAKE_INSTALL_PREFIX=`pwd`/local ../
make -j install

