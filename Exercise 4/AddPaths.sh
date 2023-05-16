PWD=`pwd`
export MY_VC_PATH="/opt/rbi/Vc/"
export PATH=$MY_VC_PATH":"$PATH 
export CPATH=$MY_VC_PATH"/include:"$CPATH 
export LD_LIBRARY_PATH=$MY_VC_PATH"/lib:"$LD_LIBRARY_PATH
export LIBRARY_PATH=$MY_VC_PATH"/lib:"$LIBRARY_PATH

export MY_TBB_PATH0=$PWD"/tbb42_20131118oss"
export MY_TBB_PATH=$MY_TBB_PATH0"/build/linux_intel64_gcc_cc4.7_libc2.13_kernel3.2.0_"
export PATH=$MY_TBB_PATH"release:"$MY_TBB_PATH"debug:"$PATH 
export CPATH=$MY_TBB_PATH0"/include:"$CPATH 
export LD_LIBRARY_PATH=$MY_TBB_PATH"release:"$MY_TBB_PATH"debug:"$LD_LIBRARY_PATH
export LIBRARY_PATH=$MY_TBB_PATH"release:"$MY_TBB_PATH"debug:"$LIBRARY_PATH

export LD_LIBRARY_PATH=/opt/rbi/drupal/lib/root
alias root=/opt/rbi/drupal/bin/root