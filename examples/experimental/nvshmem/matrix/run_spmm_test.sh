NPERNODE=4

# total GPUs
NPROC=$NPERNODE
if [ "${NHOSTS}" != "" ]; then
		    NPROC=$(expr $NHOSTS '*' $NPERNODE)
fi

ROOT_DIR=/home/users/chen.zhuang/bcl
N=16

mpirun -n $NPROC -npernode $NPERNODE ./spmm_test $ROOT_DIR/datasets/uk-2002/uk-2002.binary $N
# mpirun -n $NPROC -npernode $NPERNODE ./spmm_test $ROOT_DIR/datasets/arabic-2005/arabic-2005.binary $N
