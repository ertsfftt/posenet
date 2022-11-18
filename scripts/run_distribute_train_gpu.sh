echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_distribute_train.sh DATA_PATH RANK_SIZE"
echo "For example: bash run_distribute_train.sh /path/dataset 8"
echo "It is better to use the absolute path."
echo "=============================================================================================================="
set -e

export DEVICE_NUM=$1
export RANK_SIZE=$1
export DATASET_NAME=$2
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

cd ../
rm -rf ./train_parallel
mkdir ./train_parallel
cd ./train_parallel
mkdir src
cd ../
cp ../*.py ./train_parallel
cp ../src/*.py ./train_parallel/src
cd ./train_parallel
env > env.log
echo "start training"
    mpirun -n $1 --allow-run-as-root --output-filename log_output --merge-stderr-to-stdout \
           python train.py --device_num $1 \
                           --dataset $2 --is_modelarts False \
                           --run_distribute True \
                           --device_target "GPU" > train.log 2>&1 &

