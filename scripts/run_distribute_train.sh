echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run.sh DATASET_NAME RANK_SIZE"
echo "For example: bash run_distribute.sh dataset_name rank_table"
echo "It is better to use the absolute path."
echo "=============================================================================================================="
set -e
get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

DATASET_NAME=$1
export DATASET_NAME
export RANK_TABLE_FILE=$(get_real_path $2)
export RANK_SIZE=8

EXEC_PATH=$(pwd)
echo "$EXEC_PATH"

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

cd ../
for((i=0;i<8;i++))
do
    rm -rf device$i
    mkdir device$i
    cd ./device$i
    mkdir src
    cd ../
    cp ./train.py ./device$i
    cp ./src/*.py ./device$i/src
    cd ./device$i
    export DEVICE_ID=$i
    export RANK_ID=$i
    echo "start training for device $i"
    env > env$i.log
    python train.py --run_distribute True \
                    --dataset $1 --device_num 8 \
                    --is_modelarts False \
                    --device_target "Ascend" > train$i.log 2>&1 &
    echo "$i finish"
    cd ../
done

if [ $? -eq 0 ];then
    echo "training success"
else
    echo "training failed"
    exit 2
fi
echo "finish"
cd ../
