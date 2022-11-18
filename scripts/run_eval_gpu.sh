echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run.sh DEVICE_ID DATASET_NAME CKPT_PATH"
echo "For example: bash run_eval.sh device_id dataset ckpt_url"
echo "It is better to use the absolute path."
echo "=============================================================================================================="
set -e

DEVICE_ID=$1
DATASET_NAME=$2
CKPT_PATH=$3
export DEVICE_ID
export DATASET_NAME
export CKPT_PATH

EXEC_PATH=$(pwd)
echo "$EXEC_PATH"

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

cd ../
rm -rf eval/
mkdir eval
cd ./eval
mkdir src
cd ../
cp ./eval.py ./eval
cp ./src/*.py ./eval/src
cd ./eval

env > env0.log
echo "Eval begin."
python eval.py --device_id $1 --dataset $2 --ckpt_url $3 --is_modelarts False --device_target "GPU" > ./eval.log 2>&1 &

if [ $? -eq 0 ];then
    echo "evaling success"
else
    echo "evaling failed"
    exit 2
fi
echo "finish"
cd ../
