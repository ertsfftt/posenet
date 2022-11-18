if [ $# -ne 3 ]
then
  echo "Wrong parameter format."
  echo "Usage:"
  echo "         bash $0 [INPUT_AIR_PATH] [AIPP_PATH] [OUTPUT_OM_PATH_NAME]"
  echo "Example: "
  echo "         bash convert_om.sh  xxx.air ./aipp.cfg xx"

  exit 1
fi

# The model path to be converted
model_path=$1
# The Aipp configuration file path
aipp_cfg_file=$2
# The name of the generated model
output_model_name=$3

atc \
--model=$model_path \
--framework=1 \
--output=$output_model_name \
--input_format=NCHW \
--input_shape="x:1,3,224,224" \
--enable_small_channel=1 \
--log=error \
--soc_version=Ascend310 \
--insert_op_conf=$aipp_cfg_file
