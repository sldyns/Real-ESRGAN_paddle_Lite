#!/bin/bash
source test_tipc/common_func.sh

FILENAME=$1
MODE=$2

# MODE be one of ['lite_train_lite_infer']          

dataline=$(cat ${FILENAME})

# parser params
IFS=$'\n'
lines=(${dataline})

# The training params
model_name=$(func_parser_value "${lines[1]}")

trainer_list=$(func_parser_value "${lines[14]}")


if [ ${MODE} = "lite_train_lite_infer" ];then
    # prepare lite data
    rm -rf ./test_tipc/data/DF2K_HR
    rm -rf ./test_tipc/data/inputs
    cd ./test_tipc/data/ && unzip DF2K_HR.zip && unzip inputs.zip && cd ../../
    python tools/generate_meta_info.py --input ./test_tipc/data/DF2K_HR --root ./test_tipc/data/ --meta_info ./test_tipc/data/meta_info_DF2K.txt
fi