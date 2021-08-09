# #!bin/bash

export ROOT=$(pwd)
ckpt_dir=$1
ckpt=$2
lang_s=$3
d_idx=$4
direction=$5

mkdir -p out/$ckpt
exp_name="_eval.txt"

result=`python generate_onlyBleu.py config.yaml \
--task shared-multilingual-translation --dataset-name $lang_s \
--dataset-idx $d_idx --$direction --sacrebleu \
--path $ROOT/checkpoints/$ckpt_dir/$ckpt.pt \
--remove-bpe 'sentencepiece' `
echo $result >> out/$ckpt/$lang_s$exp_name

