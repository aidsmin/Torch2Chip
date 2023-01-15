PYTHON="/home/mengjian/anaconda3/envs/myenv/bin/python"

if [ ! -d "$DIRECTORY" ]; then
    mkdir ../save
fi

export CUDA_VISIBLE_DEVICES=0

model=mobilenetv1_Q
wbit=4
abit=4
epochs=200
batch_size=128
lr=0.01
loss=cross_entropy
weight_decay=0.0

dataset="cifar10"
save_path="../save/${dataset}/${model}/${model}_w${wbit}_a${abit}_lr${lr}_batch${batch_size}_${loss}loss/"
pretrained_model="../save/cifar10/mobilenetv1_Q/mobilenetv1_Q_w32_a32_lr0.05_batch128_cross_entropyloss/model_best.pth.tar"
log_file="training.log"

$PYTHON -W ignore ../main.py \
    --save_path ${save_path} \
    --model ${model} \
    --epochs ${epochs} \
    --log_file ${log_file} \
    --lr ${lr} \
    --loss_type ${loss} \
    --wbit ${wbit} \
    --abit ${abit} \
    --dataset ${dataset} \
    --resume ${pretrained_model} \
    --fine_tune \
    --ngpu 1 \
    --optimizer sgd \