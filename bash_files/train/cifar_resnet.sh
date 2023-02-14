PYTHON="/home/mengjian/anaconda3/envs/myenv/bin/python"

if [ ! -d "$DIRECTORY" ]; then
    mkdir ../save
fi

export CUDA_VISIBLE_DEVICES=0

model=resnet20_Q
wbit=4
abit=4
epochs=200
batch_size=128
lr=0.1
loss=cross_entropy
weight_decay=0.0002
dataset="cifar10"
save_path="./save/${dataset}/${model}/${model}_w${wbit}_a${abit}_lr${lr}_batch${batch_size}_${loss}loss/"
log_file="training.log"

$PYTHON -W ignore ./main.py \
    --save_path ${save_path} \
    --model ${model} \
    --epochs ${epochs} \
    --log_file ${log_file} \
    --lr ${lr} \
    --batch_size ${batch_size} \
    --loss_type ${loss} \
    --wbit ${wbit} \
    --abit ${abit} \
    --dataset ${dataset} \
    --ngpu 1 \
    --optimizer sgd \
    --weight-decay ${weight_decay} \