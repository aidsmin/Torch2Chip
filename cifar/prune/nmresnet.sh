PYTHON="/home2/jmeng15/anaconda3/envs/myenv/bin/python"

if [ ! -d "$DIRECTORY" ]; then
    mkdir ../save
fi

export CUDA_VISIBLE_DEVICES=1

model=resnet18_cifar
wbit=4
abit=4
epochs=200
batch_size=128
lr=0.01
loss=cross_entropy
weight_decay=0.0002
dataset="cifar10"
save_path="./save/${dataset}/prune/${model}/${model}_w${wbit}_a${abit}_lr${lr}_batch${batch_size}_${loss}loss/"
log_file="training.log"
pretrained_model="/home2/jmeng15/Torch2Chip/save/cifar10/resnet18_cifar/resnet18_cifar_w4_a4_lr0.1_batch128_cross_entropyloss/model_best.pth.tar"

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
    --mixed_prec True \
    --optimizer sgd \
    --weight-decay ${weight_decay} \
    --resume ${pretrained_model} \
    --fine_tune \
    --nchw False \
    --ltype "nm" \
    --final_prune_epoch 10 \
    --trainer "prune";