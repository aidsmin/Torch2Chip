PYTHON="/home/mengjian/anaconda3/envs/myenv/bin/python"

if [ ! -d "$DIRECTORY" ]; then
    mkdir ../save
fi

export CUDA_VISIBLE_DEVICES=0

model=mobilenetv1_Q
wbit=32
abit=32
epochs=200
batch_size=128
lr=0.05
loss=cross_entropy
weight_decay=1e-5

dataset="cifar10"
save_path="../save/cifar10/mobilenetv1_Q/mobilenetv1_Q_w32_a32_lr0.05_batch128_cross_entropyloss/eval/"
pretrained_model="../save/cifar10/mobilenetv1_Q/mobilenetv1_Q_w32_a32_lr0.05_batch128_cross_entropyloss/model_best.pth.tar"
log_file="training.log"

$PYTHON -W ignore ../ptq.py \
    --save_path ${save_path} \
    --model ${model} \
    --epochs ${epochs} \
    --log_file ${log_file} \
    --lr ${lr} \
    --loss_type ${loss} \
    --wbit ${wbit} \
    --abit ${abit} \
    --dataset ${dataset} \
    --optimizer sgd \
    --fine_tune \
    --resume ${pretrained_model} \
    --ngpu 1 \
    --evaluate;