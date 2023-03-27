
if [ ! -d "$DIRECTORY" ]; then
    mkdir ../save
fi

export CUDA_VISIBLE_DEVICES=1

model=resnet18_cifar
wbit=4
abit=4
epochs=200
batch_size=128
lr=0.05
loss=cross_entropy
weight_decay=1e-5

dataset="cifar10"
save_path="/home2/jmeng15/Torch2Chip/save/cifar10/prune/resnet18_cifar/resnet18_cifar_w4_a4_lr0.01_batch128_cross_entropyloss/eval/"
pretrained_model="/home2/jmeng15/Torch2Chip/save/cifar10/prune/resnet18_cifar/resnet18_cifar_w4_a4_lr0.01_batch128_cross_entropyloss/model_best.pth.tar"
log_file="training.log"

python -W ignore ./main.py \
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
    --nchw False \
    --ltype "nm" \
    --evaluate;