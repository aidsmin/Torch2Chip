
save_path="./save/${dataset}/${model}/${model}_w${wbit}_a${abit}_lr${lr}_batch${batch_size}_${loss}loss/"

if [ ! -d "$DIRECTORY" ]; then
    mkdir ../save
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3

model=resnet18_cifar
wbit=4
abit=4
epochs=200
batch_size=64
lr=0.1
loss=cross_entropy
weight_decay=0.0002
dataset="cifar10"
log_file="training.log"

python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 ./ddp.py \
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
    --mixed_prec True \
    --optimizer sgd \
    --trainer ddp \
    --ddp True \
    --weight-decay ${weight_decay} \