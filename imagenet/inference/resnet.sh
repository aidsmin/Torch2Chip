PYTHON="/home/jmeng15/anaconda3/envs/myenv/bin/python"


if [ ! -d "$DIRECTORY" ]; then
    mkdir ../save
fi

export CUDA_VISIBLE_DEVICES=0

model=resnet18
wbit=32
abit=32
epochs=90
batch_size=64
lr=0.1
loss=cross_entropy
weight_decay=1e-4
dataset="imagenet"
log_file="training.log"
save_path="./save/${dataset}/${model}/${model}_w${wbit}_a${abit}_lr${lr}_batch${batch_size}_${loss}loss/eval/"

# pretrained model
pretrained_model="/home/jmeng15/Torch2Chip/save/imagenet/resnet18/resnet18_w32_a32_lr0.1_batch64_cross_entropyloss/model_best.pth.tar"

mkdir ${save_path}

python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=0 ./ddp.py \
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
    --resume ${pretrained_model} \
    --fine_tune \
    --evaluate;
