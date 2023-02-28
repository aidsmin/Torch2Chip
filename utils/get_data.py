"""
dataset and data loader
"""
import os
import torch
import torchvision.transforms as transforms
from torchvision import datasets

# Nvidia DALI
try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
    from nvidia.dali.pipeline import pipeline_def
    import nvidia.dali.types as types
    import nvidia.dali.fn as fn
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")

@pipeline_def
def create_dali_pipeline(data_dir, crop, size, shard_id, num_shards, dali_cpu=False, is_training=True):
    r"""
    DALI Pipeline adopted from:
    
    https://github.com/NVIDIA/DALI/blob/main/docs/examples/use_cases/pytorch/resnet50/main.py
    """
    images, labels = fn.readers.file(file_root=data_dir,
                                    shard_id=shard_id,
                                    num_shards=num_shards,
                                    random_shuffle=is_training,
                                    pad_last_batch=True,
                                    name="Reader")
    dali_device = 'cpu' if dali_cpu else 'gpu'
    decoder_device = 'cpu' if dali_cpu else 'mixed'
    # ask nvJPEG to preallocate memory for the biggest sample in ImageNet for CPU and GPU to avoid reallocations in runtime
    device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
    host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
    # ask HW NVJPEG to allocate memory ahead for the biggest image in the data set to avoid reallocations in runtime
    preallocate_width_hint = 5980 if decoder_device == 'mixed' else 0
    preallocate_height_hint = 6430 if decoder_device == 'mixed' else 0
    if is_training:
        images = fn.decoders.image_random_crop(images,
                                               device=decoder_device, output_type=types.RGB,
                                               device_memory_padding=device_memory_padding,
                                               host_memory_padding=host_memory_padding,
                                               preallocate_width_hint=preallocate_width_hint,
                                               preallocate_height_hint=preallocate_height_hint,
                                               random_aspect_ratio=[0.8, 1.25],
                                               random_area=[0.1, 1.0],
                                               num_attempts=100)
        images = fn.resize(images,
                           device=dali_device,
                           resize_x=crop,
                           resize_y=crop,
                           interp_type=types.INTERP_TRIANGULAR)
        mirror = fn.random.coin_flip(probability=0.5)
    else:
        images = fn.decoders.image(images,
                                   device=decoder_device,
                                   output_type=types.RGB)
        images = fn.resize(images,
                           device=dali_device,
                           size=size,
                           mode="not_smaller",
                           interp_type=types.INTERP_TRIANGULAR)
        mirror = False

    images = fn.crop_mirror_normalize(images.gpu(),
                                      dtype=types.FLOAT,
                                      output_layout="CHW",
                                      crop=(crop, crop),
                                      mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                      std=[0.229 * 255,0.224 * 255,0.225 * 255],
                                      mirror=mirror)
    labels = labels.gpu()
    return images, labels


def get_loader(args):
    # Preparing data
    if args.dataset == 'cifar10':
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]

        train_xforms = [transforms.RandomHorizontalFlip(), 
                        transforms.RandomCrop(32, padding=4)]

        train_xforms += [
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]

        train_transform = transforms.Compose(train_xforms)
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        trainset = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=train_transform)
        testset = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=test_transform)

        if args.ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, 
                            num_workers=args.workers, pin_memory=True, sampler=train_sampler)
        else:
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
        
        # test loader
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        num_classes, img_size = 10, 32
    
    elif args.dataset == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])  # here is actually the validation dataset

        train_dir = os.path.join(args.data_path, 'train')
        test_dir = os.path.join(args.data_path, 'val')

        trainset = datasets.ImageFolder(train_dir, transform=train_transform)
        testset = datasets.ImageFolder(test_dir, transform=test_transform)

        if args.ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, 
                            shuffle=True, num_workers=args.workers, pin_memory=True, sampler=train_sampler)
        else:
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
        
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
        num_classes, img_size = 1000, 224
    else:
        raise ValueError("Unrecegonized dataset!")
    
    return trainloader, testloader, num_classes, img_size

def dali_loader(args):
    """
    DALI loader for ImageNet training
    """
    crop_size = 224
    val_size = 256
    # train_loader
    pipe = create_dali_pipeline(batch_size=args.batch_size,
                                num_threads=args.workers,
                                device_id=args.local_rank,
                                seed=12 + args.local_rank,
                                data_dir='/dataset/imagenet/train/',
                                crop=crop_size,
                                size=val_size,
                                dali_cpu=args.dali_cpu,
                                shard_id=args.local_rank,
                                num_shards=args.world_size,
                                is_training=True)
    pipe.build()

    trainloader = DALIClassificationIterator(pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)

    # valid loader
    pipe = create_dali_pipeline(batch_size=args.batch_size,
                                num_threads=args.workers,
                                device_id=args.local_rank,
                                seed=12 + args.local_rank,
                                data_dir='/dataset/imagenet/val/',
                                crop=crop_size,
                                size=val_size,
                                dali_cpu=args.dali_cpu,
                                shard_id=args.local_rank,
                                num_shards=args.world_size,
                                is_training=False)
    pipe.build()
    testloader = DALIClassificationIterator(pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)

    num_classes, img_size = 1000, 224
    return trainloader, testloader, num_classes, img_size