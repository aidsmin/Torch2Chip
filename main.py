"""
CIFAR-VIT training
"""
import os
import logging
import argparse
import torch
import models
from t2c import LayerFuser
from collections import OrderedDict
from utils import get_loader, str2bool
from trainer import BaseTrainer

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/ImageNet Training')
parser.add_argument('--model', type=str, help='model architecture')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--lr_sch', type=str, default='step', help='learning rate scheduler')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--batch_size', default=128, type=int, metavar='N', help='mini-batch size (default: 64)')

parser.add_argument('--weight-decay', default=1e-5, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
parser.add_argument('--log_file', type=str, default=None, help='path to log file')

# loss and gradient
parser.add_argument('--loss_type', type=str, default='cross_entropy', help='loss func')
parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer')

# precision
parser.add_argument('--wbit', type=int, default=4, help='activation precision')
parser.add_argument('--abit', type=int, default=4, help='Weight precision')

# dataset
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset: CIFAR10 / ImageNet_1k')
parser.add_argument('--data_path', type=str, default='./data/', help='data directory')

# model saving
parser.add_argument('--save_path', type=str, default='./save/', help='Folder to save checkpoints and log.')
parser.add_argument('--evaluate', action='store_true', help='evaluate the model')
parser.add_argument('--save_param', action='store_true', help='save the model parameters')

# online logging
parser.add_argument("--wandb", type=str2bool, nargs='?', const=True, default=False, help="enable the wandb cloud logger")
parser.add_argument("--name")
parser.add_argument("--project")
parser.add_argument("--entity", default=None, type=str)

# Acceleration
parser.add_argument('--ngpu', type=int, default=2, help='0 = CPU.')
parser.add_argument('--workers', type=int, default=16,help='number of data loading workers (default: 2)')

# Fine-tuning
parser.add_argument('--fine_tune', dest='fine_tune', action='store_true',
                    help='fine tuning from the pre-trained model, force the start epoch be zero')
parser.add_argument('--resume', default='', type=str, help='path of the pretrained model')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
args.use_cuda = args.ngpu > 0 and torch.cuda.is_available()

def main():
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    # initialize terminal logger
    logger = logging.getLogger('training')
    if args.log_file is not None:
        fileHandler = logging.FileHandler(args.save_path+args.log_file)
        fileHandler.setLevel(0)
        logger.addHandler(fileHandler)
    streamHandler = logging.StreamHandler()
    streamHandler.setLevel(0)
    logger.addHandler(streamHandler)
    logger.root.setLevel(0)
    logger.info(args)

    # dataloaders
    trainloader, testloader, num_classes, img_size = get_loader(args)
    args.num_classes = num_classes
    
    # model
    model_cfg = getattr(models, args.model)
    model_cfg.kwargs.update({"num_classes": num_classes, "wbit": args.wbit, "abit":args.abit})
    model = model_cfg.base(*model_cfg.args, **model_cfg.kwargs) 
    logger.info(model)

    # resume from the checkpoint
    if args.fine_tune:
        checkpoint = torch.load(args.resume)
        sdict = checkpoint['state_dict']

        new_state_dict = OrderedDict()
        logger.info("=> loading checkpoint...")
        
        for k, v in sdict.items():
            name = k[7:]   
            new_state_dict[name] = v
        
        state_tmp = model.state_dict()
        state_tmp.update(new_state_dict)

        model.load_state_dict(state_tmp)
        logger.info("=> loaded checkpoint! acc = {}%".format(checkpoint['acc']))

    trainer = BaseTrainer(
        model=model,
        loss_type=args.loss_type,
        trainloader=trainloader,
        validloader=testloader,
        args=args,
        logger=logger,
    )

    if args.evaluate:
        trainer.valid_epoch()
        print("Test accuracy = {:.3f}".format(trainer.logger_dict["valid_top1"]))
        
        # post training fusion
        fuser = LayerFuser(model)
        fuser.layers()
        fused_model = fuser.fuse()
        print(fused_model)

        # update model
        setattr(trainer, "model", fused_model)
        trainer.valid_epoch()
        print("After fusion Test accuracy = {:.3f}".format(trainer.logger_dict["valid_top1"]))

        exit()

    # start training
    trainer.fit()

if __name__ == '__main__':
    main()