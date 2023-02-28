"""
Trainer for distributed training
"""

import time
import torch
import torch.nn as nn
from .trainer import BaseTrainer
from utils.utils import accuracy, AverageMeter, print_table, convert_secs2time, save_checkpoint


class DisTrainer(BaseTrainer):
    r"""
    Trainer for distributed training
    """
    def __init__(self, model: nn.Module, loss_type: str, trainloader, validloader, args, logger):
        super().__init__(model, loss_type, trainloader, validloader, args, logger)

        if args.ddp:
            self.local_rank = args.local_rank
            self.device = torch.device("cuda:{}".format(self.local_rank))
            self.model = self.model.to(self.device)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[args.local_rank])
        
        self.steps = len(self.trainloader)

    def adjust_lr(self, epoch):
        factor = epoch // 30

        if epoch >= 80:
            factor = factor + 1

        lr = self.args.lr*(0.1**factor)

        """Warmup"""
        if epoch < 5:
            lr = lr*float(1 + self.steps + epoch * self.args.epochs)/(5.* self.args.epochs)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
    
    def train_epoch(self):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        self.model.train()

        for idx, data in enumerate(self.trainloader):
            inputs = data[0]["data"]
            target = data[0]["label"].squeeze(-1).long()
            
            if self.args.use_cuda:
                inputs = inputs.to(self.device)
                target = target.to(self.device).squeeze(-1).long()
            
            out, loss = self.train_step(inputs, target)

            if self.local_rank == 0:
                acc1, acc5 = accuracy(out.data, target, topk=(1, 5))
                losses.update(loss.item(), inputs.size(0))
                top1.update(acc1.item(), inputs.size(0))
                top5.update(acc5.item(), inputs.size(0))
                
                # logger
                self.logger_dict["train_loss"] = losses.avg
                self.logger_dict["train_top1"] = top1.avg
                self.logger_dict["train_top5"] = top5.avg
        
    def valid_epoch(self):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        
        self.model.eval()
        with torch.no_grad():
            for idx, data in enumerate(self.validloader):
                inputs = data[0]["data"]
                target = data[0]["label"].squeeze(-1).long()
                
                if self.args.use_cuda:
                    inputs = inputs.cuda()
                    target = target.cuda(non_blocking=True)

                out, loss = self.valid_step(inputs, target)
                prec1, prec5 = accuracy(out.data, target, topk=(1, 5))

                losses.update(loss.mean().item(), inputs.size(0))
                top1.update(prec1.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))

        self.logger_dict["valid_loss"] = losses.avg
        self.logger_dict["valid_top1"] = top1.avg
        self.logger_dict["valid_top5"] = top5.avg
    
    def fit(self):
        self.logger.info("\nStart training: lr={}, loss={}, optim={}, ddp={}, lr_sch={}".format(self.args.lr, self.args.loss_type, self.args.optimizer, self.args.ddp, self.args.schedule))

        start_time = time.time()
        epoch_time = AverageMeter()
        best_acc = 0.
        for epoch in range(self.args.epochs):            
            # training and validation
            self.train_epoch()

            # lr step
            self.lr_scheduler.step()
            
            if self.local_rank == 0:
                # valid epoch
                self.valid_epoch()

                is_best = self.logger_dict["valid_top1"] > best_acc
                if is_best:
                    best_acc = self.logger_dict["valid_top1"]

                state = {
                    'state_dict': self.model.state_dict(),
                    'acc': best_acc,
                    'epoch': epoch,
                    'optimizer': self.optimizer.state_dict(),
                }

                filename=f"checkpoint.pth.tar"
                save_checkpoint(state, is_best, self.args.save_path, filename=filename)

                self.logger_dict["ep"] = epoch+1
                self.logger_dict["lr"] = self.optimizer.param_groups[0]['lr']

                # online log
                if self.args.wandb:
                    self.wandb_logger.log(self.logger_dict)

                # terminal log
                columns = list(self.logger_dict.keys())
                values = list(self.logger_dict.values())
                print_table(values, columns, epoch, self.logger)

                # record time
                e_time = time.time() - start_time
                epoch_time.update(e_time)
                start_time = time.time()

                need_hour, need_mins, need_secs = convert_secs2time(
                epoch_time.avg * (self.args.epochs - epoch))
                print('[Need: {:02d}:{:02d}:{:02d}]'.format(
                        need_hour, need_mins, need_secs))
                self.validloader.reset()
            
            self.trainloader.reset()
            
