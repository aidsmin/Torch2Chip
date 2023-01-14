"""
CPU based test function
"""

import time
from utils import accuracy, AverageMeter

def calibrate_model(model, loader):

    model.cpu()
    model.eval()

    for idx, (inputs, labels) in enumerate(loader):
        inputs = inputs.cpu()
        labels = labels.cpu()
        _ = model(inputs)
        if idx > 1:
            break

def cputest(model, testloader):
    top1 = AverageMeter()
    start = time.time()
    for idx, (inputs, target) in enumerate(testloader):
        out = model(inputs)
        prec1, prec5 = accuracy(out.data, target, topk=(1, 5))
        top1.update(prec1.item(), inputs.size(0))
        break
    t = time.time() - start
    return top1.avg, t