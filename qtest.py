"""
Quantizer module test
"""

import torch
from methods import SAWB, RCF

def test():
    wf = torch.randn(32, 32, 3, 3)
    qw = SAWB(nbit=4, train_flag=False)
    qw.inference()
    wq = qw(wf)
    print("Number of unique levels of quantized weights = {}".format(len(wq.unique())))
    print("Scaling factor of weight quantizer = {}".format(qw.scale))
    print("Quantization boundary of the weight quantizer = {}".format(qw.alpha))
    print("Unique levels of quantized weights = {}".format(wq.unique()))

    # activation
    xf = torch.randn(64, 32, 16, 16)
    relu = torch.nn.ReLU()
    qx = RCF(nbit=4, train_flag=True, alpha=6.0)
    xf = relu(xf)
    xq = qx(xf)
    print("Number of unique levels of quantized input = {}".format(len(xq.unique())))
    print("Scaling factor of input quantizer = {}".format(qx.scale))
    print("Quantization boundary of the input quantizer = {}".format(qx.alpha))

if __name__ == "__main__":
    test()
