"""
Quantization methods
"""

import torch

def stats_quant(x, nbit, qmode='symm', dequantize=True):
    z_typical = {'4bit': [0.077, 1.013], '8bit':[0.027, 1.114]}
    z = z_typical[f'{int(nbit)}bit']

    m = x.abs().mean()
    std = x.std()

    if qmode == 'symm':
        n_lv = 2 ** (nbit - 1) - 1
        alpha_w = 1/z[0] * std - z[1]/z[0] * m
    elif qmode == 'asymm':
        n_lv = (2 ** (nbit) - 1)/2
        alpha_w = 2*m
    else:
        raise NotImplemented

    x = x.clamp(-alpha_w.item(), alpha_w.item())
    scale = n_lv / alpha_w
    
    xq = x.mul(scale).round()
    if len(xq.unique()) > 2**nbit:
        xq = xq.clamp(-2**nbit//2, 2**nbit//2-1)
    
    if dequantize:
        xq = xq.div(scale)
    return xq, scale

class STE(torch.autograd.Function):
    """
    Straight through estimator
    """
    @staticmethod
    def forward(ctx, input, scale):
        input_q = input.mul(scale).round()
        out = input_q.div(scale)
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None

class RoundUQ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, alpha, nbit): 
        ctx.save_for_backward(input, alpha)

        scale = (2**nbit - 1) / alpha
        input_div = input.mul(scale)
        input_q = input_div.round().div(scale)
        return input_q

    @staticmethod
    def backward(ctx, grad_output):
        input, alpha = ctx.saved_tensors

        lower_bound = input < 0
        upper_bound = input > alpha

        x_range = ~(lower_bound|upper_bound)

        grad_alpha = torch.sum(grad_output * torch.ge(input, alpha).float()).view(-1)
        grad_input = grad_output * x_range.float()
        return grad_input, grad_alpha, None

class RCFQuantUQ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, alpha, nbit):
        # clamp
        input = input.div(alpha)
        input_c = input.clamp(max=1)
        scale = 2**nbit - 1

        # quant and de-quant
        input_div = input_c.mul(scale)
        input_q = input_div.round()

        ctx.save_for_backward(input, input_q)
        input_q = input_q.div(scale).mul(alpha)
        return input_q
    
    @staticmethod    
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()             # grad for weights will not be clipped
        input, input_q = ctx.saved_tensors
        i = (input.abs()>1.).float()
        sign = input.sign()
        grad_alpha = (grad_output*(sign*i + (input_q-input)*(1-i))).sum()
        return grad_input, grad_alpha, None