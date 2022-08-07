# Torch2Chip: End-to-end Pytorch-based tool for DNN hardware deployment (Beta)
A library of tool for Pytorch-based deep neural network (DNN) compression and the subsequent preparation for hardware deployment. The tool aim to provide the end-to-end solution from compressed DNN training all the way to layer fusion and parameter extraction. Unlike the TF-Lite of Pytorch-Quantization tools, this library supports user-customized quantization methods and allows the post-training integer-only parameter conversion, including integer-only weights and scaling and shifting operations. More details are provided in the following documentation.

## Outline

- **Layers**
  - Customized quantization method
    - QBase
  - Customized quantization layer
    - QBaseConv2d
    - QBaseLinear
- **Model Training**
  - Trainers
- **Post-training model conversion**
  - Fuse the quantization scalers
  - Fuse the normalization parameters in to scalers and bias
  - T2C: Integer-only parameter conversion and parameter extraction
- **Notes for transformer**
- **Usage and requirements**

## Layers

This section introduces the customized quantization modules and layers for DNNs or vision transformers (ViT). Since the INT8 post-training quantization methods are quite popular, the basic quantization methods of this library are all designed for low-precision (e.g., 4-bit) quantization-aware-training (QAT). 

------

### Customized quantization method

Basic template for quantization method.

#### QBase ([source code](https://github.com/mengjian0502/Torch2Chip/blob/871de10b7e7f9e2c105af05116489804a213ee24/methods/base.py#L10))

The base method for quantization is `QBase`, the user-customized quantization methods should be constructed on top of  `QBase` 

```python
class QBase(nn.Module):
    """
    Basic quantizer module
    """
    def __init__(self, nbit:int, train_flag:bool=True):
        super(QBase, self).__init__()
        self.nbit = nbit
        self.train_flag = train_flag
        self.dequantize = True
        self.qflag = True
    
    def q(self, input:Tensor):
        """
        Quantization operation
        """
        return input
    
    def trainFunc(self, input:Tensor):
        """
        Forward pass of quantization-aware training 
        """
        out = self.q(input)
        return out
    
    def evalFunc(self, input:Tensor):
        return self.trainFunc(input)
    
    def inference(self):
        """
        Inference mode
        """
        self.train_flag = False
        self.dequantize = False

    def forward(self, input:Tensor):
        if self.train_flag:
            y = self.trainFunc(input)
        else:
            y = self.evalFunc(input)
        return y
    
    def extra_repr(self) -> str:
        return super().extra_repr() + "nbit={}".format(self.nbit)
```

`QBase` provides high degree of freedom for customization based on the following basic class parameters and methods:

**Class Parameters:**

- `nbit `(*int*): Quantization precisions
- `train_flag` (*bool*): Specify the normal training/validation process or integer only conversion
- `dequantize` (*bool*): Dequantization flag 
- `qflag`(*bool*): Quantization flag

**Class Methods**:

- `q`: Customized quantization methods.
  - An extra backward function (e.g., [STE](https://github.com/mengjian0502/Torch2Chip/blob/75c3f4a8a7eb2ff701583f2e277654efde485c29/methods/q.py#L34)) is required to perform the correct training process.
- `trainFunc`: Forward pass of the quantization-aware-training.
- `evalFunc`: Forward pass of the evaluation (no gradient-trace involved).
  - *Default*: Same as trianing.
- `inference`: Activate inference mode for integer-conversion and integer-only-inference.
- `extra_repr`: Show module info.

This library provides two example quantizers as the references: 

- Weight quantization: `SAWB` ([**paper**](https://mlsys.org/Conferences/2019/doc/2019/168.pdf)) ([**code**](https://github.com/mengjian0502/Torch2Chip/blob/75c3f4a8a7eb2ff701583f2e277654efde485c29/methods/qlayer.py#L14))
- Activation quantization: `RCF` ([**paper**](https://openreview.net/pdf?id=BkgXT24tDS)) ([**code**](https://github.com/mengjian0502/Torch2Chip/blob/75c3f4a8a7eb2ff701583f2e277654efde485c29/methods/qlayer.py#L67))

Regarding the details of the autograd backward pass function, please refer to the official Pytorch documentation ([link](https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function)).

------

### Customized quantization layers

On top of the base quantization method, Torch2Chip provides the layer template for customized quantization-aware-training and inference.

#### QBaseConv2d ([source code](https://github.com/mengjian0502/Torch2Chip/blob/871de10b7e7f9e2c105af05116489804a213ee24/methods/base.py#L54))

Basic method for low-precision convolution.

```python
class QBaseConv2d(nn.Conv2d):
    """
    Basic low precision convolutional layer
    """
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int, stride:int=1, 
                padding:int=0, dilation:int=1, groups:int=1, bias:bool=True, wbit:int=32, abit:int=32, train_flag=True):
        super(QBaseConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.train_flag = train_flag
        
        self.wbit = wbit
        self.abit = abit
        
        # quantizer
        self.wq = nn.Identity()
        self.aq = nn.Identity()
    
    def inference(self):
        """
        Inference mode
        """
        ...

    def get_fm_info(self, y:Tensor):
        """
        Get info of the layer
        """
        ...

    def forward(self, input:Tensor):
      	"""
      	Forward pass
	      """
        wq = self.wq(self.weight)
        xq = self.aq(input)
        y = F.conv2d(xq, wq, self.bias, self.stride, self.padding, self.dilation, self.groups)
        
        ...

        return y
```

**Class Parameters:**

In addition to the basic parameters/configuration of the Pytorch convolutional layer (`nn.Conv2d`), `QBaseConv2d` requires the following parameters/variables for quantization: 

- `wbit`(*int*): Weight precision
- `abit`(*int*): Activation/input precision
- `wq` (*QBase*): Weight quantizer
  - *Default*: `nn.Identity`
- `aq` (*QBase*): Input quantizer
  - *Default:* `nn.Identity`

**Class Methods**:

- `inference`: Activate inference mode for integer-conversion and integer-only-inference.
- `get_fm_info`: Get info of output feature map, including maximum data precision and feature map size.
- `forward`: Forward pass of convolutional layer

The detailed content of the class methods can be found in the source code.  

#### QBaseLinear ([source code](https://github.com/mengjian0502/Torch2Chip/blob/871de10b7e7f9e2c105af05116489804a213ee24/methods/base.py#L100))

Similar to `QBaseConv2d`, except the forward operation becomes `torch.nn.functional.linear`. 



