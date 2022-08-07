# Torch2Chip: End-to-end Pytorch-based tool for DNN hardware deployment (Beta)

[Jian Meng](https://mengjian0502.github.io/) @ [SeoLab](https://faculty.engineering.asu.edu/jseo/) of Arizona State University

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
  - Quantization scaling
  - BatchNorm scaling and shifting
  - Fuse the normalization parameters in to scalers and bias
    - MulShift
    - MulQuant
    - LayerFuser
    - XFormerFuser
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

------

## Model Training

------

#### BaseTrainer ([source code](https://github.com/mengjian0502/Torch2Chip/blob/43034a8e558c8b897e4aef2a24eef231bcca39d0/trainer.py#L13))

The model training method is implemented in the `BaseTrainer` module ([source code](https://github.com/mengjian0502/Torch2Chip/blob/main/trainer.py)), where the entire training process has been decomposed into different stages to support the various customizations. 

```python
class BaseTrainer(object):
    def __init__(self,
        model: nn.Module,
        loss_type: str, 
        trainloader, 
        validloader,
        args,
        logger,
    ):
```

**Class Parameters:** 

- `model`(`torch.nn.Module`): DNN / Transformer model
- `loss_type`(*str*): Loss function type (Option: MSELoss, CE Loss, and Smooth CE loss)
- `trainloader`: Data loader of training set
- `validloader`: Data loader for valid / test set
- `args`: Global training argument, passed from the main file
- `logger`: Logger of training

**Class Methods**:

```markdown
Trainer.fit()
|__[train_epoch]
|	|__[train_step]
|		|__[base_forward]
|		|__[base_backward]
|
|__[valid_epoch]
	|__[valid_step]
		|_[base_forward]
```

**Example Usage:** **Four step setup to train a DNN model!**

```python
# Step1: Get dataloader
trainloader, testloader, num_classes, img_size = get_loader(args)
# Step2: Define your model
model = ...
# Step3: Initialize the trainer
trainer = BaseTrainer(
        model=model,
        loss_type=args.loss_type,
        trainloader=trainloader,
        validloader=testloader,
        args=args,
        logger=logger,
    )
# Step 4: Start DNN training with one-line execution
trainer.fit()
```

------

## Post training model conversion

------

### Quantization scaling

To maintain the same data range, the quantized integers have to be re-scaled back by a pre-computed scaling factor S. As demonstrated in `QBase`, the scaling factor is computed and updated during the entire training process and saved as a non-learable parameter of the model `scale`. Due to the property of convolution and weight multiplication, the scaling processs can be performed after the integer-only computation. 

Given the quantized weights $W_Q$ and quantized input $X_Q$, the forward pass of the inference is formulated as

$$
Y = \frac{1}{\text{scale}_X \times \text{scale}_W}(X_Q * W_Q)
$$

Where $\text{scale}_X$ and $\text{scale}_W$ are the scaling factors of the activation and weight quantizers, parametrized as the `scale` variable and embedded into the `state_dict` of the pre-trained model. 

On top of the scaling process of the current layer, the quantization (**not dequantization**) process of the subsequent layer should be collectively merged. In hardware, the quantization process is performed as 1) Scaling, 2) Rounding, 3) Clamping. Without considering BatchNorm, the scaling process can be merged into the above scaling process: 

$$
S = \frac{\text{scale}_X^{*}}{\text{scale}_X \times \text{scale}_W}
$$

Where ${\text{scale}_X^{*}}$​ represents the quantization scaling factor of the next layer. 

------

### BatchNorm scaling 

During the forward pass of the DNN inference, BatchNorm operation can be formulated as:

$$
\bar{Y} = \gamma\times \frac{Y-\mu}{\sigma} + \beta
$$

Where $\gamma$, $\mu$, $\sigma$, and $\beta$ represents the weight, running mean, running standard deviation, and bias of the BatchNorm module (`torch.nn.BatchNorm2d`). Such normalization process can further re-written as: 

$$
\bar{Y} = \frac{\gamma}{\sigma}Y + (\beta-\frac{\gamma \mu}{\sigma})
$$

Where ${\gamma}/{\sigma}$ and $(\beta-\frac{\gamma \mu}{\sigma})$ are characterized as the scaling factor and bias values. 

Together with the quantization scaling, the re-formmulated scaling factor $S$ and bias $b$ are :

$$
S = \frac{\gamma}{\sigma} \frac{\text{scale}_X^{*}}{\text{scale}_X \times \text{scale}_W}
$$

$$
b = (\beta-\frac{\gamma \mu}{\sigma}) \times \text{scale}_X^{*}
$$

------

### Fuse the normalization parameters in to scalers and bias

Based on the mathematical derivation above, Torch2Chip library implemented the scaling and shifting process based on the following customized modules, which are also closely corporated with the model fusion modules. 

#### MulShift ([source code](https://github.com/mengjian0502/Torch2Chip/blob/3a8766d5f0eecbbf332f637ab7a59e3196f290fa/methods/base.py#L127))

```python
class MulShift(nn.Module):
    def __init__(self):
        super(MulShift, self).__init__()
        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("bias", torch.tensor(0.0))
        
        # fractional bit width
        self.fl = 0.

    def forward(self, input:Tensor):
        out = input.mul(self.scale[None, :, None, None]).add(self.bias[None, :, None, None])
        out = out.mul(2**(-self.fl))
        return out
```

**Class Parameters:**

- `scale`(`torch.Tensor`): Merged scaling factor.
- `bias`(`torch.Tensor`): Merged bias.
- `fl`: Shifted fractional bits after the integer operation.

#### MulQuant ([source code](https://github.com/mengjian0502/Torch2Chip/blob/871de10b7e7f9e2c105af05116489804a213ee24/methods/base.py#L160))

Currently, the scaling and output quantization process are available for vision transformer only. The detailed implementation is available in the `transformer` branch ([link](https://github.com/mengjian0502/Torch2Chip/tree/transformer)). 

```python
class MulQuant(nn.Module):
    def __init__(self, nbit:int=4):
        super(MulQuant, self).__init__()
        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("bias", torch.tensor(0.0))
        self.nbit = nbit
        self.nlv = 2**(nbit-1) - 1

        # fractional bit width
        self.fl = 0.

    def forward(self, input:Tensor):
        out = input.mul(self.scale)
        out = out.mul(2**(-self.fl)).round()

        out = out.clamp(min=-self.nlv, max=self.nlv)
        return out
```

Similar to `MulShift`, `MulQuant` module first applies the scaling and shifting to the input tensor (resultant of linear or convolution operation), then rounds the scaled value to the nearest integer, followed by the clamping operation to clip the value range. 

**Class Parameters:**

- `scale`(`torch.Tensor`): Merged scaling factor.
- `bias`(`torch.Tensor`): Merged bias.
- `fl`: (*int*) Shifted fractional bits after the integer operation.
- `nbit`:(*int*) Input precision of the subsequent layer.

#### LayerFuser ([source code](https://github.com/mengjian0502/Torch2Chip/blob/3a8766d5f0eecbbf332f637ab7a59e3196f290fa/t2c/fuser.py#L11))

Fuse the Conv-BN-ReLU layers altogether, then return a newly-constructed fused model:

```python
class LayerFuser(object):
    def __init__(self, model:nn.Module):
        self.model = model
        # flag
        self.flag = False
        
        # layers
        self.groups = []

    def inference(self, model:nn.Module):
        """
        Switch to inference mode
        """

    def layers(self):
        """
        Fetch layer information from pretrained model
        """
       
    def fuse(self):
        """
        Fuse conv, bn, and relu layers
        """
```

**Class Parameters:**

- `model`(`nn.Module`): Pre-trained low precision model
- `flag` (*bool*): Flag of merging
- `groups` (*List*): Groups of Conv-BN-ReLU layer

**Class Methods:**

- `inference`: Switch the model (including sub-modules) to inference mode
- `layers`: Fetch layer information and Conv-BN-ReLU groups
- `fuse`: Model fusion

**Example Usage:**

Before fusion (Pretrained model):

```python
(3): QConv2d(
      128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
      (wq): SAWB(nbit=4)
      (aq): RCF(nbit=4)
    )
(4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
(5): ReLU(inplace=True)
```

After layer fusion:

```python
(3): ConvBNReLU(
      (conv): QConv2d(
        128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        (wq): SAWB(nbit=4)
        (aq): RCF(nbit=4)
      )
      (bn): Identity()
      (relu): ReLU(inplace=True)
      (scaler): MulShift()
    )
(4): Identity()
(5): Identity()
```

The original `BatchNorm2d` and `ReLU` modules are replaced by `nn.Identity()` modules. The post-convolution scaling are performed by the `MulShift` module. 

#### XformerFuser ([source code](https://github.com/mengjian0502/Torch2Chip/blob/871de10b7e7f9e2c105af05116489804a213ee24/t2c/xformer_fuser.py#L11))

```python
class XformerFuser(object):
    """
    Applies layer fusion for vision transformer (ViT)

    Args:
    - model: nn.Module, Pre-trained low-precision vision transformer
    """
    def __init__(self, model:nn.Module):
        self.model = model
    
    def inference(self, model:nn.Module):
        """
        Switch to inference mode
        """    
      
    def fused_linear(self, linear:QBaseLinear, qflag:bool=True, obit:int=32):
      	"""
      	Fuse the linear layer and scaling into the fused module
      	"""
      	...
    
    def encoder_fuser(self):
      	"""
      	Fuse the layers of multi head attention
      	"""
      	...
    
    def mlp_fuser(self, model:nn.Module):
      	"""
      	Fuse the multi-layer perceptron
      	"""
        ...

    def fuse(self):
      	"""
      	Fuse the entire transformer
      	"""
        ...
```

**Class Parameters:**

- `model`(`nn.Module`): Pre-trained low precision model

**Class Methods:**

- `inference`: Switch the model (including sub-modules) to inference mode
- `fused_linear`: Fuse the linear layer and scaling module into `LinearMulShift`([code](https://github.com/mengjian0502/Torch2Chip/blob/871de10b7e7f9e2c105af05116489804a213ee24/methods/base.py#L201)) or `LinearMulShiftReLU`([code](https://github.com/mengjian0502/Torch2Chip/blob/871de10b7e7f9e2c105af05116489804a213ee24/methods/base.py#L219)) 
- `encoder_fuser`: Fuse the multi-head-attention block
- `mlp_fuser`: Fuse the multi-layer perceptron

**Example Usage:**

Before fusion (Pretrained model):

```python
(msa): MultiHeadSelfAttention(
      (qq): RCFSQ(nbit=4)
      (kq): RCFSQ(nbit=4)
      (vq): RCFSQ(nbit=4)
      (oq): RCFSQ(nbit=4)
      (q): QLinear(
        in_features=384, out_features=384, bias=True
        (wq): SAWB(nbit=4)
        (aq): Identity()
      )
      (k): QLinear(
        in_features=384, out_features=384, bias=True
        (wq): SAWB(nbit=4)
        (aq): Identity()
      )
      (v): QLinear(
        in_features=384, out_features=384, bias=True
        (wq): SAWB(nbit=4)
        (aq): Identity()
      )
      (o): QLinear(
        in_features=384, out_features=384, bias=True
        (wq): SAWB(nbit=4)
        (aq): Identity()
      )
      (dropout): Dropout(p=0.0, inplace=False)
      (deq): MulShift()
      (vdeq): MulShift()
    )
```

After fusion:

```python
(msa): MultiHeadSelfAttention(
        (qq): Identity()
        (kq): Identity()
        (vq): Identity()
        (oq): Identity()
        (q): LinearMulShift(
          (linear): QLinear(
            in_features=384, out_features=384, bias=True
            (wq): SAWB(nbit=4)
            (aq): Identity()
          )
          (scaler): MulQuant()
        )
        (k): LinearMulShift(
          (linear): QLinear(
            in_features=384, out_features=384, bias=True
            (wq): SAWB(nbit=4)
            (aq): Identity()
          )
          (scaler): MulQuant()
        )
        (v): LinearMulShift(
          (linear): QLinear(
            in_features=384, out_features=384, bias=True
            (wq): SAWB(nbit=4)
            (aq): Identity()
          )
          (scaler): MulQuant()
        )
        (o): LinearMulShift(
          (linear): QLinear(
            in_features=384, out_features=384, bias=True
            (wq): SAWB(nbit=4)
            (aq): Identity()
          )
          (scaler): MulShift()
        )
        (dropout): Dropout(p=0.0, inplace=False)
        (deq): MulShift()
        (vdeq): MulQuant()
      )
```

The original quantization modules are all replaced by `nn.Identity` and the scaling, rounding process are embedded into the `MulQuant` module. 

------

### T2C: Integer-only parameter conversion and parameter extraction

#### T2C ([source code](https://github.com/mengjian0502/Torch2Chip/blob/94cc93a1ff1524bbcc12087b626ff9f0226729ac/t2c/t2c.py#L14))

Extract the weights and scaling parameters of pre-trained model. The model is fused inside `T2C` by using `LayerFuser` or `XformerFuser`

```python
class T2C(object):
    """
    Deploying the pretrained Pytorch model to hardware-feasible parameters: 
    - Layer fusion
    - Integer conversion
    - Parameter saving
    - Define the precision of the high precision scaling / shifting
    Args:
    - model: Pretrained DNN model (after fusion)
    - fuser: Model fuser (For CNN or Transformer)
    - swl: World length of the high precision scaling/shifting factor
    - swl: Fractional bits the high precision scaling/shifting factor
    """
    def __init__(self, model:nn.Module, fuser, swl:int, sfl:int, args):
        self.swl = swl
        self.sfl = sfl
        self.args = args
       
    def scale_bias2int(self, model:nn.Module):
        """
        Convert the pre-computed scaling factor and bias to high precision integer
        """
    
    def get_info(self, model:nn.Module):
      	"""
      	Get model info and the maximum data precision of the intermediate results
      	"""
        
    def nn2chip(self, save_model:bool=False):
      	"""
      	Convert and save the model
      	"""
```

**Class Parameters:**

- `model`(`nn.Module`): Pre-trained low precision model.
- `fuser`: Model fuser (Option: `LayerFuser` or `XformerFuser`)
- `swl` (*int*): Bit width of the high precision scaling and bias.
- `sfl` *(int)*: Fractional bit width of the high precision scaling and bias.
- `args`: Global argument, defined in the main file.

**Class Methods:**

- `scale_bias2int`: Convert the floating point scaling factor and bias into high precision fixed point integer. 
- `get_info`: Get the detailed information of the converted model, including the data precision and model parameters. 
- `nn2chip`: Convert and save the model

Given a pre-trained low-precision model, `T2C` first fuses the layers then convert the high precision parameters (e.g., scaling factors and bias) in to high precision integer, where the precision is directly specified by the user to justify the tradeoff between the accuracy and data precision. 

**Example Usage:** **3-line code for model fusion, conversion, and inference!**

```python
# Define T2C
nn2c = T2C(model, fuser=XformerFuser, swl=args.wl, sfl=args.fl, args=args)
# Fuse and convert the model
qnn = nn2c.nn2chip(save_model=True)
# Update the model of the trainer to perform inference
setattr(trainer, "model", qnn)
# Inference
trainer.valid_epoch()
```

**Experimental results with CIFAR-10 dataset**

| Model | W/A  |  S/b  | SW Baseline | T2C Acc. |
| :---: | :--: | :---: | :---------: | :------: |
| ViT7  | 4/4  | 16/16 |    88.54    |  88.49   |
| VGG7  | 4/4  | 16/16 |    92.55    |  92.51   |

------

## Notes for Transformer

For the Beta version, the transformer-based implementation is currently available in the `transformer` branch ([link](https://github.com/mengjian0502/Torch2Chip/tree/transformer))

## Usage and Requirements

The training and inference scripts `.sh` are all available in the `bash_files` folder of each branch. Execute the corresponding `.sh` file for training, fine-tuning, and inference. **For each script file, please specify your Python path before running the script.**

**CNN Training**

```bash
bash qtrain.sh
```

**CNN Inference**

```sh
bash inference.sh
```

**Transformer Training** 

```bash
bash vit_train.sh
```

**Transformer Fine-tuning**

```bash
bash vit_finetune.sh
```

**Transformer Inference**

```bash
bash vit_inference.sh
```

### Requirements

```markdown
python >= 3.7.4
pytorch >= 1.9.1
fxpmath = 0.4.5
```

