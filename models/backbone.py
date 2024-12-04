
"""
Backbone modules. Resnet
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(nn.Module):
    """
    冻结方差均值等参数  
    n 是通道数 num_channels 
    在实现的时候，需要将4个量注册到buffer，以便阻止梯度反向传播而更新它们，同时又能够记录在模型的state_dict中。
    """
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.zeros(n))
    """
    当你在冻结模型时，num_batches_tracked 
    是用于训练过程中的统计量（如累积的均值和方差），在推理时是不需要的。
    而且，num_batches_tracked 的状态可能会影响迁移学习的效果，或者可能是来自于之前训练的模型，因此要移除它。
    """
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]
        super(FrozenBatchNorm2d, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
    
    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        # 计算 scale 和 bias
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return scale * x + bias 

"""
 对预训练中每一层 的mask进行插值 与特征大小一致  然后拼接 再存回去
"""
class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels
    # NestedTensor这个类的实例，其实质就是将图像张量和对应的mask封装到一起。
    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        # 遍历每一层的输出特征 x
        for name, x in xs.items():
            # 获取输入图像的掩码 形状是（batch_size, height, width)
            # m[None] 扩充到 （1， batch_size, height, width）
            m = tensor_list.mask
            assert m is not None
            # x.shape[-2:] --> (height, width)
            # 然后插值
            mask = F.interpolate(m[None].float(), size = x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out

class Backbone(BackboneBase):
     """ResNet backbone with frozen BatchNorm."""
     def __init__(self, name: str, train_backbone: bool, return_interm_layers: bool, dilation: bool):
         # getattr(object, name, default) getattr(torchvision.models, 'resnet50') 就相当于 torchvision.models.resnet50。
         # dilation: 是否使用膨胀卷积（dilated convolution），通常用于改变卷积步长而不减少特征图的尺寸。
         # replace_stride_with_dilation=[False, False, dilation]：
         # 该参数设置是否替换 ResNet 的步幅（stride）为膨胀卷积。
         # False 表示不使用膨胀卷积，只有在最后一个层（dilation 为 True 时）才会进行膨胀卷积。
         backbone = getattr(torchvision.models, name)(replace_stride_with_dilation=[False, False, dilation],
                                                        pretrained=is_main_process(), 
                                                        norm_layer=FrozenBatchNorm2d)
         num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
         super().__init__(backbone, train_backbone, num_channels, return_interm_layers)

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super.__init__(backbone, position_embedding)
    
    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list) # 调用self[0]，也就是backbone部分的forward
        out : List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            pos.append(self[1](x).to(x.tensors.dtype)) # 使用position_embedding对特征图x进行位置编码
        return out, pos  # 返回特征图列表和位置编码列表
        

def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
