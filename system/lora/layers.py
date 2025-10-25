import torch
import math
import torch.nn.functional as F
from torch import nn
from typing import Optional, List

class LoRALayer():
    def __init__(self, rank: int, lora_alpha: int, 
                lora_dropout: float, merge_weights: bool):
        
        self.rank = rank
        self.lora_alpha = lora_alpha

        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x

        self.merge = False
        self.merge_weights = merge_weights

class LoRALinear(nn.Linear, LoRALayer):
    def __init__(self, in_features: int, out_features: int,
                rank: int = 0,
                lora_alpha: int = 1,
                lora_dropout: float = 0,
                fan_in_fan_out: bool = False,
                merge_weights: bool = True,
                **kwargs):
        
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, rank=rank, lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout, merge_weights=merge_weights)
        
        self.fan_in_fan_out = fan_in_fan_out

        if rank > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((rank, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, rank)))
            self.scaling = self.lora_alpha / self.rank

            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        
        self.reset_parameters()

        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)
    
    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.rank > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)
        
    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)

        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.rank > 0:
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = True

class ConvLoRA(nn.Module, LoRALayer):
    def __init__(self, conv_module, in_channels, out_channels, kernel_size, rank=0, lora_alpha=1, lora_dropout=0., merge_weights=True, **kwargs):
        super(ConvLoRA, self).__init__()
        self.conv = conv_module(in_channels, out_channels, kernel_size, **kwargs)
        for name, param in self.conv.named_parameters():
            self.register_parameter(name, param)

        LoRALayer.__init__(self, rank=rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        assert isinstance(kernel_size, int)

        if rank > 0:
            self.lora_A = nn.Parameter(
                self.conv.weight.new_zeros((rank * kernel_size, in_channels * kernel_size))
            )
            self.lora_B = nn.Parameter(
              self.conv.weight.new_zeros((out_channels//self.conv.groups*kernel_size, rank*kernel_size))
            )
            self.scaling = self.lora_alpha / self.rank

            # Freezing the pre-trained weight matrix
            self.conv.weight.requires_grad = False
        self.reset_parameters()
        self.merged = False

    def reset_parameters(self):
        self.conv.reset_parameters()
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
    
    def forward(self, x):
        if self.rank > 0 and not self.merged:
            return self.conv._conv_forward(
                x, 
                self.conv.weight + (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling,
                self.conv.bias
            )
        return self.conv(x)
    
    def train(self, mode=True):
        super(ConvLoRA, self).train(mode)
        
        if mode:
            if self.merge_weights and self.merged:
                if self.rank > 0:
                    # Make sure that the weights are not merged
                    self.conv.weight.data -= (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.rank > 0:
                    # Merge the weights and mark it
                    self.conv.weight.data += (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
                self.merged = True

class Conv2d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv2d, self).__init__(nn.Conv2d, *args, **kwargs)