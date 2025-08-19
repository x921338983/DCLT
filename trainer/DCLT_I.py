import torch
import torch.nn as nn
import loralib as lora
import re
import logging
import math

try:
    from visual_prompt import LoR_VP
except ImportError:
    from trainer.LoR_VP import LoR_VP

logger = logging.getLogger(__name__)


class LoR_VP_with_LoRA(nn.Module):
    def __init__(self, network, args, rank=4, is_vit=False, lora_target_blocks=None):
        super(LoR_VP_with_LoRA, self).__init__()
        self.original_network = network
        self.rank = args.lora_rank
        self.is_vit = is_vit
        self.args = args
        self.lora_target_blocks = lora_target_blocks
        if self.rank > 0:
            self.lora_applied = True
            self._apply_lora()
            lora.mark_only_lora_as_trainable(self.original_network, bias='lora_only')
            lora_param_count = 0
            for name, param in self.original_network.named_parameters():
                if param.requires_grad and 'lora_' in name:
                    lora_param_count += param.numel()
        else:
            self.lora_applied = False
            logger.info("LoR_VP_with_LoRA: rank <= 0, LoRA not applied.")
        device = next(self.original_network.parameters()).device
        self.visual_prompt = LoR_VP(self.args, normalize=getattr(self.args, 'normalize', None)).to(device)
        vp_param_count = 0
        for param_vp in self.visual_prompt.parameters():
            param_vp.requires_grad = True
            vp_param_count += param_vp.numel()

    def _apply_lora(self):
        to_replace = []
        for name, module in self.original_network.named_modules():
            if self.is_vit:
                parent_name, child_name = name.rsplit('.', 1) if '.' in name else ('', name)
                if parent_name.endswith(('.attn', '.attention')):
                    if isinstance(module, nn.Linear) and child_name in ['query', 'key', 'value', 'out', 'qkv', 'proj']:
                        match = re.search(r'\.(?:blocks|layers?|resblocks)\.(\d+)', parent_name)
                        if match:
                            block_idx = int(match.group(1))
                            if self.lora_target_blocks is None or block_idx in self.lora_target_blocks:
                                parent_module = self.original_network.get_submodule(parent_name)
                                to_replace.append((parent_module, child_name, module, name))
                        else:
                            parent_module = self.original_network.get_submodule(parent_name)
                            to_replace.append((parent_module, child_name, module, name))
            else:
                if isinstance(module, nn.Conv2d) and name.startswith('layer4'):
                    parent_name, child_name = name.rsplit('.', 1)
                    parent_module = self.original_network.get_submodule(parent_name)
                    to_replace.append((parent_module, child_name, module, name))

        for parent_module, child_name, old_module, full_name in to_replace:
            new_lora_layer = None
            if isinstance(old_module, nn.Linear):
                if child_name == 'qkv' and old_module.out_features == 3 * old_module.in_features:
                    new_lora_layer = self._create_lora_merged_linear_original(old_module, full_name)
                else:
                    new_lora_layer = self._create_lora_linear_original(old_module, full_name)

            elif isinstance(old_module, nn.Conv2d):
                new_lora_layer = self._create_lora_conv2d_original(old_module, full_name)

            if new_lora_layer:
                setattr(parent_module, child_name, new_lora_layer)

    def _create_lora_linear_original(self, layer_to_replace, layer_full_name="UnknownLinear"):
        lora_layer = lora.Linear(
            in_features=layer_to_replace.in_features,
            out_features=layer_to_replace.out_features,
            r=self.rank,
            bias=layer_to_replace.bias is not None
        )
        if hasattr(lora_layer, 'lora_A'):
            torch.nn.init.zeros_(lora_layer.lora_A)
        if hasattr(lora_layer, 'lora_B'):
            torch.nn.init.normal_(lora_layer.lora_B, mean=0, std=0.01)

        with torch.no_grad():
            lora_layer.weight.copy_(layer_to_replace.weight)
            if layer_to_replace.bias is not None and lora_layer.bias is not None:
                lora_layer.bias.copy_(layer_to_replace.bias)
        return lora_layer

    def _create_lora_conv2d_original(self, layer_to_replace, layer_full_name="UnknownConv2d"):
        def get_single_param(param_val):
            return param_val[0] if isinstance(param_val, tuple) else param_val

        lora_layer = lora.Conv2d(
            in_channels=layer_to_replace.in_channels,
            out_channels=layer_to_replace.out_channels,
            kernel_size=get_single_param(layer_to_replace.kernel_size),
            stride=get_single_param(layer_to_replace.stride),
            padding=get_single_param(layer_to_replace.padding),
            dilation=get_single_param(layer_to_replace.dilation),
            groups=layer_to_replace.groups,
            bias=layer_to_replace.bias is not None,
            r=self.rank
        )
        if hasattr(lora_layer, 'lora_A'):
            torch.nn.init.zeros_(lora_layer.lora_A)
        if hasattr(lora_layer, 'lora_B'):
            torch.nn.init.normal_(lora_layer.lora_B, mean=0, std=0.01)

        with torch.no_grad():
            lora_layer.weight.copy_(layer_to_replace.weight)
            if layer_to_replace.bias is not None and lora_layer.bias is not None:
                lora_layer.bias.copy_(layer_to_replace.bias)
        return lora_layer

    def _create_lora_merged_linear_original(self, layer_to_replace, layer_full_name="UnknownMergedLinear"):
        lora_layer = lora.MergedLinear(
            in_features=layer_to_replace.in_features,
            out_features=layer_to_replace.out_features,
            r=self.rank,
            enable_lora=[True, True, True],
            bias=layer_to_replace.bias is not None
        )
        for i in range(3):
            key_A = f'lora_A_{i}'
            key_B = f'lora_B_{i}'
            if hasattr(lora_layer, 'lora_A') and key_A in lora_layer.lora_A:
                torch.nn.init.zeros_(lora_layer.lora_A[key_A])
            if hasattr(lora_layer, 'lora_B') and key_B in lora_layer.lora_B:
                torch.nn.init.normal_(lora_layer.lora_B[key_B], mean=0, std=0.01)

        with torch.no_grad():
            lora_layer.weight.copy_(layer_to_replace.weight)
            if layer_to_replace.bias is not None and lora_layer.bias is not None:
                lora_layer.bias.copy_(layer_to_replace.bias)
        return lora_layer

    def forward(self, x):
        x = self.visual_prompt(x)
        x = self.original_network(x)
        return x