from __future__ import annotations

import logging
from typing import List

import torch
import torch.nn as nn
import loralib as lora

try:
    from LoR_VP import LoR_VP
except ImportError:
    from trainer.LoR_VP import LoR_VP

logger = logging.getLogger(__name__)

class TwoLayerMapper(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int):
        super().__init__()
        self.proj1 = nn.Linear(in_dim, hidden, bias=False)
        self.proj2 = nn.Linear(hidden, out_dim, bias=False)
        nn.init.normal_(self.proj1.weight, std=0.02)
        nn.init.zeros_(self.proj2.weight)
    def forward(self, z):
        return self.proj2(self.proj1(z))
class OuterProductMapper(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, rank: int):
        super().__init__()
        self.rank = rank
        self.V = nn.Linear(in_dim, rank, bias=False)
        self.U = nn.Parameter(torch.empty(out_dim, rank))
        nn.init.normal_(self.V.weight, std=0.02)
        nn.init.normal_(self.U, std=0.02)
    def forward(self, z):
        codes = self.V(z)
        return torch.matmul(codes, self.U.t())
class LoR_VP_with_LoRA(nn.Module):
    def __init__(self, network: nn.Module, args, rank: int = 4, is_vit: bool = False):
        super().__init__()
        self.original_network = network
        self.rank = getattr(args, "lora_rank", rank)
        self.is_vit = is_vit
        self.args = args
        self.vit_target_layers= {8, 9, 10, 11}
        if self.rank > 0:
            self._replaced_module_ids: set[int] = set()
            self._replace_layers_recursive(self.original_network, prefix="")
            self._replaced_module_ids.clear()
        self.lora_targets: List[nn.Module] = []
        total_lora_params = 0
        for m in self.original_network.modules():
            if hasattr(m, "lora_A") and hasattr(m, "lora_B"):
                self.lora_targets.append(m)
                total_lora_params += m.lora_A.numel() + m.lora_B.numel()
                m.lora_A.requires_grad_(False)
                m.lora_B.requires_grad_(False)
            elif hasattr(m, "lora_A") and isinstance(m.lora_A, nn.ParameterDict):
                self.lora_targets.append(m)
                for k in m.lora_A.keys():
                    total_lora_params += m.lora_A[k].numel() + m.lora_B[k.replace("A","B")].numel()
                    m.lora_A[k].requires_grad_(False)
                    m.lora_B[k.replace("A","B")].requires_grad_(False)
        device = next(self.original_network.parameters()).device
        self.visual_prompt = LoR_VP(args, normalize=getattr(args, "normalize", None)).to(device)
        flat_c, flat_d = self.visual_prompt.get_low_rank()
        in_dim = flat_c.numel() + flat_d.numel()
        mapper_type = getattr(args, "mapper_type", "outer")      # {outer|mlp}
        if mapper_type == "mlp":
            hidden = getattr(args, "shared_hidden", 128)
            self.shared_mapper = TwoLayerMapper(in_dim, total_lora_params, hidden).to(device)
            logger.info("[SynLOR] mapper=MLP  hidden=%d", hidden)
        elif mapper_type == "outer":
            mapper_rank = getattr(args, "mapper_rank", 4)
            self.shared_mapper = OuterProductMapper(in_dim, total_lora_params, mapper_rank).to(device)
            logger.info("[SynLOR] mapper=OuterProduct  rank=%d", mapper_rank)
        else:
            raise ValueError("Unknown mapper_type: " + mapper_type)
    def _create_lora_linear_original(self, layer_to_replace, layer_full_name="UnknownLinear"):
        lora_layer = lora.Linear(
            in_features=layer_to_replace.in_features,
            out_features=layer_to_replace.out_features,
            r=self.rank,
            bias=layer_to_replace.bias is not None
        )
        if hasattr(lora_layer, 'lora_A') and isinstance(lora_layer.lora_A, nn.Parameter):
            torch.nn.init.zeros_(lora_layer.lora_A)
        elif hasattr(lora_layer, 'lora_A') and hasattr(lora_layer.lora_A, 'weight'):  # 兼容原始LORA.py中对Linear的特殊写法
            torch.nn.init.zeros_(lora_layer.lora_A.weight)
            logger.warning(f"Initialized lora_A.weight for {layer_full_name} due to submodule structure.")
        else:
            logger.error(
                f"Cannot initialize lora_A for {layer_full_name}: lora_A not found or not a Parameter/known submodule.")
        if hasattr(lora_layer, 'lora_B') and isinstance(lora_layer.lora_B, nn.Parameter):
            torch.nn.init.normal_(lora_layer.lora_B, mean=0, std=0.01)
        elif hasattr(lora_layer, 'lora_B') and hasattr(lora_layer.lora_B, 'weight'):  # 兼容原始LORA.py中对Linear的特殊写法
            torch.nn.init.normal_(lora_layer.lora_B.weight, mean=0, std=0.01)
            logger.warning(f"Initialized lora_B.weight for {layer_full_name} due to submodule structure.")
        else:
            logger.error(
                f"Cannot initialize lora_B for {layer_full_name}: lora_B not found or not a Parameter/known submodule.")
        with torch.no_grad():
            lora_layer.weight.copy_(layer_to_replace.weight)
            if layer_to_replace.bias is not None and lora_layer.bias is not None:
                lora_layer.bias.copy_(layer_to_replace.bias)
        self._replaced_module_ids.add(id(lora_layer))  # 标记这个新创建的LoRA层，避免在同一递归中被再次处理
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
        if hasattr(lora_layer, 'lora_A') and isinstance(lora_layer.lora_A, nn.Parameter):
            torch.nn.init.zeros_(lora_layer.lora_A)
        else:
            logger.error(f"Cannot initialize lora_A for {layer_full_name} (Conv2d).")
        if hasattr(lora_layer, 'lora_B') and isinstance(lora_layer.lora_B, nn.Parameter):
            torch.nn.init.normal_(lora_layer.lora_B, mean=0, std=0.01)
        else:
            logger.error(f"Cannot initialize lora_B for {layer_full_name} (Conv2d).")
        with torch.no_grad():
            lora_layer.weight.copy_(layer_to_replace.weight)
            if layer_to_replace.bias is not None and lora_layer.bias is not None:
                lora_layer.bias.copy_(layer_to_replace.bias)
        self._replaced_module_ids.add(id(lora_layer))
        return lora_layer
    def _create_lora_merged_linear_original(self, layer_to_replace, layer_full_name="UnknownMergedLinear",
                                            enable_lora_on_qkv=None):
        if enable_lora_on_qkv is None:
            enable_lora_on_qkv = [True, True, True]
        logger.debug(f"    Creating lora.MergedLinear for '{layer_full_name}' (Original Style Attempt)")
        lora_layer = lora.MergedLinear(
            in_features=layer_to_replace.in_features,
            out_features=layer_to_replace.out_features,
            r=self.rank,
            enable_lora=enable_lora_on_qkv,
            bias=layer_to_replace.bias is not None
        )
        for i in range(len(enable_lora_on_qkv)):
            if enable_lora_on_qkv[i]:
                key_A = f'lora_A_{i}'
                key_B = f'lora_B_{i}'
                if hasattr(lora_layer, 'lora_A') and key_A in lora_layer.lora_A and isinstance(lora_layer.lora_A[key_A],
                                                                                               nn.Parameter):
                    torch.nn.init.zeros_(lora_layer.lora_A[key_A])
                else:
                    logger.error(f"Cannot initialize {key_A} for {layer_full_name} (MergedLinear).")

                if hasattr(lora_layer, 'lora_B') and key_B in lora_layer.lora_B and isinstance(lora_layer.lora_B[key_B],
                                                                                               nn.Parameter):
                    torch.nn.init.normal_(lora_layer.lora_B[key_B], mean=0, std=0.01)
                else:
                    logger.error(f"Cannot initialize {key_B} for {layer_full_name} (MergedLinear).")
        with torch.no_grad():
            lora_layer.weight.copy_(layer_to_replace.weight)
            if layer_to_replace.bias is not None and lora_layer.bias is not None:
                lora_layer.bias.copy_(layer_to_replace.bias)
        self._replaced_module_ids.add(id(lora_layer))
        return lora_layer
    def _replace_vit_attention_module_projections(self, attention_module, attention_module_full_name):
        if id(attention_module) in self._replaced_module_ids:
            return False
        replaced_in_this_attn_module = False
        for submodule_name, submodule_instance in attention_module.named_children():
            if id(submodule_instance) in self._replaced_module_ids:
                continue
            full_submodule_name = f"{attention_module_full_name}.{submodule_name}"
            if isinstance(submodule_instance, nn.Linear):
                is_qkv_like = submodule_name == 'qkv'
                is_proj_like = submodule_name in ['proj', 'out']  # 'out' 来自您的日志
                is_qkv_separate = submodule_name in ['query', 'key', 'value']
                if is_qkv_like or is_proj_like or is_qkv_separate:
                    logger.info(f"    ViT Attn: Applying LoRA to nn.Linear '{full_submodule_name}'.")
                    new_lora_layer = None
                    if is_qkv_like and submodule_instance.out_features == 3 * submodule_instance.in_features:
                        new_lora_layer = self._create_lora_merged_linear_original(
                            submodule_instance, full_submodule_name, enable_lora_on_qkv=[True, True, True]
                        )
                    else:
                        new_lora_layer = self._create_lora_linear_original(submodule_instance, full_submodule_name)
                    if new_lora_layer:
                        setattr(attention_module, submodule_name, new_lora_layer)
                        replaced_in_this_attn_module = True

        if not replaced_in_this_attn_module:
            pass
        else:
            self._replaced_module_ids.add(id(attention_module))  # 标记父Attention模块已处理其内部
        return replaced_in_this_attn_module
    def _replace_layers_recursive(self, current_module, prefix):
        if self.rank <= 0:
            return False
        if id(current_module) in self._replaced_module_ids:
            return False

        has_replaced_child = False

        for name, child_module_instance in current_module.named_children():
            if id(child_module_instance) in self._replaced_module_ids:
                continue

            full_name = f"{prefix}.{name}" if prefix else name

            if self.is_vit:
                is_attention_block = name == 'attn' or "attention" in name.lower()
                is_mha_instance = isinstance(child_module_instance, nn.MultiheadAttention)
                if is_attention_block or is_mha_instance:
                    layer_index = -1
                    try:
                        parts = full_name.split('.')
                        for part in parts:
                            if part.isdigit():
                                layer_index = int(part)
                                break
                    except:
                        pass

                    if layer_index in self.vit_target_layers:
                        logger.info(f"  ViT: Applying LoRA to target layer {layer_index} -> module '{full_name}'")
                        if self._replace_vit_attention_module_projections(child_module_instance, full_name):
                            has_replaced_child = True
                    continue
                if self._replace_layers_recursive(child_module_instance, full_name):
                    has_replaced_child = True

            else:
                if isinstance(child_module_instance, nn.Conv2d):
                    apply_lora_to_this_conv = False
                    if prefix.startswith("layer4") or prefix.startswith("layer3"):
                        apply_lora_to_this_conv = True

                    if apply_lora_to_this_conv:
                        new_lora_conv = self._create_lora_conv2d_original(child_module_instance, full_name)
                        setattr(current_module, name, new_lora_conv)
                        has_replaced_child = True
                    elif list(child_module_instance.children()):
                        if self._replace_layers_recursive(child_module_instance, full_name):
                            has_replaced_child = True

                elif isinstance(child_module_instance, nn.Linear):
                    apply_lora_to_this_linear = False
                    if full_name == 'fc':
                        pass
                    if apply_lora_to_this_linear:
                        new_lora_linear = self._create_lora_linear_original(child_module_instance, full_name)
                        setattr(current_module, name, new_lora_linear)
                        has_replaced_child = True
                    elif list(child_module_instance.children()):
                        if self._replace_layers_recursive(child_module_instance, full_name):
                            has_replaced_child = True
                else:
                    if self._replace_layers_recursive(child_module_instance, full_name):
                        has_replaced_child = True

        return has_replaced_child
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        flat_c, flat_d = self.visual_prompt.get_low_rank()
        z = torch.cat([flat_c.flatten(), flat_d.flatten()]).unsqueeze(0)  # (1,in_dim)
        theta = self.shared_mapper(z).squeeze(0)                          # (total,)

        ptr = 0
        for m in self.lora_targets:
            if isinstance(m.lora_A, nn.Parameter):
                nA, nB = m.lora_A.numel(), m.lora_B.numel()
                m.lora_A.data = theta[ptr:ptr+nA].view_as(m.lora_A); ptr += nA
                m.lora_B.data = theta[ptr:ptr+nB].view_as(m.lora_B); ptr += nB
            else:  # ParameterDict (MergedLinear)
                for i in range(3):
                    kA, kB = f"lora_A_{i}", f"lora_B_{i}"
                    if kA in m.lora_A:
                        nA, nB = m.lora_A[kA].numel(), m.lora_B[kB].numel()
                        m.lora_A[kA].data = theta[ptr:ptr+nA].view_as(m.lora_A[kA]); ptr += nA
                        m.lora_B[kB].data = theta[ptr:ptr+nB].view_as(m.lora_B[kB]); ptr += nB

        return self.original_network(self.visual_prompt(x))
