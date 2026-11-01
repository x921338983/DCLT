from collections import OrderedDict
from pathlib import Path
import hashlib
import torch
from torch import nn
import torchvision.models as tvm
from .adapters import GENERATED, LoRALinear, LoRAConv2d, SplitQKV, VisionSelfAttention
from .controller import LowRankPrompt, InstanceController, ParameterGenerator


class LabelMapping(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.register_buffer("indices", torch.zeros(classes, dtype=torch.long))
        self.register_buffer("ready", torch.tensor(False))

    def forward(self, source_logits):
        if not self.ready.item():
            raise RuntimeError("Initialize the FLM/ILM label mapping before training")
        return source_logits.index_select(-1, self.indices)


def _load_backbone(m, task):
    name, provider = m["backbone"], m["provider"]
    pretrained = m["pretrained"] and not m["checkpoint"]
    metadata = {"provider": provider, "pretrained": pretrained, "architecture": name}
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    source_head = None
    if task == "segmentation":
        weights = tvm.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tvm.segmentation.deeplabv3_resnet50(
            weights=None, weights_backbone=weights, num_classes=m["num_classes"], aux_loss=False)
        width = 2048
        metadata["weight_id"] = "ResNet50_Weights.IMAGENET1K_V1" if pretrained else "random"
    elif provider == "timm":
        try:
            import timm
        except ImportError as exc:
            raise ImportError("Install timm for the specified ImageNet-21K ViT. No fallback to other weights is performed.") from exc
        backbone = timm.create_model(name, pretrained=pretrained)
        source_head = backbone.get_classifier()
        width = backbone.num_features
        backbone.reset_classifier(0)
        pre_cfg = backbone.pretrained_cfg
        mean, std = pre_cfg.get("mean", mean), pre_cfg.get("std", std)
        if pre_cfg.get("input_size", (3, m["image_size"], m["image_size"]))[-1] != m["image_size"]:
            raise ValueError("Checkpoint input resolution differs from config")
        metadata["weight_id"] = name
        metadata["timm_version"] = timm.__version__
        metadata["pretrained_cfg"] = {k: pre_cfg.get(k) for k in ("hf_hub_id", "url", "num_classes", "mean", "std")}
    elif name in ("resnet18", "resnet50"):
        constructor = getattr(tvm, name)
        weights_type = tvm.ResNet18_Weights if name == "resnet18" else tvm.ResNet50_Weights
        weights = weights_type.IMAGENET1K_V1 if pretrained else None
        backbone = constructor(weights=weights)
        width, source_head = backbone.fc.in_features, backbone.fc
        backbone.fc = nn.Identity()
        metadata["weight_id"] = name + ":IMAGENET1K_V1" if pretrained else "random"
    elif name in ("vit_b_16", "vit_tiny_test"):
        if name == "vit_tiny_test":
            if pretrained:
                raise ValueError("Test ViT has no pretrained checkpoint")
            backbone = tvm.VisionTransformer(image_size=m["image_size"], patch_size=8,
                num_layers=2, num_heads=2, hidden_dim=32, mlp_dim=64, num_classes=10)
        else:
            weights = tvm.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
            if pretrained and m["image_size"] != 224:
                raise ValueError("Torchvision pretrained ViT requires 224px inputs")
            backbone = tvm.vit_b_16(weights=weights, image_size=m["image_size"])
        source_head, width = backbone.heads, backbone.hidden_dim
        backbone.heads = nn.Identity()
        metadata["weight_id"] = name + ":IMAGENET1K_V1" if pretrained else "random"
        metadata["pretraining_note"] = "Torchvision ViT ImageNet-1K pretraining"
    else:
        raise ValueError("Unsupported backbone/provider: " + name + "/" + provider)
    if m["checkpoint"]:
        checkpoint = Path(m["checkpoint"])
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = state.get("state_dict", state)
        # Load a plain backbone checkpoint before adaptation. Include source head for LP/FM consistency.
        if source_head is not None:
            if provider == "timm":
                backbone.reset_classifier(_head_out(source_head))
                backbone.load_state_dict(state, strict=True)
                source_head = backbone.get_classifier()
                backbone.reset_classifier(0)
            else:
                attr = "fc" if name.startswith("resnet") else "heads"
                setattr(backbone, attr, source_head)
                backbone.load_state_dict(state, strict=True)
                source_head = getattr(backbone, attr)
                setattr(backbone, attr, nn.Identity())
        else:
            backbone.load_state_dict(state, strict=True)
        metadata["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        metadata["checkpoint"] = str(checkpoint.resolve())
    return backbone, source_head, width, m["mean"] or mean, m["std"] or std, metadata


def _head_out(head):
    linears = [module for module in head.modules() if isinstance(module, nn.Linear)]
    if not linears:
        raise ValueError("Source checkpoint has no classifier")
    return linears[-1].out_features


def _set_module(root, path, module):
    parent_path, _, attr = path.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    setattr(parent, attr, module)


class DCLTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg, self.task, self.m = cfg, cfg["task"], cfg["model"]
        m, method = self.m, self.m["method"]
        self.backbone, source_head, width, mean, std, self.metadata = _load_backbone(m, self.task)
        self.register_buffer("input_mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("input_std", torch.tensor(std).view(1, 3, 1, 1))
        self.metadata["input_normalization"] = {"mean": list(mean), "std": list(std)}
        if self.task == "classification":
            if m["head"] == "lp":
                self.head = nn.Linear(width, m["num_classes"])
                self.source_head = None
            else:
                self.source_head = source_head
                source_classes = _head_out(source_head)
                if m["source_classes"] is not None and source_classes != m["source_classes"]:
                    raise ValueError("Configured source label count does not match actual checkpoint")
                self.metadata["source_classes"] = source_classes
                self.head = (nn.Linear(source_classes, m["num_classes"]) if m["head"] == "fm"
                             else LabelMapping(m["num_classes"]))
                self.source_head.requires_grad_(False)
        else:
            self.head = nn.Identity()
            self.source_head = None
        self.backbone.requires_grad_(method == "full_ft")
        if self.task == "segmentation":
            self.backbone.classifier.requires_grad_(True)
        self.prompt = (LowRankPrompt(m["image_size"], m["prompt_rank"])
                       if method in ("lor_vp", "dclt_i", "dclt_s") else None)
        self.cond_maps = nn.ModuleDict()
        self.target_names = []
        if method in ("lora", "dora", "condlora", "dclt_i", "dclt_s"):
            self._install_adapters()
        self.controller = None
        self.generator = None
        if method == "dclt_s":
            prompt_dim = 3 * m["prompt_rank"] * 2 * m["image_size"]
            self.controller = InstanceController(m["controller"], m["pool_size"], m["ln_eps"], prompt_dim)
            specs = [(key, self._b_shape(module)) for key, module in self.adapters().items()]
            decoder_seed = cfg["train"]["seed"] if m["decoder_seed"] in (None, "train_seed") else m["decoder_seed"]
            self.generator = ParameterGenerator(specs, self.controller.dim, m["latent_dim"], m["mapping"],
                                                m["decoder_init"], decoder_seed)
            self.metadata["decoder_seed"] = decoder_seed
        self._inference_prepared = False
        self.metadata["targets"] = self.target_names
        self.metadata["lora_scales"] = {key: adapter.scale for key, adapter in self.adapters().items()}

    @staticmethod
    def _b_shape(module):
        if isinstance(module, LoRALinear):
            return (module.base.out_features, module.rank)
        return (module.base.out_channels, module.rank, 1, 1)

    def _install_adapters(self):
        m, method = self.m, self.m["method"]
        candidates = []
        if m["backbone"].startswith("resnet") or self.task == "segmentation":
            for name, module in self.backbone.named_modules():
                if isinstance(module, nn.Conv2d) and name.split(".")[0] in ("layer3", "layer4"):
                    candidates.append((name, None))
            # DeepLab's IntermediateLayerGetter uses layer3/layer4 under its backbone child.
            if self.task == "segmentation":
                candidates = [(name, None) for name, module in self.backbone.named_modules()
                              if isinstance(module, nn.Conv2d) and name.startswith(("backbone.layer3.", "backbone.layer4."))]
        else:
            for block in m["vit_blocks"]:
                if m["provider"] == "timm":
                    prefix = f"blocks.{block}.attn"
                    attention = self.backbone.get_submodule(prefix)
                    attention.qkv = SplitQKV(attention.qkv)
                    candidates += [(prefix + ".qkv." + t.lower(), t) for t in ("Q", "K", "V")]
                    candidates.append((prefix + ".proj", "O"))
                else:
                    prefix = f"encoder.layers.encoder_layer_{block}.self_attention"
                    attention = VisionSelfAttention(self.backbone.get_submodule(prefix))
                    _set_module(self.backbone, prefix, attention)
                    candidates += [(prefix + "." + t.lower(), t) for t in ("Q", "K", "V")]
                    candidates.append((prefix + ".out_proj", "O"))
        if not candidates:
            raise ValueError("No target modules found")
        if method == "condlora":
            for name, projection in candidates:
                if projection not in self.cond_maps:
                    base = self.backbone.get_submodule(name)
                    map_a = nn.Linear(base.in_features, m["rank"], bias=False)
                    map_b = nn.Linear(base.out_features, m["rank"], bias=False)
                    nn.init.xavier_normal_(map_a.weight)
                    nn.init.zeros_(map_b.weight)
                    self.cond_maps[projection] = nn.ModuleList([map_a, map_b])
        for name, projection in candidates:
            base = self.backbone.get_submodule(name)
            options = dict(rank=m["rank"], alpha=m["alpha"], dynamic=method == "dclt_s", key=name, a_init=m["a_init"])
            if isinstance(base, nn.Linear):
                wrapped = LoRALinear(base, **options, dora=method == "dora",
                    cond_maps=self.cond_maps[projection] if method == "condlora" else None)
            else:
                wrapped = LoRAConv2d(base, **options)
            _set_module(self.backbone, name, wrapped)
            self.target_names.append(name)

    def adapters(self):
        return OrderedDict((name, self.backbone.get_submodule(name)) for name in self.target_names)

    def conditioning(self, image):
        prompted = self.prompt(image) if self.prompt is not None else image
        normalized = (prompted - self.input_mean) / self.input_std
        z, factors = None, None
        if self.controller is not None:
            plain = ((image - self.input_mean) / self.input_std
                     if self.controller.kind == "image_only" else None)
            z = self.controller(normalized, plain, self.prompt)
            factors = self.generator(z)
        return normalized, z, factors

    def forward(self, image, source_logits=False):
        normalized, _, factors = self.conditioning(image)
        token = GENERATED.set(factors)
        try:
            features = self.backbone(normalized)
        finally:
            GENERATED.reset(token)
        if self.task == "segmentation":
            return features["out"]
        if self.source_head is not None:
            features = self.source_head(features)
        if source_logits:
            if self.source_head is None:
                raise ValueError("LP does not expose source logits")
            return features
        return self.head(features)

    def train(self, mode=True):
        if mode and getattr(self, "_inference_prepared", False):
            raise RuntimeError("Merged model is inference-only; reload an unmerged checkpoint to train")
        super().train(mode)
        if mode and self.m["method"] != "full_ft" and self.m["bn_policy"] == "freeze_backbone":
            for name, module in self.backbone.named_modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm) and not name.startswith("classifier"):
                    module.eval()
        return self

    @torch.no_grad()
    def prepare_inference(self):
        self.eval()
        if self._inference_prepared:
            return self
        if self.prompt is not None:
            self.prompt.cache()
        if self.m["method"] != "dclt_s":
            for adapter in self.adapters().values():
                adapter.merge()
        self._inference_prepared = True
        return self


def parameter_report(model):
    def trainable(module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad) if module is not None else 0
    pA, pB = 0, 0
    for adapter in model.adapters().values():
        if hasattr(adapter, "A"):
            pA += adapter.A.numel()
        pB += torch.tensor(model._b_shape(adapter)).prod().item()
    frozen_decoder = 0
    if model.generator is not None and model.generator.mapping in ("shared", "layer_specific"):
        frozen_decoder = model.generator.U.numel()
    all_parameters = sum(p.numel() for p in model.parameters())
    output = {
        "trainable": trainable(model), "persistent_parameters": all_parameters,
        "frozen_decoder_coefficients": frozen_decoder,
        "stored_coefficients": all_parameters + frozen_decoder,
        "prompt_trainable": trainable(model.prompt), "pA": int(pA), "pB_generated_or_static": int(pB),
        "controller_trainable": trainable(model.controller), "mapping_trainable": trainable(model.generator),
        "conditional_mapping_trainable": trainable(model.cond_maps),
        "head_trainable": trainable(model.head) + (trainable(model.backbone.classifier) if model.task == "segmentation" else 0),
        "targets": model.target_names,
        "exclusions": "optimizer states, normalization statistics, runtime B activations; frozen U is included separately",
    }
    return output

