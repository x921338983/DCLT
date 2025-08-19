import torch
from torch.nn.functional import one_hot
import torch.nn.functional as F
import torch.nn as nn


def label_mapping_base(logits, mapping_sequence):
    if not isinstance(mapping_sequence, torch.Tensor):
        mapping_sequence = torch.as_tensor(mapping_sequence, dtype=torch.long, device=logits.device)
    elif mapping_sequence.device != logits.device:
        mapping_sequence = mapping_sequence.to(logits.device)
    if not isinstance(mapping_sequence, torch.LongTensor):
        mapping_sequence = mapping_sequence.long()

    if logits.size(1) == 0:
        return logits

    if mapping_sequence.numel() == 0:
        return logits

    if mapping_sequence.max().item() >= logits.size(1) or mapping_sequence.min().item() < 0:
        return logits

    modified_logits = logits[:, mapping_sequence]
    return modified_logits


def get_dist_matrix(fx, y):
    device = fx.device
    y = y.to(device)

    num_classes_in_fx = fx.size(-1)
    if num_classes_in_fx == 0:
        num_unique_y = len(y.unique()) if y.numel() > 0 else 0
        return torch.zeros(0, num_unique_y, device=device, dtype=fx.dtype)

    if y.numel() == 0:
        return torch.zeros(num_classes_in_fx, 0, device=device, dtype=fx.dtype)

    try:
        argmax_fx = torch.argmax(fx, dim=-1)
        if argmax_fx.max() >= num_classes_in_fx:
            argmax_fx = torch.clamp(argmax_fx, 0, num_classes_in_fx - 1)
        fx_one_hot = one_hot(argmax_fx, num_classes=num_classes_in_fx)
    except RuntimeError:
        num_unique_y = len(y.unique())
        return torch.zeros(num_classes_in_fx, num_unique_y, device=device, dtype=fx.dtype)

    unique_y_values = y.unique()
    if unique_y_values.numel() == 0:
        return torch.zeros(num_classes_in_fx, 0, device=device, dtype=fx.dtype)

    dist_matrix_list = []
    for i_val in unique_y_values:
        i = i_val.item()
        dist_matrix_list.append(fx_one_hot[y == i].sum(0).unsqueeze(1))

    if not dist_matrix_list:
        return torch.zeros(num_classes_in_fx, 0, device=device, dtype=fx.dtype)

    try:
        dist_matrix = torch.cat(dist_matrix_list, dim=1)
    except RuntimeError:
        return torch.zeros(num_classes_in_fx, 0, device=device, dtype=fx.dtype)

    return dist_matrix


def predictive_distribution_based_multi_label_mapping(dist_matrix, mlm_num: int):
    if not isinstance(dist_matrix, torch.Tensor) or dist_matrix.ndim != 2:
        return torch.empty(0, 0, dtype=torch.int,
                           device=dist_matrix.device if isinstance(dist_matrix, torch.Tensor) else 'cpu')

    num_source_labels, num_target_labels = dist_matrix.shape

    if num_source_labels == 0 or num_target_labels == 0:
        return torch.zeros_like(dist_matrix, dtype=torch.int)

    if mlm_num <= 0:
        mlm_num = 1

    mapping_matrix = torch.zeros_like(dist_matrix, dtype=torch.int)
    dist_matrix_copy = dist_matrix.clone()

    num_iterations = mlm_num * num_target_labels
    dist_matrix_flat = dist_matrix_copy.flatten()

    for _ in range(num_iterations):
        if dist_matrix_flat.numel() == 0 or (dist_matrix_flat <= -1).all():
            break

        loc_flat = dist_matrix_flat.argmax().item()
        row_idx = loc_flat // num_target_labels
        col_idx = loc_flat % num_target_labels

        if not (0 <= row_idx < num_source_labels and 0 <= col_idx < num_target_labels):
            continue

        if mapping_matrix[row_idx, col_idx] == 1:
            dist_matrix_flat[loc_flat] = -float('inf')
            continue

        if mapping_matrix[:, col_idx].sum() >= mlm_num:
            for r_update in range(num_source_labels):
                flat_idx_update = r_update * num_target_labels + col_idx
                if 0 <= flat_idx_update < dist_matrix_flat.numel():
                    dist_matrix_flat[flat_idx_update] = -1
            continue

        mapping_matrix[row_idx, col_idx] = 1

        start_idx_row = row_idx * num_target_labels
        end_idx_row = start_idx_row + num_target_labels
        if 0 <= start_idx_row < end_idx_row <= dist_matrix_flat.numel():
            dist_matrix_flat[start_idx_row:end_idx_row] = -1

        if mapping_matrix[:, col_idx].sum() >= mlm_num:
            for r_update in range(num_source_labels):
                flat_idx_update = r_update * num_target_labels + col_idx
                if 0 <= flat_idx_update < dist_matrix_flat.numel():
                    dist_matrix_flat[flat_idx_update] = -1

    return mapping_matrix


def generate_label_mapping_by_frequency(network, data_loader, args_obj):
    device = next(network.parameters()).device

    original_mode_is_train = network.training
    if original_mode_is_train:
        network.eval()

    fx0s = []
    ys = []

    with torch.no_grad():
        for i, batch_data in enumerate(data_loader):
            if not (isinstance(batch_data, (list, tuple)) and len(batch_data) >= 2):
                if original_mode_is_train:
                    network.train()
                return None

            x, y = batch_data[0].to(device), batch_data[1].to(device)
            try:
                fx0 = network(x)
            except Exception:
                if original_mode_is_train:
                    network.train()
                return None

            if isinstance(fx0, tuple):
                fx0 = fx0[0]
            fx0s.append(fx0)
            ys.append(y)

    if not fx0s or not ys:
        if original_mode_is_train:
            network.train()
        return None

    try:
        fx0s_cat = torch.cat(fx0s).cpu().float()
        ys_cat = torch.cat(ys).cpu().int()
    except Exception:
        if original_mode_is_train:
            network.train()
        return None

    if ys_cat.size(0) != fx0s_cat.size(0):
        if fx0s_cat.size(0) > 0 and ys_cat.size(0) > 0 and fx0s_cat.size(0) % ys_cat.size(0) == 0:
            ys_cat = ys_cat.repeat(int(fx0s_cat.size(0) / ys_cat.size(0)))
        else:
            if original_mode_is_train:
                network.train()
            return None

    dist_matrix = get_dist_matrix(fx0s_cat, ys_cat)
    if dist_matrix.numel() == 0 or dist_matrix.size(0) == 0:
        if original_mode_is_train:
            network.train()
        return None

    mapping_num = getattr(args_obj, 'mapping_freq', 1)
    pairs_matrix = predictive_distribution_based_multi_label_mapping(dist_matrix, mapping_num)
    if pairs_matrix.numel() == 0 and dist_matrix.numel() != 0:
        if original_mode_is_train:
            network.train()
        return None
    elif pairs_matrix.numel() == 0 and dist_matrix.numel() == 0:
        if original_mode_is_train:
            network.train()
        return None

    pairs = torch.nonzero(pairs_matrix)
    num_target_classes_from_args = getattr(args_obj, 'class_cnt', 0)

    if pairs.numel() == 0:
        mapping_sequence = None
    else:
        try:
            mapping_num = getattr(args_obj, 'mapping_freq', 1)
            if mapping_num == 1:
                num_actual_target_classes = dist_matrix.size(1)
                temp_map = {}
                for p_idx in range(pairs.size(0)):
                    src_idx = pairs[p_idx, 0].item()
                    tgt_idx_unique = pairs[p_idx, 1].item()
                    if tgt_idx_unique not in temp_map:
                        temp_map[tgt_idx_unique] = src_idx

                mapping_sequence_list = [temp_map[i] for i in sorted(temp_map.keys()) if i < num_actual_target_classes]

                if len(mapping_sequence_list) != num_actual_target_classes:
                    if 0 < num_target_classes_from_args <= fx0s_cat.size(1):
                        mapping_sequence = torch.arange(num_target_classes_from_args, device=device)
                    else:
                        mapping_sequence = None
                elif not mapping_sequence_list:
                    mapping_sequence = None
                else:
                    mapping_sequence = torch.as_tensor(mapping_sequence_list, dtype=torch.long, device=device)
            else:
                mapping_sequence = None
        except IndexError:
            mapping_sequence = None

    if original_mode_is_train:
        network.train()

    return mapping_sequence


class CustomNetwork(torch.nn.Module):
    def __init__(self, network_input_arg, visual_prompt_external_arg,
                 label_mapping_func_arg, mapping_sequence_tensor_arg, args_obj):
        super(CustomNetwork, self).__init__()

        self.args = args_obj
        self.device = args_obj.device
        self.method = args_obj.prompt_method
        self.downstream_mapping = args_obj.downstream_mapping
        self.label_mapping_func = label_mapping_func_arg
        self.mapping_sequence_tensor = mapping_sequence_tensor_arg

        self.network = network_input_arg
        self.visual_prompt = None
        self.fm_layer = None

        for param in self.network.parameters():
            param.requires_grad = False

        if visual_prompt_external_arg is not None and isinstance(visual_prompt_external_arg, nn.Module):
            for param in visual_prompt_external_arg.parameters():
                param.requires_grad = False

        model_to_operate_on = self.network
        if hasattr(self.network, 'original_network'):
            model_to_operate_on = self.network.original_network

        if self.method == 'lor_vp':
            if visual_prompt_external_arg is not None and isinstance(visual_prompt_external_arg, nn.Module):
                self.visual_prompt = visual_prompt_external_arg
                for param in self.visual_prompt.parameters():
                    param.requires_grad = True

        elif self.method == 'lor_vp_lora':
            lora_wrapper = self.network
            if hasattr(lora_wrapper, 'visual_prompt') and isinstance(lora_wrapper.visual_prompt, nn.Module):
                for param in lora_wrapper.visual_prompt.parameters():
                    param.requires_grad = True
            if hasattr(lora_wrapper, 'shared_mapper'):
                for param in lora_wrapper.shared_mapper.parameters():
                    param.requires_grad = True
                lora_wrapper.shared_mapper.U.requires_grad = False

        if self.downstream_mapping == 'lp':
            self._setup_linear_probe_head(model_to_operate_on)
        elif self.downstream_mapping == 'fm':
            self._setup_fm_layer(model_to_operate_on)
        elif self.downstream_mapping in ['ilm', 'flm', 'origin']:
            pass
        else:
            pass

    def _get_feature_output_dim(self, model_to_inspect):
        head_attributes = ['head', 'fc', 'classifier']
        for attr_name in head_attributes:
            if hasattr(model_to_inspect, attr_name):
                potential_head = getattr(model_to_inspect, attr_name)
                if isinstance(potential_head, nn.Linear):
                    return potential_head.in_features

        is_vit_model = False
        if hasattr(model_to_inspect, 'default_cfg') and isinstance(model_to_inspect.default_cfg, dict) and \
                'vit' in model_to_inspect.default_cfg.get('architecture', '').lower():
            is_vit_model = True
        elif 'vit' in str(type(model_to_inspect)).lower():
            is_vit_model = True

        if is_vit_model:
            if hasattr(model_to_inspect, 'hidden_size'):
                return model_to_inspect.hidden_size
            if hasattr(model_to_inspect, 'embed_dim'):
                return model_to_inspect.embed_dim

        return None

    def _setup_fm_layer(self, model_to_modify):
        num_classes = getattr(self.args, 'class_cnt', None)
        if num_classes is None:
            return False

        in_features = self._get_feature_output_dim(model_to_modify)
        if in_features is None:
            return False

        self.fm_layer = nn.Linear(in_features, num_classes).to(self.device)
        for param in self.fm_layer.parameters():
            param.requires_grad = True

        replaced_original_head = False
        for head_attr_name in ['head', 'fc', 'classifier']:
            if hasattr(model_to_modify, head_attr_name):
                original_head = getattr(model_to_modify, head_attr_name)
                if isinstance(original_head, nn.Linear):
                    setattr(model_to_modify, head_attr_name, nn.Identity().to(self.device))
                    replaced_original_head = True
                    break
        return True

    def _setup_linear_probe_head(self, model_to_modify):
        num_classes = getattr(self.args, 'class_cnt', None)
        if num_classes is None:
            return False

        head_found_and_replaced = False
        possible_head_names = ['head', 'fc', 'classifier']

        for attr_name in possible_head_names:
            if hasattr(model_to_modify, attr_name):
                original_head_module = getattr(model_to_modify, attr_name)
                if isinstance(original_head_module, nn.Linear):
                    in_features = original_head_module.in_features
                    new_head = nn.Linear(in_features, num_classes).to(self.device)
                    setattr(model_to_modify, attr_name, new_head)
                    for param in new_head.parameters():
                        param.requires_grad = True
                    head_found_and_replaced = True
                    break
                elif isinstance(original_head_module, nn.Identity):
                    pass
                else:
                    pass

        if not head_found_and_replaced:
            in_features = self._get_feature_output_dim(model_to_modify)
            if in_features is not None:
                new_head_name = 'head'
                if hasattr(model_to_modify, new_head_name) and isinstance(getattr(model_to_modify, new_head_name),
                                                                          nn.Identity) and self.downstream_mapping == 'fm':
                    return False
                new_head = nn.Linear(in_features, num_classes).to(self.device)
                setattr(model_to_modify, new_head_name, new_head)
                for param in new_head.parameters():
                    param.requires_grad = True
                head_found_and_replaced = True
            else:
                return False
        return head_found_and_replaced

    def get_tunable_params(self):
        trainable_params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
        return trainable_params

    def forward(self, x):
        input_to_main_network = x
        if self.method == 'lor_vp' and self.visual_prompt is not None:
            input_to_main_network = self.visual_prompt(x)

        current_output = self.network(input_to_main_network)

        if isinstance(current_output, tuple):
            current_output = current_output[0]

        if self.downstream_mapping == 'fm':
            if self.fm_layer is not None:
                final_logits = self.fm_layer(current_output)
            else:
                final_logits = current_output
        else:
            final_logits = current_output

        output_log_probs = F.log_softmax(final_logits, dim=-1)

        if self.label_mapping_func is not None and self.mapping_sequence_tensor is not None and \
                self.downstream_mapping in ('ilm', 'flm', 'origin'):
            output_log_probs = self.label_mapping_func(output_log_probs)

        return output_log_probs
