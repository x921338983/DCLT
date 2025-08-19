import logging
import traceback
import torch
import torch.distributed as dist
import sys
import os
import time
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import json
from functools import partial
from timm.utils import accuracy, AverageMeter
# from DCLT_I import LoR_VP_with_LoRA
from DCLT_S import LoR_VP_with_LoRA
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.model_and_data import get_torch_dataset, get_model, get_specified_model
from utils.exp_utils import set_seed, get_optimizer, try_cuda, AverageMeter, reduce_tensor
# from utils.my_utils_mapping import generate_label_mapping_by_frequency, label_mapping_base, CustomNetwork
from utils.my_utils_mapping_share import generate_label_mapping_by_frequency, label_mapping_base, CustomNetwork
from LoR_VP import LoR_VP
import copy
from thop import profile
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Image Classification')
    parser.add_argument('--network', type=str, default='vit_b_16',
                        choices=['resnet18', 'resnet50', 'mobilenet', 'vit_b_16'])
    parser.add_argument('--pretrain_path', type=str, default='../models/imagenet21k_ViT-B_16.npz',
                        help='pretrained model directory')
    parser.add_argument('--head_init', type=str, default='pretrain',
                        choices=['uniform', 'normal', 'xavier_uniform', 'kaiming_normal', 'zero', 'default', 'pretain'])
    parser.add_argument('--randomcrop', type=int, default=1, choices=[1, 0])
    parser.add_argument("--shuffle", default=1, type=int, help="whether shuffle the train dataset")
    parser.add_argument("--is_observe", default=0, type=int, help="whether observe images, vp, and features")
    parser.add_argument('--input_size', type=int, default=224, help='image size before prompt')
    parser.add_argument('--output_size', type=int, default=224, help='image size before prompt')
    parser.add_argument('--downstream_mapping', type=str, default='lp', choices=['origin', 'fm', 'ilm', 'flm', 'lp'])
    parser.add_argument('--mapping_freq', type=int, default=1, help='frequency of label mapping')
    parser.add_argument('--prompt_method', type=str, default='lor_vp', choices=['lor_vp', 'dclt'])
    parser.add_argument('--bar_width', type=int, default=4)
    parser.add_argument('--bar_height', type=int, default=224)
    parser.add_argument('--init_method', type=str, default='zero,normal')
    parser.add_argument('--dataset', type=str, default='tiny_imagenet',
                        choices=['cifar10', 'cifar100', 'tiny_imagenet', 'imagenet'])
    parser.add_argument('--datadir', type=str, default='../data', help='data directory')
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--eval_batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--optimizer', type=str, default='adamW')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'linear'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--eval_frequency', type=int, default=5)
    parser.add_argument('--warmup_ratio', type=float, default=0)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="Max gradient norm.")
    parser.add_argument('--fp16', default=True, type=bool,
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--global_step', type=int, default=0)
    parser.add_argument('--save_path', type=str, default='ckpt/',
                        help='path to save the final model')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--specified_path', type=str, default='')
    parser.add_argument('--lora_rank', type=str, default='4')
    args = parser.parse_args()

    if isinstance(args.lora_rank, str):
        args.lora_rank = int(args.lora_rank)

    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])

    return args


def evaluate(network, loader, args):
    network.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    start = time.time()
    for batch in loader:
        with torch.no_grad():
            inputs, targets = try_cuda(*batch[:2])
            with torch.cuda.amp.autocast():
                pred = network(inputs)
            loss = F.cross_entropy(pred, targets)
            acc = accuracy(pred, targets, topk=(1,))[0]
            if args.local_rank != -1 and dist.is_initialized():
                acc = reduce_tensor(acc.clone())
                loss = reduce_tensor(loss.clone())
            loss_meter.update(loss.item(), inputs.size(0))
            acc_meter.update(acc.item(), inputs.size(0))
    end = time.time()
    eval_elapsed = end - start
    eval_idx_log = args.epoch // args.eval_frequency if args.eval_frequency > 0 else 0
    log_str = '| Eval {:3d} at epoch {:>8d} | time: {:5.2f}s | acc: {:5.2f} | loss: {:5.2f} |'.format(
        eval_idx_log, args.epoch, eval_elapsed, acc_meter.avg, loss_meter.avg)
    logger.info(log_str)
    return acc_meter.avg


def train(network, optimizer, scheduler, loader, args):
    scaler = torch.cuda.amp.GradScaler()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    start = time.time()
    optimizer.zero_grad()
    for idx, (inputs, targets) in enumerate(loader):
        network.train()
        inputs, targets = try_cuda(inputs, targets)
        with torch.cuda.amp.autocast():
            pred = network(inputs)
        loss = F.cross_entropy(pred, targets) / args.gradient_accumulation_steps
        acc = accuracy(pred, targets, topk=(1,))[0]
        loss_meter.update(loss.item() * args.gradient_accumulation_steps, inputs.size(0))
        acc_meter.update(acc.item(), inputs.size(0))
        if args.fp16:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (idx + 1) % args.gradient_accumulation_steps == 0:
            if args.fp16:
                scaler.unscale_(optimizer)
            if args.gradient_accumulation_steps > 1:
                params_to_clip = []
                if isinstance(network, DDP) and hasattr(network.module, 'get_tunable_params'):
                    params_to_clip = network.module.get_tunable_params()
                elif hasattr(network, 'get_tunable_params'):
                    params_to_clip = network.get_tunable_params()

                if params_to_clip:
                    torch.nn.utils.clip_grad_norm_(params_to_clip, args.max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()
            args.global_step += 1

    end = time.time()
    train_elapsed = end - start
    current_lr = scheduler.get_last_lr()[0] if scheduler is not None and hasattr(scheduler, 'get_last_lr') else \
        optimizer.param_groups[0]['lr']
    log_str = '| epoch {:3d} | time: {:5.2f} | lr: {:.3e} | acc: {:5.2f} | loss: {:5.2f} |'.format(
        args.epoch, train_elapsed, current_lr, acc_meter.avg, loss_meter.avg)
    logger.info(log_str)
    return loss_meter.avg


def apply_label_mapping(current_network_obj,
                        visual_prompt_obj_for_customnetwork,
                        train_loader_for_mapping,
                        args_obj,
                        use_pure_base_for_lora_ilm_flm_mapping=False,
                        pure_base_model_for_mapping=None):
    label_mapping_func_to_pass = None
    mapping_sequence_tensor_to_pass = None
    network_for_initial_mapping_gen = current_network_obj

    if args_obj.prompt_method == 'dclt' and \
            args_obj.downstream_mapping in ['ilm', 'flm'] and \
            use_pure_base_for_lora_ilm_flm_mapping and \
            pure_base_model_for_mapping is not None:
        network_for_initial_mapping_gen = pure_base_model_for_mapping
    elif hasattr(current_network_obj, 'original_network'):
        network_for_initial_mapping_gen = current_network_obj.original_network

    should_generate_mapping_for_ilm_flm = False
    if args_obj.downstream_mapping in ['ilm', 'flm']:
        if args_obj.epoch == 0:
            should_generate_mapping_for_ilm_flm = True
        elif args_obj.epoch > 0 and args_obj.epoch % args_obj.mapping_freq == 0 and isinstance(current_network_obj,
                                                                                               CustomNetwork):
            should_generate_mapping_for_ilm_flm = True

    if should_generate_mapping_for_ilm_flm:
        original_training_state = network_for_initial_mapping_gen.training
        network_for_initial_mapping_gen.eval()
        mapping_sequence_tensor_to_pass = generate_label_mapping_by_frequency(network_for_initial_mapping_gen,
                                                                              train_loader_for_mapping,
                                                                              args_obj)
        if original_training_state:
            network_for_initial_mapping_gen.train()
        if mapping_sequence_tensor_to_pass is not None:
            label_mapping_func_to_pass = partial(label_mapping_base, mapping_sequence=mapping_sequence_tensor_to_pass)
        else:
            logger.error("  generate_label_mapping_by_frequency 返回了 None！")
            label_mapping_func_to_pass = None
            mapping_sequence_tensor_to_pass = None

    elif args_obj.downstream_mapping == 'origin':
        if hasattr(args_obj, 'class_cnt') and args_obj.class_cnt > 0:
            mapping_sequence_tensor_to_pass = torch.arange(args_obj.class_cnt, device=args_obj.device)
            label_mapping_func_to_pass = partial(label_mapping_base, mapping_sequence=mapping_sequence_tensor_to_pass)
            logger.info(f'  Mapping sequence (origin): {mapping_sequence_tensor_to_pass.tolist()}')
        else:
            logger.error("  无法生成 'origin' 映射序列，args.class_cnt 未定义或不合法。")

    if not isinstance(current_network_obj, CustomNetwork) or args_obj.epoch == 0:
        logger.info(f"apply_label_mapping: Creating new CustomNetwork instance at epoch {args_obj.epoch}.")
        custom_network_instance = CustomNetwork(
            current_network_obj,
            visual_prompt_obj_for_customnetwork,
            label_mapping_func_to_pass,
            mapping_sequence_tensor_to_pass,
            args_obj
        ).to(args_obj.device)
        return custom_network_instance
    else:
        current_network_obj.label_mapping_func = label_mapping_func_to_pass
        current_network_obj.mapping_sequence_tensor = mapping_sequence_tensor_to_pass
        return current_network_obj


def print_trainable_params(model):
    print("------------------------------------------")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Trainable params:{name}")
        else:
            print(f"frozen param:{name}")
    print("------------------------------------------")

def main():
    try:
        start_time = time.time()
        args = parse_args()

        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
                            handlers=[logging.StreamHandler(sys.stdout)])
        logger.info(json.dumps(vars(args), indent=4))

        if not hasattr(args, 'lora_rank'):
            args.lora_rank = 0
            logger.warning(f"args.lora_rank 未定义, 使用默认值: {args.lora_rank}")
        elif not isinstance(args.lora_rank, int):
            args.lora_rank = int(args.lora_rank)

        if args.local_rank == -1:
            device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
            if torch.cuda.is_available():
                torch.cuda.set_device(int(args.gpu))
            args.n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
        else:
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
            if not dist.is_initialized():
                torch.distributed.init_process_group(backend="nccl", init_method="env://")
            args.n_gpu = 1
            if dist.is_initialized():
                logger.info(f"Process {args.local_rank} is using GPU {torch.cuda.current_device()}")
                logger.info(f"Initialized process group; rank: {dist.get_rank()}, world size: {dist.get_world_size()}")
        args.device = device

        logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                       (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))
        set_seed(args.seed)

        train_loader, test_loader = get_torch_dataset(args, 'vp')

        logger.info(f"原始网络类型: {args.network}")
        base_model = get_model(args.network, args)
        base_model.to(args.device)

        pure_base_model_for_initial_mapping = None
        use_pure_for_lora_ilm_flm_mapping_experiment = True

        if args.prompt_method == 'dclt' and args.downstream_mapping in ['ilm', 'flm']:
            if use_pure_for_lora_ilm_flm_mapping_experiment:
                pure_base_model_for_initial_mapping = copy.deepcopy(base_model)
                pure_base_model_for_initial_mapping.to(args.device)

        total_params_base = sum(p.numel() for p in base_model.parameters())
        logger.info(f"基础模型 '{args.network}' 参数总数: {total_params_base:,}")

        if args.gradient_accumulation_steps > 0:
            args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
        if args.train_batch_size == 0:
            args.train_batch_size = 1

        processed_network = base_model
        visual_prompt_for_custom_network = None

        if args.prompt_method == 'lor_vp':
            visual_prompt_for_custom_network = LoR_VP(args, normalize=getattr(args, 'normalize', None)).to(args.device)
        elif args.prompt_method == 'dclt':
            is_vit = 'vit' in args.network.lower()

            processed_network = LoR_VP_with_LoRA(
                base_model, args,
                rank=args.lora_rank,
                is_vit=is_vit,
                # lora_target_blocks = [8,9,10,11]
            )
        else:
            visual_prompt_for_custom_network = None

        network = apply_label_mapping(
            processed_network,
            visual_prompt_for_custom_network,
            train_loader,
            args,
            use_pure_base_for_lora_ilm_flm_mapping=use_pure_for_lora_ilm_flm_mapping_experiment,
            pure_base_model_for_mapping=pure_base_model_for_initial_mapping
        )
        print_trainable_params(network)
        tunable_params = network.get_tunable_params()
        tunable_params_num = sum(p.numel() for p in tunable_params)

        if tunable_params_num == 0 and args.mode == 'train':
            logger.error("错误：没有找到任何可训练的参数！请检查模型参数冻结和解冻逻辑。")
            if args.local_rank != -1 and dist.is_initialized(): dist.destroy_process_group()
            return
        elif args.mode == 'train':
            logger.info(f"可训练参数总数 (由 CustomNetwork.get_tunable_params() 得到): {tunable_params_num}")

        optimizer, scheduler = None, None
        if args.mode == 'train':
            if len(train_loader) == 0:
                logger.error("Train loader is empty. Cannot calculate total_train_steps or train.")
                if args.local_rank != -1 and dist.is_initialized(): dist.destroy_process_group()
                return
            if args.gradient_accumulation_steps == 0:
                logger.error("gradient_accumulation_steps is 0, cannot calculate total_train_steps.")
                if args.local_rank != -1 and dist.is_initialized(): dist.destroy_process_group()
                return

            if not hasattr(args, 'total_train_steps') or args.total_train_steps == 0:
                num_train_optimization_steps = (len(train_loader) // args.gradient_accumulation_steps) * args.epochs
                args.total_train_steps = num_train_optimization_steps
                logger.info(f"自动计算 total_train_steps: {args.total_train_steps}")

            if args.total_train_steps == 0 and args.epochs > 0:
                args.total_train_steps = args.epochs

            args.warmup_steps = int(args.total_train_steps * args.warmup_ratio)
            optimizer, scheduler = get_optimizer(tunable_params, args)

        if args.specified_path and args.mode == 'test':
            network = get_specified_model(network, args)

        if args.local_rank != -1:
            network = DDP(network, device_ids=[args.local_rank], output_device=args.local_rank,
                          find_unused_parameters=False)

        logger.info(f"{'*' * 20} 开始训练和评估 {'*' * 20}")
        if args.mode == 'train' and optimizer is not None:
            logger.info("  总优化步数 = %d", args.total_train_steps)
            logger.info("  每个GPU的即时批量大小 = %d", args.train_batch_size if args.train_batch_size > 0 else 0)
            world_size = dist.get_world_size() if args.local_rank != -1 and dist.is_initialized() else 1
            logger.info("  总训练批量大小 (并行、分布式和累积) = %d",
                        args.train_batch_size * args.gradient_accumulation_steps * world_size if args.train_batch_size > 0 else 0)
            logger.info("  梯度累积步数 = %d", args.gradient_accumulation_steps)
        logger.info("  评估批量大小 = %d", args.eval_batch_size)

        best_acc = 0.0
        all_train_losses = []
        all_eval_accs = []

        acc = evaluate(network, test_loader, args)
        best_acc = acc
        all_eval_accs.append(acc)

        if args.mode == 'test':
            logger.info(f"测试模式最终准确率: {best_acc:.4f}")
            if args.local_rank != -1 and dist.is_initialized(): dist.destroy_process_group()
            return

        if args.mode == 'train':
            for epoch_loop_var in range(1, args.epochs + 1):
                args.epoch = epoch_loop_var
                if args.local_rank != -1 and hasattr(train_loader.sampler, 'set_epoch'):
                    train_loader.sampler.set_epoch(epoch_loop_var)
                epoch_loss = train(network, optimizer, scheduler, train_loader, args)
                all_train_losses.append(epoch_loss)

                if args.downstream_mapping == 'ilm' and epoch_loop_var % args.mapping_freq == 0:
                    if isinstance(network, DDP):
                        module_to_update = network.module
                    else:
                        module_to_update = network

                    base_model_for_iter_mapping = None
                    if args.prompt_method == 'dclt' and use_pure_for_lora_ilm_flm_mapping_experiment and \
                            pure_base_model_for_initial_mapping is not None:
                        base_model_for_iter_mapping = pure_base_model_for_initial_mapping

                    apply_label_mapping(
                        module_to_update,
                        None,
                        train_loader,
                        args,
                        use_pure_base_for_lora_ilm_flm_mapping=use_pure_for_lora_ilm_flm_mapping_experiment,
                        pure_base_model_for_mapping=base_model_for_iter_mapping
                    )

                if epoch_loop_var % args.eval_frequency == 0 or epoch_loop_var == args.epochs:
                    if len(test_loader) > 0:
                        acc = evaluate(network, test_loader, args)
                        all_eval_accs.append(acc)
                        if args.local_rank in [-1, 0]:
                            if acc > best_acc:
                                best_acc = acc
                    else:
                        logger.warning(
                            f"Test loader is empty, skipping evaluation at epoch {epoch_loop_var}.")
            if args.local_rank in [-1, 0]:
                logger.info(f"模型检查点路径 (如果保存): {args.save_path}")
                logger.info("最终最佳准确率: \t%f" % best_acc)
        logger.info(f"{'*' * 20} 结束 {'*' * 20}")
        total_time_seconds = time.time() - start_time
        total_time_hours = total_time_seconds / 3600
        logger.info(f"总耗时: {total_time_seconds:.2f} 秒 ({total_time_hours:.2f} 小时)")

    except Exception:
        logger.error(traceback.format_exc())
        pass
    finally:
        if args.local_rank != -1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
