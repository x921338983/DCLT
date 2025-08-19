# test_dataset.py
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder  # 假设Tiny ImageNet是ImageFolder格式
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import argparse  # 用于从命令行获取参数


# 假设您的数据加载和模型获取函数在以下utils中
# 请根据您的项目结构调整导入路径
# from utils.model_and_data import get_torch_dataset # 您获取dataloader的函数
# from utils.label_mapping import label_mapping_base # 如果需要测试标签映射

def parse_test_args():
    parser = argparse.ArgumentParser(description='Dataset Test Script')
    parser.add_argument('--datadir', type=str, default='../data', help='Data directory containing Tiny ImageNet')
    parser.add_argument('--dataset_name', type=str, default='tiny-imagenet-200', help='Name of the dataset to test')
    # 添加其他必要的参数，例如 input_size, batch_size 等，这些参数可能被您的 get_torch_dataset 使用
    parser.add_argument('--input_size', type=int, default=224)  # 根据您的配置
    parser.add_argument('--batch_size', type=int, default=4)  # 测试少量样本即可
    parser.add_argument('--num_workers', type=int, default=0)
    # 如果您的 get_torch_dataset 需要 args.class_cnt, args.network 等，也需要加上
    parser.add_argument('--class_cnt', type=int, default=200,
                        help='Number of classes for Tiny ImageNet')  # Tiny ImageNet通常是200类
    parser.add_argument('--network', type=str, default='resnet18', help='Network name, may affect transforms')

    # 您可以添加更多来自您主训练脚本的参数定义，如果 get_torch_dataset 依赖它们
    # 例如，如果您的 get_torch_dataset 中有特定于 'vp' 的逻辑
    # parser.add_argument('--prompt_method', type=str, default='lor_vp') # 示例

    return parser.parse_args()


def denormalize(tensor, mean, std):
    mean = torch.tensor(mean).reshape(1, -1, 1, 1)
    std = torch.tensor(std).reshape(1, -1, 1, 1)
    return tensor * std + mean


def inspect_dataset(args):
    print(f"Inspecting dataset: {args.dataset_name} from {args.datadir}")

    # ----------------------------------------------------------------------
    # 步骤1: 获取数据加载器 (DataLoader)
    # 您需要用您项目中实际的 get_torch_dataset 函数来替换这部分逻辑
    # 为了演示，这里用一个简化的 ImageFolder 示例 (假设Tiny ImageNet结构)
    # ----------------------------------------------------------------------

    # 示例：Tiny ImageNet 的典型目录结构
    # TRAIN_DIR = os.path.join(args.datadir, args.dataset_name, 'train')
    # VAL_DIR = os.path.join(args.datadir, args.dataset_name, 'val', 'images') # val的结构可能特殊
    # VAL_ANNOTATION_FILE = os.path.join(args.datadir, args.dataset_name, 'val', 'val_annotations.txt')

    # 定义通用的预处理 (与您训练时验证集的预处理保持一致)
    # 您需要从您的代码中找到 Tiny ImageNet 验证集使用的实际 transform
    # 例如：
    # imagenet_mean = [0.485, 0.456, 0.406]
    # imagenet_std = [0.229, 0.224, 0.225]
    # test_transform = transforms.Compose([
    #     transforms.Resize(args.input_size), # 或者先 Resize(args.output_size) 再 CenterCrop(args.input_size)
    #     transforms.CenterCrop(args.input_size),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    # ])

    # **请替换以下为您项目中实际获取 test_loader (验证集加载器) 的方式**
    # 伪代码:
    # try:
    #     # 假设您的 get_torch_dataset 返回 (train_loader, test_loader)
    #     # 并且它需要一个类似您主脚本中的 args 对象
    #     # 您可能需要创建一个模拟的 args 对象，包含 get_torch_dataset 所需的所有字段
    #     mock_args_for_dataloader = args # 直接使用传入的 args，确保它包含了所有必要字段
    #     _, val_loader = get_torch_dataset(mock_args_for_dataloader, 'vp') # 'vp' 或其他您加载数据时用的标识
    #     print(f"成功使用 get_torch_dataset 加载验证集数据加载器。批量大小: {val_loader.batch_size}")
    # except Exception as e:
    #     print(f"错误：使用您的 get_torch_dataset 加载数据失败: {e}")
    #     print("请确保 get_torch_dataset 函数路径正确，并且传入的 args 包含所有必需参数。")
    #     print("下方将使用一个通用的 ImageFolder 示例（如果适用）。")
    #     # 如果 get_torch_dataset 失败，您可以尝试下面的通用 ImageFolder 加载方式（如果适用）
    #     # 但强烈建议先修复 get_torch_dataset 的调用
    #     return

    # --- 临时的、通用的 ImageFolder 加载方式 (作为备用检查，实际应使用您的 get_torch_dataset) ---
    # 仅当您确认 Tiny ImageNet 验证集是标准 ImageFolder 结构时使用
    # Tiny ImageNet 的验证集通常需要特殊处理（val_annotations.txt）
    # 所以直接用 ImageFolder 可能不适用于验证集，但可以试试训练集目录
    temp_train_dir = os.path.join(args.datadir, args.dataset_name, 'train')
    if not os.path.exists(temp_train_dir):
        print(f"错误：训练集目录 {temp_train_dir} 不存在。无法进行基本的数据检查。")
        return

    # 使用一个简单的、不带复杂归一化的 transform 来可视化
    simple_display_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),  # 调整到期望的输入尺寸
        transforms.ToTensor()
    ])
    try:
        # 我们先尝试加载训练集，因为它的结构通常更标准
        train_dataset_for_inspection = ImageFolder(root=temp_train_dir, transform=simple_display_transform)
        train_loader_for_inspection = torch.utils.data.DataLoader(train_dataset_for_inspection,
                                                                  batch_size=args.batch_size, shuffle=True)
        print(
            f"成功加载训练集进行基本检查。共找到 {len(train_dataset_for_inspection)} 张图片，分为 {len(train_dataset_for_inspection.classes)} 个类别。")
        print(f"类别名称示例: {train_dataset_for_inspection.classes[:5]}")
        print(f"类别到索引的映射示例: {list(train_dataset_for_inspection.class_to_idx.items())[:5]}")

        # 检查类别数量是否与 args.class_cnt 匹配
        if len(train_dataset_for_inspection.classes) != args.class_cnt:
            print(
                f"警告：数据集中检测到的类别数 ({len(train_dataset_for_inspection.classes)}) 与 args.class_cnt ({args.class_cnt}) 不匹配！")

        val_loader = train_loader_for_inspection  # 为了能运行下去，暂时用训练集代替，您应该用真实的验证集加载器
        print(
            "注意：由于Tiny ImageNet验证集结构特殊，此处暂时使用训练集样本进行可视化。请务必用您项目中真实的验证集加载器进行测试。")

    except Exception as e:
        print(f"使用通用 ImageFolder 加载训练集失败: {e}")
        return
    # --- 临时加载结束 ---

    # ----------------------------------------------------------------------
    # 步骤2: 从数据加载器中获取一批数据
    # ----------------------------------------------------------------------
    try:
        images, labels = next(iter(val_loader))
        print(f"\n成功从验证集加载器获取一批数据:")
        print(f"  图像批次形状 (Images batch shape): {images.shape}")  # 应该是 [batch_size, channels, height, width]
        print(f"  标签批次形状 (Labels batch shape): {labels.shape}")  # 应该是 [batch_size]
        print(f"  图像数据类型 (Images data type): {images.dtype}")
        print(f"  标签数据类型 (Labels data type): {labels.dtype}")
        print(
            f"  图像数据的最大/最小值 (Min/Max value in images batch): {images.min().item():.4f} / {images.max().item():.4f}")
        print(f"  前几个标签 (First few labels in batch): {labels[:args.batch_size].tolist()}")

        # 检查标签范围是否合理 (例如，对于 Tiny ImageNet 200类，标签应在 0-199 之间)
        if labels.min() < 0 or labels.max() >= args.class_cnt:  # args.class_cnt 是您模型配置的类别数
            print(f"警告：标签值范围 [{labels.min().item()}, {labels.max().item()}] 超出预期 [0, {args.class_cnt - 1}]！")

    except Exception as e:
        print(f"错误：从验证集加载器获取数据失败: {e}")
        return

    # ----------------------------------------------------------------------
    # 步骤3: 可视化一些样本及其标签
    # ----------------------------------------------------------------------
    num_to_show = min(args.batch_size, 4)  # 最多显示4张

    # 尝试反归一化以正确显示图像 (如果您的 test_transform 中有 Normalize)
    # 您需要提供正确的均值和标准差
    # imagenet_mean = [0.485, 0.456, 0.406] # 示例
    # imagenet_std = [0.229, 0.224, 0.225]  # 示例

    fig, axes = plt.subplots(1, num_to_show, figsize=(15, 5))
    fig.suptitle(f'Sample Validation Images (First {num_to_show} from batch)')

    for i in range(num_to_show):
        img_tensor = images[i]

        # 反归一化 (如果原始 transform 中有 Normalize)
        # try:
        #    img_to_show = denormalize(img_tensor.unsqueeze(0), imagenet_mean, imagenet_std).squeeze(0) # denormalize需要batch维度
        #    img_to_show = img_to_show.permute(1, 2, 0) # 从 [C, H, W] 转为 [H, W, C] for Matplotlib
        #    img_to_show = torch.clamp(img_to_show, 0, 1) # 确保值在 [0,1]
        #    img_to_show = img_to_show.cpu().numpy()
        # except NameError: # 如果 imagenet_mean/std 未定义，则直接显示
        img_to_show = img_tensor.permute(1, 2, 0).cpu().numpy()  # 直接显示 ToTensor()后的结果
        img_to_show = np.clip(img_to_show, 0, 1)  # 如果没有反归一化，但值在0-1，这样可以显示

        ax = axes[i] if num_to_show > 1 else axes
        ax.imshow(img_to_show)

        actual_label_idx = labels[i].item()
        label_text = f"Label: {actual_label_idx}"

        # 如果您有类别名称映射 (例如从 ImageFolder 的 dataset.classes)
        # if hasattr(val_loader.dataset, 'classes') and val_loader.dataset.classes:
        #    if 0 <= actual_label_idx < len(val_loader.dataset.classes):
        #        class_name = val_loader.dataset.classes[actual_label_idx]
        #        label_text += f"\nName: {class_name}"

        ax.set_title(label_text)
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # 保存或显示图像
    plot_filename = f"dataset_inspection_{args.dataset_name}.png"
    plt.savefig(plot_filename)
    print(f"\n已将样本图像保存至: {plot_filename}")
    # plt.show() # 如果在GUI环境，可以取消注释这行来直接显示

    print("\n数据集检查脚本执行完毕。请检查输出信息和保存的图像。")


if __name__ == '__main__':
    test_args = parse_test_args()
    inspect_dataset(test_args)