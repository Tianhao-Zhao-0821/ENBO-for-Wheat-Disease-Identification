import os
import json
import torch
from PIL import Image
from torchvision import transforms
from model import efficientnetv2_s as create_model
from collections import Counter, defaultdict
import numpy as np


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 数据预处理 - 与训练时验证集一致
    data_transform = transforms.Compose([
        transforms.Resize(384),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # 主文件夹路径 - 包含多个类别子文件夹
    main_folder_path = r'D:\EfficientNetV2\model\W\wheat'  # 修改为您的主文件夹路径
    assert os.path.exists(main_folder_path), f"主文件夹不存在: {main_folder_path}"

    # 加载类别映射
    json_path = r'.\class_indices.json'
    assert os.path.exists(json_path), f"类别文件不存在: {json_path}"

    with open(json_path, 'r', encoding='utf-8') as f:
        class_indict = json.load(f)

    # 反转字典，用于通过类别名获取ID
    idx_to_class = {int(k): v for k, v in class_indict.items()}
    class_to_idx = {v: int(k) for k, v in class_indict.items()}

    # 创建模型
    model = create_model(num_classes=5, use_bam=True, use_odconv=True).to(device)

    # 加载训练好的最佳模型
    weights_path = "./weights/best_model.pth"

    if os.path.exists(weights_path):
        try:
            # 加载模型权重
            model.load_state_dict(torch.load(weights_path, map_location=device))
            print("训练权重加载成功")
        except Exception as e:
            print(f"权重加载失败: {e}")
            # 尝试处理权重键不匹配的情况
            try:
                checkpoint = torch.load(weights_path, map_location=device)
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint

                # 处理多GPU训练保存的权重
                new_state_dict = {}
                for k, v in state_dict.items():
                    name = k[7:] if k.startswith('module.') else k
                    new_state_dict[name] = v

                model.load_state_dict(new_state_dict, strict=False)
                print("权重加载成功（非严格模式）")
            except Exception as e2:
                print(f"权重加载最终失败: {e2}")
                return
    else:
        print(f"错误: 权重文件不存在 {weights_path}")
        return

    # 预测整个文件夹
    model.eval()

    # 收集所有图片文件，按文件夹（类别）分组
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

    # 数据结构：按文件夹（类别）存储图片路径
    folder_images = defaultdict(list)

    # 遍历主文件夹下的所有子文件夹
    for folder_name in os.listdir(main_folder_path):
        folder_path = os.path.join(main_folder_path, folder_name)
        if os.path.isdir(folder_path):
            # 查找该文件夹中的所有图片
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in image_extensions):
                        full_path = os.path.join(root, file)
                        folder_images[folder_name].append(full_path)

    print(f"找到 {len(folder_images)} 个类别文件夹")
    for folder_name, images in folder_images.items():
        print(f"  类别 '{folder_name}': {len(images)} 张图片")

    # 统计预测结果
    total_counter = Counter()  # 总预测统计
    folder_results = {}  # 每个文件夹的预测结果
    confusion_matrix = np.zeros((len(class_indict), len(class_indict)), dtype=int)  # 混淆矩阵

    print("\n开始预测...")
    print("=" * 80)

    # 遍历每个文件夹
    for folder_name, image_files in folder_images.items():
        print(f"\n📁 处理文件夹: {folder_name} ({len(image_files)} 张图片)")
        print("-" * 60)

        folder_counter = Counter()
        correct_count = 0
        folder_confidences = []

        # 获取该文件夹的真实类别ID（假设文件夹名就是类别名）
        true_class_name = folder_name
        true_class_id = class_to_idx.get(true_class_name, -1)

        if true_class_id == -1:
            print(f"⚠️  警告: 文件夹名 '{folder_name}' 不在类别列表中，跳过统计准确率")

        results = []

        for i, img_path in enumerate(image_files):
            try:
                # 加载并预处理图片
                image = Image.open(img_path).convert('RGB')
                img = data_transform(image)
                img = torch.unsqueeze(img, dim=0)

                # 预测
                with torch.no_grad():
                    output = model(img.to(device))
                    probabilities = torch.softmax(output, dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0, predicted_class].item()

                # 获取预测的类别名称
                if predicted_class in idx_to_class:
                    predicted_class_name = idx_to_class[predicted_class]
                    total_counter[predicted_class_name] += 1
                    folder_counter[predicted_class_name] += 1

                    # 更新混淆矩阵
                    if true_class_id != -1 and true_class_id < len(class_indict) and predicted_class < len(
                            class_indict):
                        confusion_matrix[true_class_id][predicted_class] += 1

                    # 检查是否预测正确
                    is_correct = (predicted_class == true_class_id)
                    if is_correct:
                        correct_count += 1

                    folder_confidences.append(confidence)

                    # 保存结果
                    results.append({
                        'file': os.path.basename(img_path),
                        'predicted_class': predicted_class_name,
                        'confidence': confidence,
                        'class_id': predicted_class,
                        'is_correct': is_correct
                    })

                    # 打印单张图片结果
                    status = "✓" if is_correct else "✗"
                    print(
                        f"{i + 1:3d}/{len(image_files)}: {status} {os.path.basename(img_path)[:20]:20s} -> {predicted_class_name:15s} (置信度: {confidence:.3f})")

                else:
                    print(
                        f"{i + 1:3d}/{len(image_files)}: {os.path.basename(img_path)[:20]:20s} -> 未知类别 (ID: {predicted_class})")

            except Exception as e:
                print(
                    f"{i + 1:3d}/{len(image_files)}: {os.path.basename(img_path)[:20]:20s} -> 预测失败: {str(e)[:30]}")

        # 计算该文件夹的准确率
        accuracy = (correct_count / len(image_files)) * 100 if len(image_files) > 0 else 0
        avg_confidence = np.mean(folder_confidences) if folder_confidences else 0

        # 保存该文件夹的结果
        folder_results[folder_name] = {
            'total_images': len(image_files),
            'correct_count': correct_count,
            'accuracy': accuracy,
            'avg_confidence': avg_confidence,
            'predictions': folder_counter,
            'results': results
        }

        print(f"\n📊 文件夹 '{folder_name}' 统计:")
        print(f"  总图片数: {len(image_files)}")
        print(f"  正确预测: {correct_count}")
        print(f"  准确率: {accuracy:.2f}%")
        print(f"  平均置信度: {avg_confidence:.3f}")

        # 显示该文件夹的预测分布
        if folder_counter:
            print(f"  预测分布:")
            for class_name, count in folder_counter.most_common():
                percentage = (count / len(image_files)) * 100 if len(image_files) > 0 else 0
                print(f"    {class_name:15s}: {count:3d} 张 ({percentage:5.1f}%)")

    # 输出总体统计结果
    print("\n" + "=" * 80)
    print("📈 总体预测结果统计:")
    print("=" * 80)

    # 计算总体准确率
    total_images = sum(len(images) for images in folder_images.values())
    total_correct = sum(info['correct_count'] for info in folder_results.values())
    overall_accuracy = (total_correct / total_images) * 100 if total_images > 0 else 0

    print(f"\n📊 总体统计:")
    print(f"  总图片数量: {total_images}")
    print(f"  总正确预测: {total_correct}")
    print(f"  总体准确率: {overall_accuracy:.2f}%")

    # 显示每个文件夹的准确率
    print(f"\n📁 各文件夹准确率:")
    print("-" * 50)
    for folder_name, info in folder_results.items():
        print(f"  {folder_name:15s}: {info['accuracy']:6.2f}% ({info['correct_count']}/{info['total_images']})")

    # 显示总体预测分布
    print(f"\n📊 总体预测分布:")
    print("-" * 50)
    for class_name, count in total_counter.most_common():
        percentage = (count / total_images) * 100 if total_images > 0 else 0
        bar_length = int((count / total_images) * 30) if total_images > 0 else 0
        bar = "█" * bar_length
        print(f"  {class_name:15s}: {bar:30s} {count:4d} ({percentage:5.1f}%)")

    # 显示混淆矩阵（如果类别匹配）
    print(f"\n📋 混淆矩阵:")
    print("-" * (len(class_indict) * 8 + 10))

    # 打印表头
    print("真实类别 \\ 预测类别", end="")
    for class_id in sorted(idx_to_class.keys()):
        print(f"{idx_to_class[class_id][:6]:>8s}", end="")
    print()

    # 打印矩阵内容
    for true_id in sorted(idx_to_class.keys()):
        true_name = idx_to_class[true_id]
        print(f"{true_name:15s}", end="")
        for pred_id in sorted(idx_to_class.keys()):
            print(f"{confusion_matrix[true_id][pred_id]:8d}", end="")
        print()

    # 保存详细结果到文件
    output_file = "folder_prediction_results.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("各文件夹图片预测结果详情\n")
        f.write("=" * 70 + "\n")
        f.write(f"主文件夹路径: {main_folder_path}\n")
        f.write(f"总图片数: {total_images}\n")
        f.write(f"总正确预测: {total_correct}\n")
        f.write(f"总体准确率: {overall_accuracy:.2f}%\n\n")

        # 写入每个文件夹的结果
        for folder_name, info in folder_results.items():
            f.write(f"\n📁 文件夹: {folder_name}\n")
            f.write("-" * 50 + "\n")
            f.write(f"总图片数: {info['total_images']}\n")
            f.write(f"正确预测: {info['correct_count']}\n")
            f.write(f"准确率: {info['accuracy']:.2f}%\n")
            f.write(f"平均置信度: {info['avg_confidence']:.3f}\n")

            f.write(f"\n预测分布:\n")
            for class_name, count in info['predictions'].most_common():
                percentage = (count / info['total_images']) * 100 if info['total_images'] > 0 else 0
                f.write(f"  {class_name}: {count} 张 ({percentage:.1f}%)\n")

            f.write(f"\n详细预测结果:\n")
            for result in info['results']:
                status = "✓" if result['is_correct'] else "✗"
                f.write(
                    f"  {status} {result['file']:20s} -> {result['predicted_class']:15s} (置信度: {result['confidence']:.3f})\n")

        # 写入混淆矩阵
        f.write(f"\n\n📋 混淆矩阵:\n")
        f.write("-" * (len(class_indict) * 8 + 10) + "\n")

        # 表头
        f.write("真实类别 \\ 预测类别")
        for class_id in sorted(idx_to_class.keys()):
            f.write(f"{idx_to_class[class_id][:6]:>8s}")
        f.write("\n")

        # 矩阵内容
        for true_id in sorted(idx_to_class.keys()):
            true_name = idx_to_class[true_id]
            f.write(f"{true_name:15s}")
            for pred_id in sorted(idx_to_class.keys()):
                f.write(f"{confusion_matrix[true_id][pred_id]:8d}")
            f.write("\n")

    print(f"\n✅ 详细结果已保存到: {output_file}")


if __name__ == '__main__':
    main()