# diagnosis.py
import torch
from model import efficientnetv2_s as create_model
import json


def diagnose_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 加载类别映射
    with open('class_indices.json', 'r') as f:
        class_indict = json.load(f)

    # 创建模型并加载权重
    model = create_model(num_classes=5, use_bam=True, use_odconv=True).to(device)
    model.load_state_dict(torch.load("./weights/best_model.pth", map_location=device))
    model.eval()

    print("=== 模型诊断 ===")

    # 测试1: 检查模型对随机噪声的反应（过拟合测试）
    print("1. 随机噪声测试:")
    random_input = torch.randn(1, 3, 384, 384).to(device)
    with torch.no_grad():
        random_output = model(random_input)
        random_probs = torch.softmax(random_output, dim=1)
        max_prob = torch.max(random_probs).item()
        print(f"   随机输入的最大置信度: {max_prob:.3f}")
        if max_prob > 0.9:
            print("   ⚠️ 模型可能过拟合：对随机噪声也给出高置信度")

    # 测试2: 检查各类别的预测倾向
    print("2. 类别偏好分析:")
    with torch.no_grad():
        for i in range(5):
            # 创建偏向某个类别的伪输入
            pseudo_input = torch.randn(1, 3, 384, 384).to(device) * 0.1
            output = model(pseudo_input)
            probs = torch.softmax(output, dim=1)
            pred_class = torch.argmax(probs).item()
            confidence = probs[0, pred_class].item()
            print(f"   伪输入 -> {class_indict[str(pred_class)]} (置信度: {confidence:.3f})")


if __name__ == '__main__':
    diagnose_model()