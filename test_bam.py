# test_bam.py
import torch
from model import efficientnetv2_s
from bam import BAM


# 测试BAM模块
def test_bam():
    bam = BAM(64)
    x = torch.randn(2, 64, 32, 32)
    output = bam(x)
    print(f"BAM输入形状: {x.shape}")
    print(f"BAM输出形状: {output.shape}")
    print("BAM测试通过!")


# 测试带BAM的EfficientNetV2
def test_effnet_bam():
    model = efficientnetv2_s(num_classes=4, use_bam=True)
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    print(f"模型输入形状: {x.shape}")
    print(f"模型输出形状: {output.shape}")
    print("带BAM的EfficientNetV2测试通过!")

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")


if __name__ == '__main__':
    test_bam()
    test_effnet_bam()