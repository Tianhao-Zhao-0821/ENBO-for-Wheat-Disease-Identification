import argparse
import math
import os
import torch
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from model import efficientnetv2_s as create_model
from my_dataset import MyDataSet
from utils import read_split_data, train_one_epoch, evaluate

from torchvision import transforms
import torch.optim as optim

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')

    tb_writer = SummaryWriter()
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    img_size = {"s": [300, 384],
                "m": [384, 480],
                "l": [384, 480]}
    num_model = "s"

    # 数据预处理 - 确保验证集预处理与预测时一致
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(img_size[num_model][0]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        "val": transforms.Compose([
            transforms.Resize(img_size[num_model][1]),  # 先调整大小到384
            transforms.CenterCrop(img_size[num_model][1]),  # 中心裁剪到384
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))

    # 添加数据集大小检查
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")

    # 修改DataLoader，添加drop_last=True
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               drop_last=True,  # 丢弃最后一个不完整的batch
                                               collate_fn=train_dataset.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             drop_last=True,  # 丢弃最后一个不完整的batch
                                             collate_fn=val_dataset.collate_fn)

    print(f"训练集batch数量: {len(train_loader)}")
    print(f"验证集batch数量: {len(val_loader)}")

    model = create_model(num_classes=args.num_classes, use_bam=args.use_bam, use_odconv=args.use_odconv).to(device)

    # 检查模型中的BatchNorm层
    bn_layers = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            bn_layers.append(name)
    print(f"模型中的BatchNorm层数量: {len(bn_layers)}")
    if bn_layers:
        print(f"BatchNorm层名称: {bn_layers[:5]}")  # 只显示前5个

    if args.weights != "":
        if os.path.exists(args.weights):
            weights_dict = torch.load(args.weights, map_location=device)
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                 if model.state_dict()[k].numel() == v.numel()}
            print(model.load_state_dict(load_weights_dict, strict=False))
        else:
            raise FileNotFoundError("not found weights file: {}".format(args.weights))

        if args.freeze_layers:
            for name, para in model.named_parameters():
                if "head" not in name:
                    para.requires_grad_(False)
                else:
                    print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=4E-5)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    best_acc = 0.0
    for epoch in range(args.epochs):
        # train
        train_result = train_one_epoch(model=model,
                                       optimizer=optimizer,
                                       data_loader=train_loader,
                                       device=device,
                                       epoch=epoch)

        # 处理train_one_epoch的返回值
        if isinstance(train_result, tuple):
            mean_loss = train_result[0]
        else:
            mean_loss = train_result

        if torch.is_tensor(mean_loss):
            mean_loss = mean_loss.item()

        scheduler.step()

        # validate
        val_result = evaluate(model=model,
                              data_loader=val_loader,
                              device=device,
                              epoch=epoch)

        # 处理evaluate的返回值
        if isinstance(val_result, tuple) and len(val_result) == 2:
            val_loss, acc = val_result
            if torch.is_tensor(acc):
                acc = acc.item()
            print("[epoch {}] loss: {:.4f}, accuracy: {:.3f}%".format(epoch, mean_loss, acc))
        else:
            acc = val_result
            if torch.is_tensor(acc):
                acc = acc.item()
            print("[epoch {}] accuracy: {:.3f}%".format(epoch, acc))

        # TensorBoard记录
        tags = ["loss", "accuracy", "learning_rate"]
        tb_writer.add_scalar(tags[0], mean_loss, epoch)
        tb_writer.add_scalar(tags[1], acc, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

        # 保存最佳模型
        if acc > best_acc:
            best_acc = acc
            # 保存为best_model.pth，方便预测时加载
            best_model_path = "./weights/best_model.pth"
            torch.save(model.state_dict(), best_model_path)
            print("Saved best model with accuracy: {:.3f}%".format(best_acc))

            # 同时保存epoch模型
            epoch_model_path = "./weights/model-{}.pth".format(epoch)
            torch.save(model.state_dict(), epoch_model_path)

    print("Training completed. Best accuracy: {:.3f}%".format(best_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.1)
    parser.add_argument('--data-path', type=str, default=r"D:\EfficientNetV2\model\W\wheat")
    parser.add_argument('--weights', type=str, default='./pre_efficientnetv2-s.pth', help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--use-bam', type=bool, default=True, help='use BAM attention mechanism')
    parser.add_argument('--use-odconv', type=bool, default=True, help='use ODCONV attention mechanism')

    opt = parser.parse_args()
    main(opt)