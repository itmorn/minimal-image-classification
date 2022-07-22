"""
@Auth: itmorn
@Date: 2022/7/19-16:08
@Email: 12567148@qq.com
"""
import time

import torch
from torch import nn
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

import utils


def inference_office_model():
    from torchvision.io.image import read_image
    from torchvision.models.resnet import resnet18, ResNet18_Weights
    img = read_image("../images/balloon.jpg")

    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.eval()

    preprocess = weights.transforms()

    batch = preprocess(img).unsqueeze(0)

    prediction = model(batch)[0].softmax(0)
    idx, score = prediction.argmax(), prediction.max()

    categories = weights.meta["categories"][idx]
    print(categories, float(score))


def vis_office_model():
    import torch
    from torchvision.io.image import read_image
    from torchvision.models.resnet import resnet18, ResNet18_Weights
    img = read_image("../images/balloon.jpg")

    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.eval()

    model_name = "resnet18"

    torch.save(model, f'{model_name}.pth')

    preprocess = weights.transforms()

    batch = preprocess(img).unsqueeze(0)

    torch.onnx.export(
        model,
        batch,
        f'{model_name}.onnx',
        export_params=True,
        opset_version=16,
    )

    # 增加维度信息
    import onnx.shape_inference
    model_file = f'{model_name}.onnx'
    onnx_model = onnx.load(model_file)
    onnx.save(onnx.shape_inference.infer_shapes(onnx_model), model_file)
    print("done")


def eval_office_model():
    import torch.backends.cudnn
    from torch import nn
    import torchvision
    import torch.utils.data
    from torchvision.models.resnet import resnet18, ResNet18_Weights
    import utils
    from tqdm import tqdm
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    valdir = '/data01/zhaoyichen/data/ImageNet1k/val'

    weights = ResNet18_Weights.DEFAULT
    preprocessing = weights.transforms()

    dataset_test = torchvision.datasets.ImageFolder(
        valdir,
        preprocessing,
    )
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=32, sampler=test_sampler, num_workers=16, pin_memory=True
    )

    model = resnet18(weights=weights)
    # model = nn.DataParallel(model,device_ids=range(torch.cuda.device_count()))
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss(label_smoothing=0)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    acc1_all = 0
    acc5_all = 0
    loss_all = 0
    num = 0
    with torch.inference_mode():
        for image, target in tqdm(data_loader_test):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            batch_size = image.shape[0]

            acc1_all += acc1.item()*batch_size
            acc5_all += acc5.item()*batch_size
            loss_all += loss.item()*batch_size
            num += batch_size

            print(f"acc1:{acc1_all/num}\tacc5:{acc5_all/num}\tloss:{loss/num}")
            # acc1:56.556	acc5:79.084	loss:0.00011162559530930594

def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema=None, scaler=None):
    model.train()
    acc1_all = 0
    acc5_all = 0
    loss_all = 0
    num = 0
    header = f"Epoch: [{epoch}]"
    for image, target in tqdm(data_loader):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.clip_grad_norm is not None:
                # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()


        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        acc1_all += acc1.item() * batch_size
        acc5_all += acc5.item() * batch_size
        loss_all += loss.item() * batch_size
        num += batch_size

        print(f"acc1:{acc1_all / num}\tacc5:{acc5_all / num}\tloss:{loss / num}")

def evaluate(model, criterion, data_loader, device, print_freq=100, log_suffix=""):
    model.eval()
    acc1_all = 0
    acc5_all = 0
    loss_all = 0
    num = 0
    with torch.inference_mode():
        for image, target in tqdm(data_loader):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            batch_size = image.shape[0]

            acc1_all += acc1.item() * batch_size
            acc5_all += acc5.item() * batch_size
            loss_all += loss.item() * batch_size
            num += batch_size

            print(f"acc1:{acc1_all / num}\tacc5:{acc5_all / num}\tloss:{loss / num}")


def train_model():
    import torch.backends.cudnn
    from torch import nn
    import torchvision
    import torch.utils.data
    from torchvision.models.resnet import resnet18, ResNet18_Weights
    import utils
    from tqdm import tqdm
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    traindir = '/data01/zhaoyichen/data/ImageNet1k/train'

    weights = ResNet18_Weights.DEFAULT
    preprocessing = weights.transforms()

    from utils import ClassificationPresetTrain
    dataset = torchvision.datasets.ImageFolder(
        traindir,
        ClassificationPresetTrain(
            crop_size=224,
            interpolation=InterpolationMode.BILINEAR,
            auto_augment_policy=None,
            random_erase_prob=0,
        ),
    )
    is_distributed = False
    if is_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)

    num_classes = len(dataset.classes)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        sampler=train_sampler,
        num_workers=16,
        pin_memory=True,
        collate_fn=None,
    )

    model =  resnet18(weights=weights, num_classes=num_classes)
    model.to(device)


    criterion = nn.CrossEntropyLoss(label_smoothing=0)

    parameters = utils.set_weight_decay(
        model,
        weight_decay=0.0001,
        norm_weight_decay=None,
        custom_keys_weight_decay=[],
    )

    opt_name = "sgd"
    optimizer = torch.optim.SGD(
        parameters,
        lr=0.1,
        momentum=0.9,
        weight_decay=0.0001,
        nesterov="nesterov" in opt_name,
    )

    amp = False
    scaler = torch.cuda.amp.GradScaler() if amp else None

    lr_scheduler = "steplr"
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    model_without_ddp = model
    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    print("Start training")
    start_time = time.time()
    start_epoch = 0
    epochs = 90
    for epoch in range(start_epoch, epochs):
        if is_distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema, scaler)
        lr_scheduler.step()
        evaluate(model, criterion, data_loader_test, device=device)
        if model_ema:
            evaluate(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA")
        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            if model_ema:
                checkpoint["model_ema"] = model_ema.state_dict()
            if scaler:
                checkpoint["scaler"] = scaler.state_dict()
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")



    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    acc1_all = 0
    acc5_all = 0
    loss_all = 0
    num = 0
    with torch.inference_mode():
        for image, target in tqdm(data_loader_test):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            batch_size = image.shape[0]

            acc1_all += acc1.item() * batch_size
            acc5_all += acc5.item() * batch_size
            loss_all += loss.item() * batch_size
            num += batch_size

            print(f"acc1:{acc1_all / num}\tacc5:{acc5_all / num}\tloss:{loss / num}")
            # acc1:56.556	acc5:79.084	loss:0.00011162559530930594
    pass



if __name__ == '__main__':
    # 使用官方预训练模型推理
    # inference_office_model()

    # 官方模型可视化
    # vis_office_model()

    # 在ImageNet的val上评估官方预训练模型的效果
    # eval_office_model()

    # 训练模型
    train_model()
