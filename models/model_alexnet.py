"""
@Auth: itmorn
@Date: 2022/7/19-16:08
@Email: 12567148@qq.com
"""


def inference_office_model():
    from torchvision.io.image import read_image
    from torchvision.models.alexnet import alexnet, AlexNet_Weights
    img = read_image("../images/balloon.jpg")

    weights = AlexNet_Weights.DEFAULT
    model = alexnet(weights=weights)
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
    from torchvision.models.alexnet import alexnet, AlexNet_Weights
    # import onnx.version_converter
    img = read_image("../images/balloon.jpg")

    weights = AlexNet_Weights.DEFAULT
    model = alexnet(weights=weights)
    model.eval()

    torch.save(model, 'alexnet.pth')

    preprocess = weights.transforms()

    batch = preprocess(img).unsqueeze(0)

    torch.onnx.export(
        model,
        batch,
        'alexnet.onnx',
        export_params=True,
        opset_version=16,
    )

    # 增加维度信息
    import onnx.shape_inference
    model_file = 'alexnet.onnx'
    onnx_model = onnx.load(model_file)
    onnx.save(onnx.shape_inference.infer_shapes(onnx_model), model_file)
    print("done")


def eval_office_model():
    import torch.backends.cudnn
    from torch import nn
    import torchvision
    import torch.utils.data
    from torchvision.models.alexnet import alexnet, AlexNet_Weights
    import utils
    from tqdm import tqdm
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    valdir = '/data01/zhaoyichen/data/ImageNet1k/val'

    weights = AlexNet_Weights.DEFAULT
    preprocessing = weights.transforms()

    dataset_test = torchvision.datasets.ImageFolder(
        valdir,
        preprocessing,
    )
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=32, sampler=test_sampler, num_workers=16, pin_memory=True
    )

    model = alexnet(weights=weights)
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


def train_model():
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
