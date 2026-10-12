import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i #B,H,W(true/false)    # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))#B,1,H,W
        output_tensor = torch.cat(tensor_list, dim=1)#-->B,9,H,W
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)#(B,224,224)->(B,class,224,224)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])#dice接近1越好
            class_wise_dice.append(1.0 - dice.item())#损失训练越小越好 1-dice系数
            loss += dice * weight[i] #总diceloss
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt) #-->1  (slice,h,w)0/1
        hd95 = metric.binary.hd95(pred, gt) #-->0
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:#(slice,h,w)
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]#切片
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input)#1,1,224,224 -->1,9,224,224
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()# (224,224)
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):#去掉背景 (slice,h,w)
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list


# 修改后
def test_single_volume_1(
        image,
        label,
        net,
        classes,
        patch_size=[256, 256],
        test_save_path=None,
        case=None,
        z_spacing=1
):
    image = image.squeeze(0).cpu().detach().numpy()
    label = label.squeeze(0).cpu().detach().numpy()

    if len(image.shape) == 3:
        # image shape: (slice, h, w)
        prediction = np.zeros_like(label)

        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]

            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(
                    slice,
                    (
                        patch_size[0] / x,
                        patch_size[1] / y
                    ),
                    order=3
                )  # previous using 0

            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()

            net.eval()

            with torch.no_grad():
                # outputs shape:
                # (1, num_classes, patch_h, patch_w)
                outputs = net(input)

                out = torch.argmax(
                    torch.softmax(outputs, dim=1),
                    dim=1
                ).squeeze(0)

                # out shape:
                # (patch_h, patch_w)
                out = out.cpu().detach().numpy()

                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(
                        out,
                        (
                            x / patch_size[0],
                            y / patch_size[1]
                        ),
                        order=0
                    )
                else:
                    pred = out

                prediction[ind] = pred

    else:
        input = torch.from_numpy(image).unsqueeze(
            0
        ).unsqueeze(
            0
        ).float().cuda()

        net.eval()

        with torch.no_grad():
            out = torch.argmax(
                torch.softmax(net(input), dim=1),
                dim=1
            ).squeeze(0)

            prediction = out.cpu().detach().numpy()

    metric_list = []

    # 从 1 开始，去掉背景类别 0
    for i in range(1, classes):
        # 当前类别的二值预测结果和标签
        # shape: (slice, h, w)
        pred_i = prediction == i
        label_i = label == i

        # 保留原来的 Dice 和 HD95 计算
        dice, hd95 = calculate_metric_percase(
            pred_i,
            label_i
        )

        # 新增 IoU、Recall、Precision
        iou, recall, precision = calculate_iou_recall_precision(
            pred_i,
            label_i
        )

        # 每个类别的指标顺序：
        # Dice、HD95、IoU、Recall、Precision
        metric_list.append(
            (
                dice,
                hd95,
                iou,
                recall,
                precision
            )
        )

    return metric_list


def calculate_iou_recall_precision(pred, label):
    """
    计算单个类别的 IoU、Recall 和 Precision。

    参数
    ----------
    pred:
        当前类别的二值预测结果，shape 可以为：
        (slice, h, w) 或 (h, w)

    label:
        当前类别的二值真实标签，shape 与 pred 相同。

    返回
    -------
    iou:
        TP / (TP + FP + FN)

    recall:
        TP / (TP + FN)

    precision:
        TP / (TP + FP)
    """

    # 转换为布尔类型，避免数值类型影响逻辑计算
    pred = pred.astype(np.bool_)
    label = label.astype(np.bool_)

    # 真阳性：
    # 模型预测为当前类别，标签也为当前类别
    true_positive = np.logical_and(
        pred,
        label
    ).sum(dtype=np.float64)

    # 假阳性：
    # 模型预测为当前类别，但标签不是当前类别
    false_positive = np.logical_and(
        pred,
        np.logical_not(label)
    ).sum(dtype=np.float64)

    # 假阴性：
    # 模型没有预测为当前类别，但标签是当前类别
    false_negative = np.logical_and(
        np.logical_not(pred),
        label
    ).sum(dtype=np.float64)

    # IoU 分母
    iou_denominator = (
        true_positive
        + false_positive
        + false_negative
    )

    # Recall 分母
    recall_denominator = (
        true_positive
        + false_negative
    )

    # Precision 分母
    precision_denominator = (
        true_positive
        + false_positive
    )

    # 预测和标签中都不存在当前类别
    if iou_denominator == 0:
        iou = 1.0
    else:
        iou = true_positive / iou_denominator

    # 标签中不存在当前类别时，Recall 分母为 0
    if recall_denominator == 0:
        # 预测中同样不存在该类别，认为没有漏检
        if precision_denominator == 0:
            recall = 1.0
        else:
            # 标签不存在，但模型错误预测出了该类别
            recall = 0.0
    else:
        recall = true_positive / recall_denominator

    # 预测中不存在当前类别时，Precision 分母为 0
    if precision_denominator == 0:
        # 标签中同样不存在该类别
        if recall_denominator == 0:
            precision = 1.0
        else:
            # 标签存在，但模型完全没有预测出来
            precision = 0.0
    else:
        precision = true_positive / precision_denominator

    return iou, recall, precision


def print_model_parameters(model: nn.Module) -> None:
    """
    输出模型总参数量、可训练参数量和冻结参数量。
    单位：M，1 M = 1,000,000个参数。
    """
    total_params = sum(p.numel() for p in model.parameters())

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    frozen_params = total_params - trainable_params

    trainable_ratio = (
        trainable_params / total_params * 100
        if total_params > 0
        else 0.0
    )

    print(
        "Model parameters | "
        "Total: %.3f M | "
        "Trainable: %.3f M | "
        "Frozen: %.3f M | "
        "Trainable ratio: %.2f%%"
        %(total_params / 1e6,
        trainable_params / 1e6,
        frozen_params / 1e6,
        trainable_ratio)
    )