import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from Architecture.CMAFRTransUnet import CMAFR_Transformer as ViT_seg
from Architecture.CMAFRTransUnet import CONFIGS as CONFIGS_ViT_seg
import sys
import time
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='./datasets/Synapse/train_npz', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=12, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()


def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    # test Dataset
    # sample = db_train[5]

    print("The length of train set is: {}".format(len(db_train)))#2211

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             )#worker_init_fn=worker_init_fn
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
#             print(image_batch)
            # Deep-supervision model returns:
            # outputs = [logit_D0, logit_D1, logit_D2, logit_D3, logit_D4]
            # Each tensor has shape [B, num_classes, 224, 224].
            outputs = model(image_batch)

            if not isinstance(outputs, (list, tuple)):
                raise TypeError(
                    "Deep-supervision CMAFR-TransUnet should return a list/tuple "
                    "of [D0, D1, D2, D3, D4] logits."
                )
            if len(outputs) != 5:
                raise ValueError(
                    "Expected 5 deep-supervision outputs [D0,D1,D2,D3,D4], "
                    "but got {}.".format(len(outputs))
                )

            # Deep-supervision ablation W1: remove D0 supervision, keep D1-D4 equally weighted.
            # Li = 0.5 * CE_i + 0.5 * Dice_i
            # Keep the original total weight sum (=5) unchanged for a clean comparison:
            # weights [D0,D1,D2,D3,D4] = [0, 1.25, 1.25, 1.25, 1.25].
            stage_ce_losses = []
            stage_dice_losses = []
            stage_losses = []

            for pred in outputs:
                ce_i = ce_loss(pred, label_batch.long())
                dice_i = dice_loss(pred, label_batch, softmax=True)
                stage_ce_losses.append(ce_i)
                stage_dice_losses.append(dice_i)
                stage_losses.append(0.5 * ce_i + 0.5 * dice_i)

            # 完全关闭辅助深监督，仅 D4
            # stage_weights = [0, 0, 0, 0, 1]
            stage_weights = [1, 1, 1, 1, 1]
            loss = sum(w * loss_i for w, loss_i in zip(stage_weights, stage_losses))
            loss_ce = sum(w * ce_i for w, ce_i in zip(stage_weights, stage_ce_losses))
            loss_dice = sum(w * dice_i for w, dice_i in zip(stage_weights, stage_dice_losses))

            # D4 is the original baseline final output and is used for
            # visualization/inference-style monitoring during training.
            main_output = outputs[-1]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss.item(), iter_num)
            writer.add_scalar('info/loss_ce_sum', loss_ce.item(), iter_num)
            writer.add_scalar('info/loss_dice_sum', loss_dice.item(), iter_num)

            # Record every stage separately so we can later judge which
            # auxiliary level is difficult or harmful.
            stage_names = ['D0', 'D1', 'D2', 'D3', 'D4']
            for name, ce_i, dice_i, loss_i in zip(
                    stage_names, stage_ce_losses, stage_dice_losses, stage_losses):
                writer.add_scalar('deep_supervision/{}/loss'.format(name), loss_i.item(), iter_num)
                writer.add_scalar('deep_supervision/{}/ce'.format(name), ce_i.item(), iter_num)
                writer.add_scalar('deep_supervision/{}/dice_loss'.format(name), dice_i.item(), iter_num)

            logging.info(
                'iteration %d : total_loss: %f, '
                'D0: %.4f, D1: %.4f, D2: %.4f, D3: %.4f, D4(main): %.4f'
                % (
                    iter_num,
                    loss.item(),
                    stage_losses[0].item(),
                    stage_losses[1].item(),
                    stage_losses[2].item(),
                    stage_losses[3].item(),
                    stage_losses[4].item(),
                )
            )

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())#归一化
                writer.add_image('train/Image', image, iter_num)

                # Only D4 is the actual final prediction.
                pred_vis = torch.argmax(
                    torch.softmax(main_output, dim=1),
                    dim=1,
                    keepdim=True
                )
                writer.add_image('train/Prediction', pred_vis[1, ...] * 50, iter_num)#(B,1,h,w)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        save_interval = 10  # int(max_epoch/6)
        if epoch_num > 100 and ((epoch_num + 1) % save_interval == 0 or (epoch_num + 1) % 30 == 0):
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break


    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset

    dataset_config = {
        'Synapse': {
            'root_path': '/hy-tmp/Synapse/train_npz',
            'list_dir': './lists/lists_Synapse',
            'num_classes': 9,
        },
    }
#     if args.batch_size != 24 and args.batch_size % 6 == 0:
#         args.base_lr *= args.batch_size / 24
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    snapshot_path = "../model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)# '../model/TU_Synapse224/TU_pretrain_R50-ViT-B_16_skip3_epo120_bs12_224'
    config_vit = CONFIGS_ViT_seg[args.vit_name]#R50-VIT-B_16
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:# grid:(14,14)
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    net.load_from(weights=np.load(config_vit.pretrained_path))

    trainer = {'Synapse': trainer_synapse,}
    trainer[dataset_name](args, net, snapshot_path)
