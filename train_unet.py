from __future__ import print_function

import os
import sys
import importlib
import shutil
import json
import numpy as np
import time
import requests
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import roc_curve, auc

root_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, "datasets"))
sys.path.append(os.path.join(root_dir, "models"))
sys.path.append(os.path.join(root_dir, "optim"))

from datasets.dataset import er_data_loader
from models.utils import init_weights
from models.unet_ablation import UNet, UNet1, UNet2, UNet3
from models.optimize import create_criterion, create_optimizer, update_learning_rate
from datasets.metric import *

print("PyTorch Version: ", torch.__version__)


def FrobeniusNorm(input):  # [b,c,h,w]
    b, c, h, w = input.size()
    triu = torch.eye(h).cuda()
    triu = triu.unsqueeze(0).unsqueeze(0)
    triu = triu.repeat(b, c, 1, 1)

    x = torch.matmul(input, input.transpose(-2, -1))
    tr = torch.mul(x, triu)
    y = torch.sum(tr)
    return y


def print_table(data):
    col_width = [max(len(item) for item in col) for col in data]
    for row_idx in range(len(data[0])):
        for col_idx, col in enumerate(data):
            item = col[row_idx]
            align = '<' if not col_idx == 0 else '>'
            print(('{:' + align + str(col_width[col_idx]) + '}').format(item), end=" ")
        print()


def gmm_loss(label, prd, mu_f, mu_b, std_f, std_b, f_k):
    b_k = 1 - f_k

    f_likelihood = - f_k * (
            torch.log(np.sqrt(2 * 3.14) * std_f) + torch.pow((prd - mu_f), 2) / (2 * torch.pow(std_f, 2)) + 1e-10)
    b_likelihood = - b_k * (
            torch.log(np.sqrt(2 * 3.14) * std_b) + torch.pow((prd - mu_b), 2) / (2 * torch.pow(std_b, 2)) + 1e-10)
    likelihood = f_likelihood + b_likelihood
    loss = torch.mean(torch.pow(label - torch.exp(likelihood), 2))
    return loss


def train_one_epoch(epoch, total_steps, dataloader, model,
                    device, criterion, optimizer, lr, lr_decay,
                    display_iter, log_file, show):
    model.train()

    smooth_loss = 0.0
    current_step = 0
    t0 = 0.0

    for inputs in dataloader:

        t1 = time.time()

        images = inputs['image'].to(device)
        # c_images = inputs['c_masks'].to(device)
        labels = inputs['mask'].to(device)
        # print(inputs["ID"])

        # forward pass
        pred = model(images)

        # compute loss
        loss = criterion(pred, labels)

        # predictions
        t0 += (time.time() - t1)

        total_steps += 1
        current_step += 1
        smooth_loss += loss.item()

        # backpropagate when training
        optimizer.zero_grad()
        lr_update = update_learning_rate(optimizer, epoch, lr, step=lr_decay)
        loss.backward()
        # loss.backward(retain_graph = True)
        optimizer.step()

        if show:
            images = images[0, 0, :, :].data.cpu().numpy()
            # reconstruct_images = pred[0, 0, :, :].data.cpu()
            # dff_imgs = diff[0, 0, :, :].data.cpu()
            raw_mask = labels[0, 0, :, :].data.cpu().numpy() * 255
            # c_mask = incorrect_pred[0, 0, :, :].data.cpu()
            raw_mask = raw_mask.astype(np.uint8)
            prd_score = pred[0, 0, :, :].data.cpu().numpy()
            # c_prd_score = correct_pred[0, 0, :, :].data.cpu().numpy()
            prd_mask = prd_score > 0.5

            # writer.add_image('raw_img', images, dataformats='HW')
            # writer.add_image('raw_mask', raw_mask, dataformats='HW')
            # writer.add_image('prd', prd_mask, dataformats='HW')

        # log loss
        if total_steps % display_iter == 0:
            smooth_loss = smooth_loss / current_step
            message = "Epoch: %d Step: %d LR: %.6f Loss: %.4f Runtime: %.2fs/%diters." % (
                epoch + 1, total_steps, lr_update, smooth_loss, t0, display_iter)
            print("==> %s" % (message))
            with open(log_file, "a+") as fid:
                fid.write('%s\n' % message)

            t0 = 0.0
            current_step = 0
            smooth_loss = 0.0

    return total_steps


def eval_one_epoch(epoch, threshold, dataloader, model, device, log_file):
    with torch.no_grad():
        model.eval()

        total_iou = 0.0
        total_f1 = 0.0
        # total_distance = 0.0
        total_acc = 0.0
        total_img = 0

        for inputs in dataloader:
            images = inputs['image'].to(device)
            labels = inputs['mask']

            total_img += len(images)
            outputs = model(images)
            preds = outputs > threshold
            preds = preds.cpu()

            # metric
            val_acc = acc(preds, labels)

            total_acc += val_acc

            val_iou = IoU(preds, labels)
            total_iou += val_iou

            val_f1 = F1_score(preds, labels)
            total_f1 += val_f1

        # iou
        epoch_iou = total_iou / total_img
        epoch_f1 = total_f1 / total_img
        epoch_acc = total_acc / total_img

        message = "total Threshold: {:.3f} =====> Evaluation IOU: {:.4f}; F1_score: {:.4f}; Acc: {:.4f}".format(
            threshold, epoch_iou, epoch_f1, epoch_acc)
        print("==> %s" % (message))
        with open(log_file, "a+") as fid:
            fid.write('%s\n' % message)

    return epoch_acc, epoch_iou, epoch_f1


def train_eval_model(opts):
    # parse model configuration
    num_epochs = opts["num_epochs"]
    train_batch_size = opts["train_batch_size"]
    val_batch_size = opts["eval_batch_size"]
    dataset_type = opts["dataset_type"]

    opti_mode = opts["optimizer"]
    loss_criterion = opts["loss_criterion"]
    lr = opts["lr"]
    lr_decay = opts["lr_decay"]
    wd = opts["weight_decay"]

    gpus = opts["gpu_list"].split(',')
    os.environ['CUDA_VISIBLE_DEVICE'] = opts["gpu_list"]
    train_dir = opts["log_dir"]

    train_data_dir = opts["train_data_dir"]
    eval_data_dir = opts["eval_data_dir"]

    pretrained = opts["pretrained_model"]
    resume = opts["resume"]
    display_iter = opts["display_iter"]
    save_epoch = opts["save_every_epoch"]
    show = opts["vis"]

    # backup train configs
    log_file = os.path.join(train_dir, "log_file.txt")
    os.makedirs(train_dir, exist_ok=True)
    model_dir = os.path.join(train_dir, "code_backup")
    os.makedirs(model_dir, exist_ok=True)
    if resume is None and os.path.exists(log_file): os.remove(log_file)
    shutil.copy("./models/unet.py", os.path.join(model_dir, "unet.py"))
    shutil.copy("./train_unet.py", os.path.join(model_dir, "train_unet.py"))
    shutil.copy("./datasets/dataset.py", os.path.join(model_dir, "dataset.py"))

    ckt_dir = os.path.join(train_dir, "checkpoints")
    os.makedirs(ckt_dir, exist_ok=True)

    # format printing configs
    print("*" * 50)
    table_key = []
    table_value = []
    n = 0
    for key, value in opts.items():
        table_key.append(key)
        table_value.append(str(value))
        n += 1
    print_table([table_key, ["="] * n, table_value])

    # format gpu list
    gpu_list = []
    for str_id in gpus:
        id = int(str_id)
        gpu_list.append(id)

    # dataloader
    print("==> Create dataloader")
    dataloaders_dict = {"train": er_data_loader(train_data_dir, train_batch_size, dataset_type, is_train=True),
                        "eval": er_data_loader(eval_data_dir, val_batch_size, dataset_type, is_train=False)}

    # define parameters of two networks
    print("==> Create network")
    num_channels = 1
    num_classes = 1
    down_num = opts["down_num"]
    if down_num == 1:
        model = UNet1(num_channels, num_classes)
    elif down_num == 2:
        model = UNet2(num_channels, num_classes)
    elif down_num == 3:
        model = UNet3(num_channels, num_classes)
    elif down_num == 4:
        model = UNet(num_channels, num_classes)

    init_weights(model)

    # loss layer
    criterion = create_criterion(criterion=loss_criterion)

    best_acc = 0.0
    start_epoch = 0

    # load pretrained model
    if pretrained is not None and os.path.isfile(pretrained):
        print("==> Train from model '{}'".format(pretrained))
        checkpoint_gan = torch.load(pretrained)
        model.load_state_dict(checkpoint_gan['model_state_dict'])
        print("==> Loaded checkpoint '{}')".format(pretrained))
        for param in model.parameters():
            param.requires_grad = False

    # resume training
    elif resume is not None and os.path.isfile(resume):
        print("==> Resume from checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if
                           k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict)
        print("==> Loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch'] + 1))

    # train from scratch
    else:
        print("==> Train from initial or random state.")

    # define mutiple-gpu mode
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.cuda()
    model = nn.DataParallel(model)

    # print learnable parameters
    print("==> List learnable parameters")
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            print("\t{}, size {}".format(name, param.size()))
    params_to_update = [{'params': model.parameters()}]

    # define optimizer
    print("==> Create optimizer")
    optimizer = create_optimizer(params_to_update, opti_mode, lr=lr, momentum=0.9, wd=wd)
    if resume is not None and os.path.isfile(resume): optimizer.load_state_dict(checkpoint['optimizer'])

    # start training
    since = time.time()

    # Each epoch has a training and validation phase
    print("==> Start training")
    total_steps = 0
    threshold = opts["threshold"]

    for epoch in range(start_epoch, num_epochs):

        print('-' * 50)
        print("==> Epoch {}/{}".format(epoch + 1, num_epochs))

        total_steps = train_one_epoch(epoch, total_steps,
                                      dataloaders_dict['train'],
                                      model, device,
                                      criterion, optimizer, lr, lr_decay,
                                      display_iter, log_file, show)

        epoch_acc, epoch_iou, epoch_f1 = eval_one_epoch(epoch, threshold, dataloaders_dict['eval'],
                                                        model, device, log_file)

        if best_acc < epoch_acc and epoch >= 5:
            best_acc = epoch_acc
            torch.save({'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_acc': best_acc},
                       os.path.join(ckt_dir, "best.pth"))

        if (epoch + 1) % save_epoch == 0 and (epoch + 1) >= 20:
            torch.save({'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_iou': epoch_iou},
                       os.path.join(ckt_dir, "checkpoints_" + str(epoch + 1) + ".pth"))

    time_elapsed = time.time() - since
    time_message = 'Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)
    print(time_message)
    with open(log_file, "a+") as fid:
        fid.write('%s\n' % time_message)
    print('==> Best val Acc: {:4f}'.format(best_acc))


if __name__ == '__main__':
    date = '20221214'
    dataset_list = ['er', 'retina', 'mito', 'stare']
    model_choice = ['unet', 'unet_multilabel', 'unet_singlelabel']
    txt_choice = ['test_drive.txt', 'train_drive.txt', 'train_mito.txt', 'test_mito_cbmi.txt', 'train_er.txt',
                  'test_er.txt', 'test_stare.txt', 'train_stare.txt']

    opts = dict()
    opts['dataset_type'] = 'stare'
    opts["num_epochs"] = 30
    opts["train_data_dir"] = "/mnt/data1/hjx_data/dataset_txts/train_stare.txt"
    opts["eval_data_dir"] = "/mnt/data1/hjx_data/dataset_txts/test_stare.txt"
    opts["train_batch_size"] = 16
    opts["eval_batch_size"] = 32
    opts["optimizer"] = "SGD"
    opts["loss_criterion"] = "iou"
    opts["lr"] = 0.085
    opts["threshold"] = 0.4
    opts["lr_decay"] = 10
    opts["down_num"] = 4
    opts["weight_decay"] = 0.0005
    opts["gpu_list"] = "0,1,2,3"
    log_dir = "./train_log/" + str(opts["dataset_type"]) + "_train_Unet_" + str(
        opts["down_num"]) + '_' + str(opts["lr"]) + '_' + str(
        opts["num_epochs"]) + '_' + str(opts["threshold"]) + '_' + date + '_' + str(opts["weight_decay"]) + '_iouloss'
    opts["log_dir"] = log_dir
    opts["pretrained_model"] = None
    opts["resume"] = None
    opts["display_iter"] = 10
    opts["save_every_epoch"] = 5
    opts["vis"] = True

    train_eval_model(opts)
