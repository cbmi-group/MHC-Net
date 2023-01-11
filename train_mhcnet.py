from __future__ import print_function

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
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
import matplotlib.pyplot as plt

root_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, "datasets"))
sys.path.append(os.path.join(root_dir, "models"))
sys.path.append(os.path.join(root_dir, "optim"))

from datasets.dataset import er_data_loader
from models.utils import init_weights
from models.unet import UNet
from models.mhc_net import MHCnet_Single_Loss, MHCnet_Up_Sampling_Loss, MHCnet_Multi_Layer_Loss, \
    MHCnet_Hierarchical_Fusing_Loss
from models.optimize import create_criterion, create_optimizer, update_learning_rate, warmup_learning_rate
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


def random_crop(images, labels):
    now_size = images.shape[2]
    aim_size = now_size / 2
    trans = transforms.Compose([transforms.RandomCrop(aim_size)])
    seed = torch.random.seed()
    torch.random.manual_seed(seed)
    cropped_img = trans(images)
    torch.random.manual_seed(seed)
    cropped_label = trans(labels)
    return cropped_img, cropped_label


def pre_process_il(images, labels):
    # level crop
    cropped_imgs = []
    cropped_labels = []

    images1, labels1 = random_crop(images, labels)
    images2, labels2 = random_crop(images1, labels1)
    images3, labels3 = random_crop(images2, labels2)
    images4, labels4 = random_crop(images3, labels3)

    cropped_imgs.append(images1)
    cropped_labels.append(labels1.contiguous())
    cropped_imgs.append(images2)
    cropped_labels.append(labels2.contiguous())
    cropped_imgs.append(images3)
    cropped_labels.append(labels3.contiguous())
    cropped_imgs.append(images4)
    cropped_labels.append(labels4.contiguous())
    return cropped_imgs, cropped_labels


def loss_compute(criterion, pred, labels, cropped_labels, model_type):
    if model_type == "MHCnet_Single_Loss":
        loss = criterion(pred, labels)
    elif model_type == "MHCnet_Up_Sampling_Loss":
        loss1 = criterion(pred[0], labels)
        loss2 = criterion(pred[1], labels)
        loss3 = criterion(pred[2], labels)
        loss4 = criterion(pred[3], labels)
        loss5 = criterion(pred[4], labels)
        loss = loss1 * 0.5 + (loss2 + loss3 + loss4 + loss5) / 8
    elif model_type == "MHCnet_Multi_Layer_Loss":
        loss1 = criterion(pred[0], labels)
        loss2 = criterion(pred[1], cropped_labels[0])
        loss3 = criterion(pred[2], cropped_labels[1])
        loss4 = criterion(pred[3], cropped_labels[2])
        loss5 = criterion(pred[4], cropped_labels[3])
        loss = loss1 * 0.5 + (loss2 + loss3 + loss4 + loss5) / 8
    elif model_type == "MHCnet_Hierarchical_Fusing_Loss":
        loss1 = criterion(pred[0], labels)
        loss2 = criterion(pred[1], labels)
        loss3 = criterion(pred[2], labels)
        loss4 = criterion(pred[3], labels)
        loss5 = criterion(pred[4], labels)
        loss6 = criterion(pred[5], cropped_labels[0])
        loss7 = criterion(pred[6], cropped_labels[1])
        loss8 = criterion(pred[7], cropped_labels[2])
        loss9 = criterion(pred[8], cropped_labels[3])
        loss = loss1 * 0.5 + (loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8 + loss9) / 16
    else:
        print("model type error!")
    return loss


def train_one_epoch(epoch, model_type, total_steps, dataloader, model,
                    device, criterion, optimizer, lr, lr_decay,
                    display_iter, log_file, warmup_step, warmup_method):
    model.train()

    smooth_loss = 0.0
    current_step = 0
    t0 = 0.0

    for inputs in dataloader:

        t1 = time.time()

        images = inputs['image'].to(device)
        labels = inputs['mask'].to(device)

        cropped_imgs, cropped_labels = pre_process_il(images, labels)
        # forward pass
        pred = model(images, cropped_imgs)

        # compute loss
        loss = loss_compute(criterion, pred, labels, cropped_labels, model_type)
        # predictions
        t0 += (time.time() - t1)

        total_steps += 1
        current_step += 1
        smooth_loss += loss.item()

        # back-propagate when training
        optimizer.zero_grad()

        lr_update = warmup_learning_rate(optimizer, total_steps, warmup_step, lr, warmup_method)
        # lr_update = update_learning_rate(optimizer, epoch, lr, step=lr_decay)

        loss.backward()
        optimizer.step()

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


def eval_one_epoch(epoch, model_type, threshold, dataloader, model, device, log_file):
    with torch.no_grad():
        model.eval()

        total_iou = 0.0
        total_f1 = 0.0
        total_acc = 0.0
        total_img = 0

        for inputs in dataloader:
            images = inputs['image'].to(device)
            labels = inputs['mask']

            total_img += len(images)
            cropped_images, cropped_labels = pre_process_il(images, labels)
            outputs = model(images, cropped_images)

            if model_type == "MHCnet_Single_Loss":
                preds = outputs > threshold
            else:
                preds = outputs[0] > threshold
            preds = preds.cpu()

            # metric
            val_acc = acc(preds, labels)
            total_acc += val_acc

            val_iou = IoU(preds, labels)
            total_iou += val_iou

            val_f1 = F1_score(preds, labels)
            total_f1 += val_f1

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
    model_type = opts["model_type"]

    opti_mode = opts["optimizer"]
    loss_criterion = opts["loss_criterion"]
    warmup_step = opts["warmup_step"]
    warmup_method = opts["warmup_method"]
    wd = opts["weight_decay"]
    lr = opts["lr"]
    lr_decay = opts["lr_decay"]

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
    shutil.copy("./models/mhc_net.py", os.path.join(model_dir, "mhc_net.py"))
    shutil.copy("./train_mhcnet.py", os.path.join(model_dir, "train_mhcnet.py"))
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
    if model_type == "MHCnet_Single_Loss":
        model = MHCnet_Single_Loss(num_channels, num_classes)
    elif model_type == "MHCnet_Up_Sampling_Loss":
        model = MHCnet_Up_Sampling_Loss(num_channels, num_classes)
    elif model_type == "MHCnet_Multi_Layer_Loss":
        model = MHCnet_Multi_Layer_Loss(num_channels, num_classes)
    elif model_type == "MHCnet_Hierarchical_Fusing_Loss":
        model = MHCnet_Hierarchical_Fusing_Loss(num_channels, num_classes)
    else:
        print("model type error!")

    init_weights(model)

    # loss layer
    criterion = create_criterion(criterion=loss_criterion)

    best_acc = 0.0
    best_iou = 0.0
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
    epochs = []
    ious = []
    for epoch in range(start_epoch, num_epochs):
        epochs.append(epoch)
        print('-' * 50)
        print("==> Epoch {}/{}".format(epoch + 1, num_epochs))

        total_steps = train_one_epoch(epoch, model_type, total_steps,
                                      dataloaders_dict['train'],
                                      model, device,
                                      criterion, optimizer, lr, lr_decay,
                                      display_iter, log_file, warmup_step, warmup_method)

        epoch_acc, epoch_iou, epoch_f1 = eval_one_epoch(epoch, model_type, threshold, dataloaders_dict['eval'],
                                                        model, device, log_file)
        ious.append(epoch_iou)
        if best_acc < epoch_acc and epoch >= 5:
            best_acc = epoch_acc
            best_iou = epoch_iou
            torch.save({'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_acc': best_acc},
                       os.path.join(ckt_dir, "best.pth"))

        if (epoch + 1) % save_epoch == 0 and (epoch + 1) >= 20:
            torch.save({'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_acc': epoch_acc},
                       os.path.join(ckt_dir, "checkpoints_" + str(epoch + 1) + ".pth"))

    time_elapsed = time.time() - since
    time_message = 'Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)
    # define size of image
    plt.figure(figsize=(10, 10))
    plt.plot(epochs, ious)
    plt.ylim(0, 0.9)
    # set the label of x and y
    plt.xlabel("epoch")
    plt.ylabel("iou")
    plt.title("Train model= " + str(model_type) + "; lr=" + str(lr) + '; warmup step=' + str(
        warmup_step) + '; warmup method=' + str(
        warmup_method))
    plt.legend()

    plt.savefig(os.path.join(train_dir, 'lr_' + str(lr) + '_warmup_step_' + str(warmup_step) + '_train_iou.png'))
    print(time_message)
    with open(log_file, "a+") as fid:
        fid.write('%s\n' % time_message)
    print('==> Best val Acc: {:4f}; Iou: {:4f}'.format(best_acc, best_iou))


if __name__ == '__main__':
    dataset_list = ['er', 'retina', 'mito', 'stare']
    # MHCnet with different loss
    model_choice = ['MHCnet_Single_Loss', 'MHCnet_Up_Sampling_Loss', 'MHCnet_Multi_Layer_Loss',
                    'MHCnet_Hierarchical_Fusing_Loss']
    txt_choice = ['test_drive.txt', 'train_drive.txt', 'train_mito.txt', 'test_mito_cbmi.txt', 'train_er.txt',
                  'test_er.txt', 'test_stare.txt', 'train_stare.txt']
    date = '20221221'

    opts = dict()
    opts['dataset_type'] = 'stare'
    opts['model_type'] = 'MHCnet_Hierarchical_Fusing_Loss'
    opts["num_epochs"] = 40
    opts["train_data_dir"] = "/mnt/data1/hjx_data/dataset_txts/train_stare.txt"
    opts["eval_data_dir"] = "/mnt/data1/hjx_data/dataset_txts/test_stare.txt"
    opts["train_batch_size"] = 16
    opts["eval_batch_size"] = 32
    opts["optimizer"] = "SGD"
    opts["loss_criterion"] = "iou"
    opts["lr"] = 0.15
    opts["threshold"] = 0.3
    opts["warmup_step"] = 20
    opts["warmup_method"] = 'exp'
    opts["lr_decay"] = 5
    opts["weight_decay"] = 0.0005
    opts["gpu_list"] = "0,1,2,3"
    log_dir = "./train_log/" + str(opts["dataset_type"]) + "_train_" + str(opts["model_type"]) + "_iouloss_" + str(
        opts["train_batch_size"]) + '_' + str(opts["lr"]) + '_' + str(
        opts["num_epochs"]) + '_' + str(opts["threshold"]) + '_' + str(
        opts["warmup_step"]) + '_' + date + '_' + str(opts["weight_decay"]) + '_iouloss_warmup'
    opts["log_dir"] = log_dir
    opts["pretrained_model"] = None
    opts["resume"] = None
    opts["display_iter"] = 10
    opts["save_every_epoch"] = 5
    opts["vis"] = False

    train_eval_model(opts)
