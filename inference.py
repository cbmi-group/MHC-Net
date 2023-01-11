from __future__ import print_function

import os
import numpy as np
import torch
from datasets.dataset import er_data_loader
from models.unet import UNet as u_net
from models.nested_unet import NestedUNet as u_net_plus
from models.mhc_net import MHCnet_Single_Loss, MHCnet_Up_Sampling_Loss, MHCnet_Multi_Layer_Loss, \
    MHCnet_Hierarchical_Fusing_Loss
from models.deeplab_v3 import DeepLab
from models.hrnet import HRNetV2
from datasets.metric import *
import cv2
import torch.nn.functional as F
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score, confusion_matrix

print("PyTorch Version: ", torch.__version__)

'''
evaluation
'''


def random_crop(images, labels):
    now_size1 = images.shape[2]
    now_size2 = images.shape[3]
    aim_size1 = int(now_size1 / 2)
    aim_size2 = int(now_size2 / 2)
    trans = transforms.Compose([transforms.RandomCrop((aim_size1, aim_size2))])
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
    cropped_labels.append(labels1)
    cropped_imgs.append(images2)
    cropped_labels.append(labels2)
    cropped_imgs.append(images3)
    cropped_labels.append(labels3)
    cropped_imgs.append(images4)
    cropped_labels.append(labels4)
    return cropped_imgs, cropped_labels


def eval_model(opts, index):
    val_batch_size = opts["eval_batch_size"]
    dataset_type = opts['dataset_type']
    load_epoch = opts['load_epoch']
    gpus = opts["gpu_list"].split(',')
    gpu_list = []
    for str_id in gpus:
        id = int(str_id)
        gpu_list.append(id)
    os.environ['CUDA_VISIBLE_DEVICE'] = opts["gpu_list"]

    eval_data_dir = opts["eval_data_dir"]
    dataset_name = os.path.split(eval_data_dir)[-1].split('.')[0]

    train_dir = opts["train_dir"]
    model_type = opts['model_type']

    model_score_dir = os.path.join(str(os.path.split(train_dir)[0]),
                                   'predict_score/' + dataset_name + '_' + str(load_epoch))
    if not os.path.exists(model_score_dir): os.makedirs(model_score_dir)

    # dataloader
    print("==> Create dataloader")
    dataloader = er_data_loader(eval_data_dir, val_batch_size, dataset_type, is_train=False)

    # define network
    print("==> Create network")

    model = None

    if model_type == 'unet':
        model = u_net(1, 1)
    elif model_type == "MHCnet_Single_Loss":
        model = MHCnet_Single_Loss(1, 1)
    elif model_type == "MHCnet_Up_Sampling_Loss":
        model = MHCnet_Up_Sampling_Loss(1, 1)
    elif model_type == "MHCnet_Multi_Layer_Loss":
        model = MHCnet_Multi_Layer_Loss(1, 1)
    elif model_type == "MHCnet_Hierarchical_Fusing_Loss":
        model = MHCnet_Hierarchical_Fusing_Loss(1, 1)
    elif model_type == 'unetPlus':
        model = u_net_plus(1, 1)
    elif model_type == 'deeplab':
        model = DeepLab(backbone='resnet50', output_stride=16)
    elif model_type == 'hrnet':
        model = HRNetV2(1)

    # load trained model
    pretrain_model = os.path.join(train_dir, str(load_epoch) + ".pth")
    # print(pretrain_model)
    # pretrain_model = os.path.join(train_dir, "checkpoints_" + str(load_epoch) + ".pth")

    if os.path.isfile(pretrain_model):
        c_checkpoint = torch.load(pretrain_model)
        model.load_state_dict(c_checkpoint["model_state_dict"])
        print("==> Loaded pretrianed model checkpoint '{}'.".format(pretrain_model))
    else:
        print("==> No trained model.")
        return 0

    # set model to gpu mode
    print("==> Set to GPU mode")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model = torch.nn.DataParallel(model, device_ids=gpu_list)
    thresholds = np.arange(0.15, 0.6, 0.05)
    best_iou = 0.00
    # enable evaluation mode
    with torch.no_grad():
        model.eval()

        for threshold in thresholds:
            total_img = 0
            for inputs in dataloader:
                images = inputs["image"].cuda()
                labels = inputs['mask']

                cropped_imgs, cropped_labels = pre_process_il(images, labels)
                img_name = inputs['ID']
                total_img += len(images)

                p_seg = 0
                # unet
                if model_type == 'unet':
                    p_seg = model(images)
                elif model_type == 'MHCnet_Up_Sampling_Loss':
                    p_seg = model(images, cropped_imgs)
                    p_seg = p_seg[0]
                elif model_type == 'MHCnet_Multi_Layer_Loss':
                    p_seg = model(images, cropped_imgs)
                    p_seg = p_seg[0]
                elif model_type == 'MHCnet_Single_Loss':
                    p_seg = model(images, cropped_imgs)
                elif model_type == 'MHCnet_Hierarchical_Fusing_Loss':
                    p_seg = model(images, cropped_imgs)
                    p_seg = p_seg[0]
                elif model_type == 'unetPlus':
                    p_seg = model(images)
                    p_seg = p_seg[-1]
                elif model_type == 'hrnet':
                    outputs_list = model(images)
                    p_seg = outputs_list[0]
                elif model_type == 'deeplab':
                    p_seg = model(images)

                for i in range(len(images)):
                    # print('predict image: {}'.format(img_name[i]))
                    if model_type == 'agnet':
                        np.save(os.path.join(model_score_dir, img_name[i].split('.')[0] + '.npy'),
                                p_seg[i][1].cpu().numpy().astype(np.float32))
                        cv2.imwrite(os.path.join(model_score_dir, img_name[i]),
                                    p_seg[i][1].cpu().numpy().astype(np.float32))
                    else:
                        now_dir = model_score_dir + '_' + str(index)
                        os.makedirs(now_dir, exist_ok=True)
                        np.save(os.path.join(now_dir, img_name[i].split('.')[0] + '.npy'),
                                p_seg[i][0].cpu().numpy().astype(np.float32))
                        cv2.imwrite(os.path.join(now_dir, img_name[i].split('.')[0] + '.tif'),
                                    p_seg[i][0].cpu().numpy().astype(np.float32))
                y_scores = p_seg.cpu().numpy().flatten()
                y_pred = y_scores > threshold
                y_true = labels.numpy().flatten()

                # total_distance = 0.0

                confusion = confusion_matrix(y_true, y_pred)
                tp = float(confusion[1, 1])
                fn = float(confusion[1, 0])
                fp = float(confusion[0, 1])
                tn = float(confusion[0, 0])

                val_acc = (tp + tn) / (tp + fn + fp + tn)
                sensitivity = tp / (tp + fn)
                specificity = tn / (tn + fp)
                precision = tp / (tp + fp)
                f1 = 2 * sensitivity * precision / (sensitivity + precision)
                iou = tp / (tp + fn + fp)
                auc = roc_auc_score(y_true, y_scores)

            epoch_iou = iou
            if epoch_iou > best_iou:
                best_iou = epoch_iou
            epoch_f1 = f1
            epoch_acc = val_acc
            epoch_auc = auc
            epoch_sen = sensitivity
            epoch_spec = specificity
            message = "inference  =====> Evaluation  ACC: {:.4f}; IOU: {:.4f}; F1_score: {:.4f}; Auc: {:.4f} ;Sen: {:.4f}; Spec: {:.4f}; threshold: {:.4f}".format(
                epoch_acc,
                epoch_iou,
                epoch_f1, epoch_auc, epoch_sen, epoch_spec, threshold)
            print("==> %s" % (message))

        print("validation image number {}".format(total_img))
    return best_iou


if __name__ == "__main__":
    model_choice = ['unet', 'unetPlus', 'deeplab', 'hrnet', 'MHCnet_Single_Loss', 'MHCnet_Up_Sampling_Loss',
                    'MHCnet_Multi_Layer_Loss',
                    'MHCnet_Hierarchical_Fusing_Loss']
    dataset_list = ['er', 'retina', 'mito', 'stare']

    txt_choice = ['test_drive.txt', 'train_drive.txt', 'train_mito.txt', 'test_mito_cbmi.txt', 'train_er.txt',
                  'test_er.txt', 'test_stare.txt', 'train_stare.txt']
    size_choice = [80, 10, 38, 40]
    opts = dict()
    opts['dataset_type'] = 'stare'
    opts["eval_batch_size"] = 40
    opts["gpu_list"] = "0,1,2,3"
    opts[
        "train_dir"] = "/mnt/data1/hjx_code/MHCNet_code/train_log/stare_train_MHCnet_Single_Loss_iouloss_16_0.15_10_0.3_20_20221221_0.0005_iouloss_warmup/checkpoints"
    opts["eval_data_dir"] = "/mnt/data1/hjx_data/dataset_txts/test_stare.txt"

    opts['model_type'] = 'MHCnet_Single_Loss'
    opts["load_epoch"] = 'best'

    best_iou = 0.0
    for i in range(10):
        print('************now time//' + str(i))
        now_iou = eval_model(opts, i)
        if now_iou > best_iou:
            best_iou = now_iou
    print('final best iou =' + str(best_iou))
