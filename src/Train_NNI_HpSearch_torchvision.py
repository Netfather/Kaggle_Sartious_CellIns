# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         Train_NNI_HpSearch
# Description:  此文件用于使用NNI进行超参数搜索，
# 2021年10月21日  V1： #1 搜索 针对 TRCUNet_256特化版本进行搜索
# Author:       Administrator
# Date:         2021/10/21
# -------------------------------------------------------------------------------

import os
import cv2
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as T
# 日志初始化
from shutil import copyfile  # 用于复制文件
from datetime import date
import albumentations as A
from sklearn.model_selection import StratifiedKFold
import timm
import gc
from timeit import default_timer as timer
import json
from tqdm.notebook import tqdm
import collections
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.model_selection import train_test_split
import time
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
# import seaborn as sns


from toolbox.log_writers.log import get_logger
import toolbox.loss_box.binaray_loss as loss_tool
from toolbox.learning_schdule_box.pytorch_cosin_warmup import CosineAnnealingWarmupRestarts
from toolbox.optimizer_box.radam import RAdam
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import logging

import nni
from nni.utils import merge_parameter


load_pth  = None

device = 'cuda' if torch.cuda.is_available() else 'cpu'
manual_seed = 3407  #固定住每一次的seed
_logger = logging.getLogger('CellInstance_ML')

total_epoch = 25
cycle_epoch = 25 # 用于调节重启次数  由于是迁移学习 轮数较少 与总epoch保持一致
IMAGENET_MEAN = [0.485, 0.456, 0.406]  # RGB
IMAGENET_STD = [0.229, 0.224, 0.225]  # RGB


hyper_parameter_group ={
    # 主要环境参数设定
    "seed": 3407,
    "out_dir": r'../model/MaskRNN/test2',
    "log_name_head": 'MaskRNN_test2',
    "Open_Parral_Training": False,
    "initial_checkpoint": None,
    "no_log": False,   # 注意这个参数只在 NNI_SEARCH中使用 其他时候为False
    # 数据集设定
    # 图片的原始信息
    "original_height": 520,
    "original_weight": 704,
    "train_path": r"../data/train",
    "batch_size":12,
    "skf_fold" : 5,


    # 模型设定
    "BOX_DETECTIONS_PER_IMG" : 540,  # 暂时不知道这个参数是做什么的
    "total_epoch": 20,
    "cycle_epoch": 20,


    "start_lr": 1e-3,
    "lr_max_cosin": 1e-3,
    "lr_min_cosin": 1e-7,
    "lr_gammar_cos": 1,
    "lr_warmsteps": 50,
    "optimizer_weight_decay": 0.0005,
    # 类型设定
    "loss_type": "bce",  # 支持 bce focal mser 三种
    "optimizer_type": "AdamW",  # 支持 AdamW RAdam Adam SGDM 三种
    "sgdm_momentum": 0.9,

    # 杂项设定  设定相关阈值
    "cell_type_dict" : {"astro": 1, "cort": 2, "shsy5y": 3},
    "mask_threshold_dict" : {1: 0.55, 2: 0.75, 3:  0.6},
    "min_score_dict": {1: 0.55, 2: 0.75, 3: 0.5},
}


# Global Parameter
train_data = None
val_data = None
model = None
optimizer = None
crition = None
crition_mix_1 = None
crition_mix_2 = None
scheduler = None
best_acc = 0.0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
iter_per_epoch = 0 # 每个epoch代表了多少iter步长


########################################################################################
# 固定seed
def set_seeds(seed=manual_seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


########################################################################################
# 必要功能函数
def rle_decode(mask_rle, shape, color=1):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height, width, channels) of array to return
    color: color for the mask
    Returns numpy array (mask)

    '''
    s = mask_rle.split()

    starts = list(map(lambda x: int(x) - 1, s[0::2]))
    lengths = list(map(int, s[1::2]))
    ends = [x + y for x, y in zip(starts, lengths)]
    if len(shape)==3:
        img = np.zeros((shape[0] * shape[1], shape[2]), dtype=np.float32)
    else:
        img = np.zeros(shape[0] * shape[1], dtype=np.float32)
    for start, end in zip(starts, ends):
        img[start : end] = color

    return img.reshape(shape)


def rle_encoding(x):
    dots = np.where(x.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return ' '.join(map(str, run_lengths))


def remove_overlapping_pixels(mask, other_masks):
    for other_mask in other_masks:
        if np.sum(np.logical_and(mask, other_mask)) > 0:
            mask[np.logical_and(mask, other_mask)] = 0
    return mask

def combine_masks(masks, mask_threshold):
    """
    combine masks into one image
    """
    maskimg = np.zeros((hyper_parameter_group["original_height"], hyper_parameter_group["original_weight"]))
    # print(len(masks.shape), masks.shape)
    for m, mask in enumerate(masks,1):
        maskimg[mask>mask_threshold] = m
    return maskimg


def get_filtered_masks(pred):
    """
    filter masks using MIN_SCORE for mask and MAX_THRESHOLD for pixels
    """
    use_masks = []
    for i, mask in enumerate(pred["masks"]):

        # Filter-out low-scoring results. Not tried yet.
        scr = pred["scores"][i].cpu().item()
        label = pred["labels"][i].cpu().item()
        if scr > hyper_parameter_group["min_score_dict"][label]:
            mask = mask.cpu().numpy().squeeze()
            # Keep only highly likely pixels
            binary_mask = mask > hyper_parameter_group["mask_threshold_dict"][label]
            binary_mask = remove_overlapping_pixels(binary_mask, use_masks)
            use_masks.append(binary_mask)

    return use_masks

# 用于 计算 平均精度   copy 自 https://www.kaggle.com/theoviel/competition-metric-map-iou
def compute_iou(labels, y_pred, verbose=0):
    """
    Computes the IoU for instance labels and predictions.

    Args:
        labels (np array): Labels.
        y_pred (np array): predictions

    Returns:
        np array: IoU matrix, of size true_objects x pred_objects.
    """

    true_objects = len(np.unique(labels))
    pred_objects = len(np.unique(y_pred))

    if verbose:
        print("Number of true objects: {}".format(true_objects))
        print("Number of predicted objects: {}".format(pred_objects))

    # Compute intersection between all objects
    intersection = np.histogram2d(
        labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects)
    )[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins=true_objects)[0]
    area_pred = np.histogram(y_pred, bins=pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection
    intersection = intersection[1:, 1:]  # exclude background
    union = union[1:, 1:]
    union[union == 0] = 1e-9
    iou = intersection / union

    return iou


def precision_at(threshold, iou):
    """
    Computes the precision at a given threshold.

    Args:
        threshold (float): Threshold.
        iou (np array): IoU matrix.

    Returns:
        int: Number of true positives,
        int: Number of false positives,
        int: Number of false negatives.
    """
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) == 1  # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    tp, fp, fn = (
        np.sum(true_positives),
        np.sum(false_positives),
        np.sum(false_negatives),
    )
    return tp, fp, fn


def iou_map(truths, preds, verbose=0):
    """
    Computes the metric for the competition.
    Masks contain the segmented pixels where each object has one value associated,
    and 0 is the background.

    Args:
        truths (list of masks): Ground truths.
        preds (list of masks): Predictions.
        verbose (int, optional): Whether to print infos. Defaults to 0.

    Returns:
        float: mAP.
    """
    ious = [compute_iou(truth, pred, verbose) for truth, pred in zip(truths, preds)]

    if verbose:
        print("Thresh\tTP\tFP\tFN\tPrec.")

    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        tps, fps, fns = 0, 0, 0
        for iou in ious:
            tp, fp, fn = precision_at(t, iou)
            tps += tp
            fps += fp
            fns += fn

        p = tps / (tps + fps + fns)
        prec.append(p)

        if verbose:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tps, fps, fns, p))

    if verbose:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))

    return np.mean(prec)


def get_score(ds, mdl):
    """
    Get average IOU mAP score for a dataset
    """
    mdl.eval()
    iouscore = 0
    for i in (range(len(ds))):
        img, targets = ds[i]
        with torch.no_grad():
            result = mdl([img.to(device)])[0]

        masks = combine_masks(targets['masks'], 0.5)
        labels = pd.Series(result['labels'].cpu().numpy()).value_counts()

        mask_threshold = hyper_parameter_group["mask_threshold_dict"][labels.sort_values().index[-1]]
        pred_masks = combine_masks(get_filtered_masks(result), mask_threshold)
        iouscore += iou_map([masks], [pred_masks])
    return iouscore / len(ds)



## 数据集定义
## 2. 数据集定义以及数据增强

# These are slight redefinitions of torch.transformation classes
# The difference is that they handle the target and the mask
# Copied from Abishek, added new ones
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class VerticalFlip:
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-2)
            bbox = target["boxes"]
            bbox[:, [1, 3]] = height - bbox[:, [3, 1]]
            target["boxes"] = bbox
            target["masks"] = target["masks"].flip(-2)
        return image, target


class HorizontalFlip:
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            target["masks"] = target["masks"].flip(-1)
        return image, target


class Normalize:
    def __call__(self, image, target):
        image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return image, target


class ToTensor:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


def get_transform(train):
    transforms = [ToTensor()]
    transforms.append(Normalize())

    # Data augmentation for train
    if train:
        transforms.append(HorizontalFlip(0.5))
        transforms.append(VerticalFlip(0.5))

    return Compose(transforms)


train_trfm = A.Compose([
    A.OneOf([
        A.RandomGamma(p=1),
        A.GaussNoise(p=1)
    ], p=0.20),
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit = 0.3, p=1),
        A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=30,
                           val_shift_limit=0, p=1)

    ], p=0.20),
])

val_trfm = A.Compose([
])

class CellDataset(Dataset):
    def __init__(self, image_dir, df, transforms=None):
        self.transforms = transforms
        self.image_dir = image_dir
        self.df = df
        self.height = hyper_parameter_group["original_height"]
        self.width = hyper_parameter_group["original_weight"]

        self.image_info = collections.defaultdict(dict)
        temp_df = self.df.groupby(["id", "cell_type"])['annotation'].agg(lambda x: list(x)).reset_index()
        for index, row in temp_df.iterrows():
            self.image_info[index] = {
                'image_id': row['id'],
                'image_path': os.path.join(self.image_dir, row['id'] + '.png'),
                'annotations': list(row["annotation"]),
                'cell_type': hyper_parameter_group["cell_type_dict"][row["cell_type"]]
            }

    def get_box(self, a_mask):
        ''' Get the bounding box of a given mask '''
        pos = np.where(a_mask)
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        return [xmin, ymin, xmax, ymax]

    def __getitem__(self, idx):
        ''' Get the image and the target'''

        img_path = self.image_info[idx]["image_path"]
        img = cv2.imread(img_path)
        # print(img.shape)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = train_trfm(image = img)['image']
        info = self.image_info[idx]

        n_objects = len(info['annotations'])
        # 修正： 为了便于  albumentation进行数据增强  修改维度
        masks = np.zeros((len(info['annotations']), self.height, self.width), dtype=np.uint8)
        # masks = np.zeros( shape = (self.height, self.width , len(info['annotations'])), dtype=np.uint8)
        boxes = []
        for i, annotation in enumerate(info['annotations']):
            a_mask = rle_decode(annotation, (hyper_parameter_group["original_height"], hyper_parameter_group["original_weight"]))

            a_mask = np.array(a_mask) > 0
            masks[i, :, :] = a_mask
            # masks[:, :, i] = a_mask

            boxes.append(self.get_box(a_mask))

        # print(masks.shape)
        # labels
        labels = [int(info["cell_type"]) for _ in range(n_objects)]
        # labels = [1 for _ in range(n_objects)]

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # print(boxes.shape)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # masks = np.array(masks, dtype=np.uint8)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((n_objects,), dtype=torch.int64)

        # This is the required target for the Mask R-CNN
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': image_id,
            'area': area,
            'iscrowd': iscrowd
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        #     al_transform = self.transforms(image =  img, mask =  target["masks"], boxes = target["boxes"])
        #
        # img = al_transform["image"]
        # target["masks"] = al_transform["mask"]
        # target["boxes"] = al_transform["boxes"]
        return img, target

    def __len__(self):
        return len(self.image_info)

## 3. 模型定义
def get_model(num_classes, model_chkpt=None):
    # This is just a dummy value for the classification head

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True,
                                                                   box_detections_per_img=hyper_parameter_group["BOX_DETECTIONS_PER_IMG"])

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes + 1)

    if model_chkpt:
        model.load_state_dict(torch.load(model_chkpt, map_location=device))
    return model


########################################################################################
# 进行数据准备


# Training
def train(epoch,args):
    global train_data
    global val_data
    global model
    global optimizer
    global iter_per_epoch
    global scheduler
    global crition



    print('\nEpoch: %d' % epoch)
    model.train()
    for i, (images, targets) in enumerate(train_data):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()



if __name__ == '__main__':
    try:
        set_seeds(seed = manual_seed)
        RCV_CONFIG = nni.get_next_parameter()
        #RCV_CONFIG = {'lr': 0.1, 'optimizer': 'Adam', 'model':'senet18'}
        _logger.debug(RCV_CONFIG)

        print(RCV_CONFIG)
        # params = vars(merge_parameter(get_params(), RCV_CONFIG))

        acc = 0.0
        best_acc = 0.0

        total_df = pd.read_csv('../data/train_test1_only_use.csv')
        print('** load train.csv file **\n')
        print('length of train.csv : \n%s\n' % (total_df.head()))
        print('\n')
        # Parameter

        fold = 0

        print('==> Training Flod {} data..'.format(fold))

        batchsize = RCV_CONFIG["batch_size"]
        # sgd_momentum = args['sgd_momentum']
        optim_weight_decay = RCV_CONFIG['weight_decay']
        # Data
        print('==> Preparing data..')

        train_df = total_df[total_df.test1_fold != (fold )].reset_index(drop=True)
        val_df = total_df[total_df.test1_fold == (fold )].reset_index(drop=True)
        train_dataset =CellDataset(hyper_parameter_group["train_path"], train_df, transforms=get_transform(train=True))
        val_dataset = CellDataset(hyper_parameter_group["train_path"], val_df, transforms=get_transform(train=False))

        print(len(train_dataset))
        print(len(val_dataset))
        # train, valid = torch.utils.data.random_split(Handvessle_tarinval, lengths=[train_len, valid_len],
        #                                              generator=torch.Generator().manual_seed(manual_seed))
        # 通过DataLoader将数据集按照batch加载到符合训练参数的 DataLoader
        # 为了使用 num_workers在windows中  必须要把这个定义定义在main中 而且保证这个DataLoadre只会出现一次
        train_data = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, pin_memory=True,
                          num_workers=2, collate_fn=lambda x: tuple(zip(*x)))
        val_data = DataLoader(val_dataset, sampler=torch.utils.data.SequentialSampler(val_dataset),
                        batch_size=4, shuffle=False, pin_memory=True,
                        num_workers=1, collate_fn=lambda x: tuple(zip(*x)))

        print(len(train_data))
        print(len(val_data))

        # Model
        print('==> Building model..')
        # 预留 用于更新所使用的是哪种模型
        model = get_model(len(hyper_parameter_group["cell_type_dict"]))


        # 模型迁移到显卡上
        model = model.to(device)

        if load_pth is not None:
            model.load_state_dict(torch.load(load_pth, map_location=device)['state_dict'], strict=True)


        # Optimizer
        if RCV_CONFIG['optimizer'] == 'SGDM':
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=RCV_CONFIG['lr'],
                                        momentum=0.9, weight_decay=optim_weight_decay)
        # if args['optimizer'] == 'Adadelta':
        #     optimizer = optim.Adadelta(model.parameters(), lr=args['lr'])
        # if args['optimizer'] == 'Adagrad':
        #     optimizer = optim.Adagrad(model.parameters(), lr=args['lr'])
        if RCV_CONFIG['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=RCV_CONFIG['lr'],
                                   weight_decay=optim_weight_decay)
        if RCV_CONFIG['optimizer'] == 'AdamW':
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=RCV_CONFIG['lr'],
                                    weight_decay=optim_weight_decay)
        if RCV_CONFIG['optimizer'] == 'RAdam':
            optimizer = RAdam(filter(lambda p: p.requires_grad, model.parameters()), lr=RCV_CONFIG['lr'],
                              weight_decay=optim_weight_decay)

        meta_iteration_per_epoch = len(train_data)
        num_iteration = total_epoch * meta_iteration_per_epoch
        Cycle_Step = meta_iteration_per_epoch * cycle_epoch

        # scheduler
        scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                                  first_cycle_steps=Cycle_Step,  # 50个epoch为一个cycle
                                                  cycle_mult=1.0,
                                                  max_lr=RCV_CONFIG['lr'],
                                                  min_lr= 1e-6,
                                                  warmup_steps=50,
                                                  gamma=1.0)

        for epoch in range(start_epoch, start_epoch+total_epoch):
            train(epoch,RCV_CONFIG)
            score = get_score(val_dataset,model)
            best_acc = max(best_acc,score)

            nni.report_intermediate_result(score)

        # 此处待修改
        nni.report_final_result( best_acc)
    except Exception as exception:
        _logger.exception(exception)
        raise

