# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         Test_MRNN_Model1
# Description:  
# Author:       Administrator
# Date:         2021/11/18
# -------------------------------------------------------------------------------
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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

hyper_parameter_group ={
    # 主要环境参数设定
    "seed": 3407,
    "out_dir": r'../model/MaskRNN/test1',
    "log_name_head": 'MaskRNN_test1',
    "Open_Parral_Training": False,
    "initial_checkpoint": None,
    "no_log": False,   # 注意这个参数只在 NNI_SEARCH中使用 其他时候为False
    "checkpoint_dir": r"/storage/Kaggle_Cell_Segmentation/model/MaskRNN/test1/checkpoint",

    # 数据集设定
    # 图片的原始信息
    "original_height": 520,
    "original_weight": 704,
    "PCT_IMAGES_VALIDATION": 0.1, # 此项为数据集划分比例   0.1 表示10%的数据作为 val
    "train_path": r"../data/train",
    "batch_size":12,
    "skf_fold" : 5,


    # 模型设定
    "BOX_DETECTIONS_PER_IMG" : 540,  # 暂时不知道这个参数是做什么的
    "total_epoch": 30,
    "cycle_epoch": 15,


    "start_lr": 1e-3,
    "lr_max_cosin": 1e-3,
    "lr_min_cosin": 1e-7,
    "lr_gammar_cos": 0.01,
    "lr_warmsteps": 50,
    "optimizer_weight_decay": 0.0001,
    # 类型设定
    "loss_type": "bce",  # 支持 bce focal mser 三种
    "optimizer_type": "AdamW",  # 支持 AdamW RAdam Adam 三种





    # 杂项设定
    "cell_type_dict" : {"astro": 1, "cort": 2, "shsy5y": 3},
    "mask_threshold_dict" : {1: 0.55, 2: 0.75, 3:  0.6},
    "min_score_dict": {1: 0.55, 2: 0.75, 3: 0.5},
}


## 0. 必要初始化 提供必要函数
def set_seeds(seed= hyper_parameter_group["seed"] ):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# 打印信息 设定 返回一个 当前信息的字符串 输出
def message(mode='print'):
    if mode == ('print'):
        asterisk = ' '
        loss = batch_loss
    if mode == ('log'):
        asterisk = '*' if iteration % iter_save == 0 else ' '
        loss = train_loss

    text = \
        '%0.7f  %5.4f%s %4.2f  | ' % (rate, iteration / 10000, asterisk, epoch,) + \
        '%4.3f  %5.2f  | ' % (*valid_loss,) + \
        '%4.3f  %4.3f  | ' % (*loss,) + \
        '%s' % (time_to_str(timer() - start_timer, 'min'))

    return text

# 输出当前所用时间
def time_to_str(t, mode='min'):
    if mode=='min':
        t  = int(t)/60
        hr = t//60
        min = t%60
        return '%2d hr %02d min'%(hr,min)

    elif mode=='sec':
        t   = int(t)
        min = t//60
        sec = t%60
        return '%2d min %02d sec'%(min,sec)

    else:
        raise NotImplementedError

def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
        lr +=[ param_group['lr'] ]

    assert(len(lr)>=1) #we support only one param_group
    lr = lr[0]

    return lr

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
    for i in tqdm(range(len(ds))):
        img, targets = ds[i]
        with torch.no_grad():
            result = mdl([img.to(device)])[0]

        masks = combine_masks(targets['masks'], 0.5)
        labels = pd.Series(result['labels'].cpu().numpy()).value_counts()

        mask_threshold = hyper_parameter_group["mask_threshold_dict"][labels.sort_values().index[-1]]
        pred_masks = combine_masks(get_filtered_masks(result), mask_threshold)
        iouscore += iou_map([masks], [pred_masks])
    return iouscore / len(ds)


def visualize_cell_type_distributions(df, title):
    fig, ax = plt.subplots(figsize=(15, 10), dpi=100)

    # sns.barplot(
    #     x=df['cell_type'].value_counts().index,
    #     y=df['cell_type'].value_counts().values,
    #     ax=ax
    # )

    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticklabels([f'{target} ({value_count:,})' for value_count, target in
                        zip(df['cell_type'].value_counts().values, ['cort', 'shsy5y', 'astro'])])
    ax.tick_params(axis='x', labelsize=15, pad=10)
    ax.tick_params(axis='y', labelsize=15, pad=10)
    ax.set_title(title, size=20, pad=15)

    plt.show()



## setup  ----------------------------------------
# 在output目标路径下 新建一个文件夹名字为 checkpoint
set_seeds(hyper_parameter_group["seed"])

for f in ['checkpoint']:
    os.makedirs(hyper_parameter_group["out_dir"] + '/' + f, exist_ok=True)
# 在output目标路径下 新建一个文件夹名字为 checkpoint_best 用于存储 最优的验证结果
for f in ['checkpoint_best']:
    os.makedirs(hyper_parameter_group["out_dir"] + '/' + f, exist_ok=True)
# 2021年7月14日更新： 在输出目录下 放入 source_code目录 存储每一次运行时候的 源代码赋值一份加入
for f in ['source_code']:
    os.makedirs(hyper_parameter_group["out_dir"] + '/' + f, exist_ok=True)


if hyper_parameter_group["no_log"]:
    stdout = print
else:
    logdir = hyper_parameter_group["out_dir"] + '/log'
    logger = get_logger(logdir,OutputOnConsole = True,log_initial= hyper_parameter_group["log_name_head"],logfilename=hyper_parameter_group["log_name_head"])
    stdout = logger.info

stdout('** total seeds setting **\n')
stdout('manual_seed : \n%s\n' % (hyper_parameter_group["seed"]))
stdout('\n')

stdout('** hyper parameter setting **\n')
stdout('hyper parameter group : \n%s\n' % (hyper_parameter_group))
stdout('\n')

if hyper_parameter_group["no_log"] == False:
    stdout('Store Source_Code :')
    stdout('\n')
    copyfile("Train_MRNN_Model1_test1.py", hyper_parameter_group["out_dir"] + "/source_code/train_source_code.py")


device = 'cuda'

## 1. Data Loading 导入
total_df = pd.read_csv('../data/train_test1_only_use.csv')  # 这个是只对test1有效的划分 后续应该使用 5fold划分进行训练
# train_processed 文件为经过  pdata生成的数据  为了方便进行 半监督学习 已经加入了  半监督学习的图片信息

stdout('** load train.csv file **\n')
stdout('length of train.csv : \n%s\n' % (total_df.head()))
stdout('\n')

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
    A.VerticalFlip(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.OneOf([
        A.RandomGamma(p=1),
        A.GaussNoise(p=1)
    ], p=0.20),
    A.OneOf([
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1,p = 1),
        A.RandomBrightnessContrast(brightness_limit=0.4, p=1),
    ], p=0.20),
    A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ToTensorV2(),
])

val_trfm = A.Compose([
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        max_pixel_value=255.0,
        p=1.0,
    ),
    ToTensorV2(),
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


##

df_train = total_df.loc[total_df.test1_fold == 1].reset_index(drop=True)
df_val = total_df.loc[total_df.test1_fold == 0].reset_index(drop=True)

stdout(f"Annotations in train set:      {len(df_train)}")
stdout(f"Annotations in validation set: {len(df_val)}")

# 导入数据集
ds_train = CellDataset(hyper_parameter_group["train_path"], df_train, transforms=get_transform(train=True))
dl_train = DataLoader(ds_train, batch_size=hyper_parameter_group["batch_size"], shuffle=True, pin_memory=True,
                      num_workers=2, collate_fn=lambda x: tuple(zip(*x)))

ds_val = CellDataset(hyper_parameter_group["train_path"], df_val, transforms=get_transform(train=False))
dl_val = DataLoader(ds_val, batch_size=hyper_parameter_group["batch_size"], shuffle=False, pin_memory=True,
                    num_workers=2, collate_fn=lambda x: tuple(zip(*x)))

## 模型定义
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


# Get the Mask R-CNN model
# The model does classification, bounding boxes and MASKs for individuals, all at the same time
# We only care about MASKS
# model = get_model(len(hyper_parameter_group["cell_type_dict"]))
# # model = torch.nn.DataParallel(model)
# model.to(device)

## 1. 检查 训练图的表现
# Plots: the image, The image + the ground truth mask, The image + the predicted mask

def analyze_train_sample(model, ds_train, sample_index):
    img, targets = ds_train[sample_index]
    print(img.shape)
    l = np.unique(targets["labels"])
    ig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 60), facecolor="#fefefe")
    ax[0].imshow(img.numpy().transpose((1, 2, 0)))
    ax[0].set_title(f"cell type {l}")
    ax[0].axis("off")

    masks = combine_masks(targets['masks'], 0.5)
    # plt.imshow(img.numpy().transpose((1,2,0)))
    ax[1].imshow(masks)
    ax[1].set_title(f"Ground truth, {len(targets['masks'])} cells")
    ax[1].axis("off")

    model.eval()
    with torch.no_grad():
        output= model([img.to(device)])
        print(output)
        preds = output[0]

    l = pd.Series(preds['labels'].cpu().numpy()).value_counts()
    lstr = ""
    for i in l.index:
        lstr += f"{l[i]}x{i} "
    # print(l, l.sort_values().index[-1])
    # plt.imshow(img.cpu().numpy().transpose((1,2,0)))
    mask_threshold = hyper_parameter_group["mask_threshold_dict"][l.sort_values().index[-1]]
    # print(mask_threshold)
    pred_masks = combine_masks(get_filtered_masks(preds), mask_threshold)
    ax[2].imshow(pred_masks)
    ax[2].set_title(f"Predictions, labels: {lstr}")
    ax[2].axis("off")
    plt.show()

    # print(masks.shape, pred_masks.shape)
    score = iou_map([masks], [pred_masks])
    print("Score:", score)

# 随便载入一个 模型权重
# model = get_model(len(hyper_parameter_group["cell_type_dict"]), r"C:\Users\Administrator\Desktop\fsdownload\MaskRCNN_TEST1\test1\checkpoint\model_30.pth")
# model.load_state_dict(torch.load(r"C:\Users\Administrator\Desktop\fsdownload\MaskRCNN_TEST1\test1\checkpoint\model_30.pth"))
# # NOTE: It puts the model in eval mode!! Revert for re-training
# analyze_train_sample(model, ds_train, 15)


## 2. 搜索最优结果


val_scores = pd.DataFrame()
for e in range(1,31):
    model_chk = "model_{}.pth".format(e)
    print("Loading:", model_chk)
    model = get_model(len(hyper_parameter_group["cell_type_dict"]), os.path.join(hyper_parameter_group["checkpoint_dir"],model_chk))
    model.load_state_dict(torch.load(os.path.join(hyper_parameter_group["checkpoint_dir"],model_chk)))
    model = model.to(device)
    val_scores.loc[e, "score"] = get_score(ds_val, model)

print(val_scores.sort_values("score", ascending=False))

best_epoch = np.argmax(val_scores["score"])
print(best_epoch + 1)
