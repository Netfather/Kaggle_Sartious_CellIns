import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from pathlib import Path
from typing import Any, Iterator, List, Union

import cv2
import detectron2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycocotools.mask as mask_util
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from PIL.ImageColor import getrgb

def precision_at(threshold, iou):
    """
    Computes the precision at a given threshold.

    Args:
        threshold (float): Threshold.
        iou (np array [n_truths x n_preds]): IoU matrix.

    Returns:
        int: Number of true positives,
        int: Number of false positives,
        int: Number of false negatives.
    """
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) >= 1  # Correct objects
    false_negatives = np.sum(matches, axis=1) == 0  # Missed objects
    false_positives = np.sum(matches, axis=0) == 0  # Extra objects
    tp, fp, fn = (
        np.sum(true_positives),
        np.sum(false_positives),
        np.sum(false_negatives),
    )
    return tp, fp, fn


def iou_map(ious,verbose=0):
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

    if verbose:
        print("Thresh\tTP\tFP\tFN\tPrec.")

    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        tps, fps, fns = 0, 0, 0
        tp, fp, fn = precision_at(t, ious)
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
######################################################################修改一下的参数#################################################################################
# # Only For MR_RES50
SCORE_THRESHOLDS =[0.15,0.30,0.55]
MIN_PIXELS = [60, 140, 75]

# Only For MR_RESX101
# SCORE_THRESHOLDS =[0.25,0.45,0.65]
# MIN_PIXELS = [60, 140, 75]


# 法1： 不论输出的是什么类别 只要有mask 一律添加到最终输出
def score_method1(pred, targ):

    # pred_masks = pred['instances'].pred_masks.cpu().numpy()


    # pred_class_labels = pd.Series(pred['instances'].pred_classes.cpu().numpy()).value_counts()
    # pred_class = pred_class_labels.sort_values().index[-1]
    pred_class = torch.mode(pred['instances'].pred_classes)[0]
    take = pred['instances'].scores >= SCORE_THRESHOLDS[pred_class]
    pred_masks = pred['instances'].pred_masks[take]
    pred_masks = pred_masks.cpu().numpy()


    # masks_after_threshold = []
    # for mask in pred_masks:
    #     masks_after_threshold.append(np.where(mask > MASK_THRESHOLDS[pred_class],1,0).astype(np.uint8))

    # if masks_after_threshold == []:
    #     return 0

    masks_after_threshold = []
    for mask in pred_masks:
        if mask.sum() >= MIN_PIXELS[pred_class]: # skip predictions with small area
            masks_after_threshold.append(mask)
    
    if masks_after_threshold == []:
        return 0
    

    enc_preds = [mask_util.encode(np.asarray(p, order='F')) for p in masks_after_threshold]

    # # enc_preds = [mask_util.encode(np.asarray(p, order='F')) for p in pred_masks]
    enc_targs = list(map(lambda x:x['segmentation'], targ['annotations']))
    ious = mask_util.iou(enc_preds, enc_targs, [0]*len(enc_targs))
    # print(ious)
    # prec = []
    # for t in np.arange(0.5, 1.0, 0.05):
    #     tp, fp, fn = precision_at(t, ious)
    #     p = tp / (tp + fp + fn)
    #     prec.append(p)
    return iou_map(ious)


# 法2： 考虑预测类别  将类别不正确的 统一扔出结果区
cnt_pre_class_right = 0
cnt_pre_class_wrong = 0
table = pd.DataFrame()
LookupTable = {
    0: "shsy5y",
    1: "astro",
    2: "cort",
    3: "null",
    4: "null",
    5: "null",
    6: "null",
    7: "null",
}
def score_method2(pred, targ):

    # pred_masks = pred['instances'].pred_masks.cpu().numpy()


    pred_class_labels = pd.Series(pred['instances'].pred_classes.cpu().numpy()).value_counts()
    pred_class = pred_class_labels.sort_values().index[-1]

#     # 取出 预测类别对应的
    take_mojorities = pred['instances'].pred_classes == pred_class
#     print(take_mojorities.shape)
#     print(take_mojorities)
    pred_masks = pred['instances'].pred_masks[take_mojorities]
    pred_scores = pred['instances'].scores[take_mojorities]

    # len1 = len(pred_scores)

    take = pred_scores >= SCORE_THRESHOLDS[pred_class]
    pred_masks = pred_masks[take]
    pred_masks = pred_masks.cpu().numpy()
    pred_scores = pred_scores[take]
    pred_scores = pred_scores.cpu().numpy()

    # masks_after_threshold = []
    # for mask in pred_masks:
    #     masks_after_threshold.append(np.where(mask > MASK_THRESHOLDS[pred_class],1,0).astype(np.uint8))

    # if masks_after_threshold == []:
    #     return 0

    masks_after_threshold = []
    for mask in pred_masks:
        if mask.sum() >= MIN_PIXELS[pred_class]: # skip predictions with small area
            masks_after_threshold.append(mask)
    
    if masks_after_threshold == []:
        return 0

    enc_preds = [mask_util.encode(np.asarray(p, order='F')) for p in masks_after_threshold]
    
    # # enc_preds = [mask_util.encode(np.asarray(p, order='F')) for p in pred_masks]
    enc_targs = list(map(lambda x:x['segmentation'], targ['annotations']))

    ious = mask_util.iou(enc_preds, enc_targs, [0]*len(enc_targs))
    # print(ious)
    # prec = []
    # for t in np.arange(0.5, 1.0, 0.05):
    #     tp, fp, fn = precision_at(t, ious)
    #     p = tp / (tp + fp + fn)
    #     prec.append(p)

    # 确认一下  那张图的表现最糟糕
    score_image = iou_map(ious)


    #### 如下皆为功能函数 可以删去
    table.loc[targ['file_name'],"score"] = score_image
    table.loc[targ['file_name'],"predic_cell"] = LookupTable[pred_class]
    table.loc[targ['file_name'],"truth_cell"] = LookupTable[targ['annotations'][0]['category_id']]
    table.loc[targ['file_name'],"predic_nums_before_clean"] = pred['instances'].pred_masks.shape[0]
    table.loc[targ['file_name'],"predic_nums_after_all"] = len(masks_after_threshold)
    table.loc[targ['file_name'],"truth_num"] = len(targ['annotations'])

    # 生成图片
    image = cv2.imread(targ['file_name'])

    
    Predict_masks = np.stack(masks_after_threshold)
    layer0 = np.zeros_like(image)
    for mask in Predict_masks: 
        cont, hier = cv2.findContours(mask.astype('uint8'),cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(layer0, cont, -1, getrgb('blue'), thickness = 2)


    Truths_masks = np.stack([mask_util.decode(p) for p in enc_targs])
    layer1 = np.zeros_like(image)
    for mask in Truths_masks: 
        cont, hier = cv2.findContours(mask.astype('uint8'),cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(layer1, cont, -1, getrgb('green'), thickness = 2)
    overlay = (layer0+layer1) 
    m = m = overlay.sum(2) > 0
    image[m]=overlay[m]

    image_name = targ['file_name'].split("/")[-1].split(".")[0]
    cv2.imwrite(Vis_Path + "/{}_{}.png".format(image_name,table.loc[targ['file_name'],"truth_cell"]),image)

    return score_image

def score_all():
    scores = []
    for idx,item in enumerate(val_ds):
        im =  cv2.imread(item['file_name'])
        pred = predictor(im)   
        print("{}/{}".format(idx+1,len(val_ds)))    
        
        sc = score_method1(pred, item)
        scores.append(sc)
        
        # break
    return np.mean(scores)



import glob

pth_names = glob.glob("/storage/Kaggle_Cell_Segmentation/model/MaskRNN/test34/model_best_fold1.pth")
print(pth_names)
pth_scroe_table = pd.DataFrame()
register_coco_instances('sartorius_val',{},'../data/2021_new_split_val_fold1.json', 
                        '../data/')

# Vis_Path = "./res101x_PreLiveCell_vis_fold5"
# os.makedirs(Vis_Path,exist_ok= True)

for idx,pth_name in enumerate(pth_names):
    csv_name = pth_name.split('/')[-1].split('.')[0]
    print("Now Processing {}".format(pth_name))
    cfg = get_cfg()
    # 1. mask_rcnn_R_50_FPN_3x.yaml
    # 2. mask_rcnn_R_101_FPN_3x.yaml
    # 3. mask_rcnn_X_101_32x8d_FPN_3x.yaml
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.INPUT.MASK_FORMAT='bitmask'
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3 

    # MODEL
    ######################################################################################################################################################
    cfg.MODEL.RESNETS.DEFORM_MODULATED = True
    cfg.MODEL.RESNETS.DEFORM_NUM_GROUPS = 2
    cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE = [False, True, True, True]


    cfg.MODEL.BACKBONE.FREEZE_AT = 0
    cfg.MODEL.RESNETS.NORM = "SyncBN"
    cfg.MODEL.RESNETS.STRIDE_IN_1X1 = False
    cfg.MODEL.FPN.NORM = "SyncBN"
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[9], [17], [31], [64], [127]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.25, 0.5, 1.0, 2.0, 4.0]]
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8  
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NAME = "CascadeROIHeads"
    cfg.MODEL.ROI_BOX_HEAD.NORM = "SyncBN"
    cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = True
    cfg.MODEL.ROI_BOX_HEAD.NUM_CONV = 4 
    cfg.MODEL.ROI_BOX_HEAD.NUM_FC = 1
    cfg.MODEL.ROI_MASK_HEAD.NORM = "SyncBN"
    cfg.MODEL.ROI_MASK_HEAD.NUM_CONV= 8

    cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256
    # cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 3000
    # cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 3000
    # cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 6000
    # cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 12000
    cfg.MODEL.PIXEL_MEAN = [128, 128, 128]
    cfg.MODEL.PIXEL_STD = [11.578, 11.578, 11.578]

    cfg.TEST.DETECTIONS_PER_IMAGE = 1000
    ##########################################################################################################

    cfg.TEST.DETECTIONS_PER_IMAGE = 1000
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    # cfg.MODEL.WEIGHTS = '/storage/Kaggle_Cell_Segmentation/src/q6PT7AnE/HbYPx/fold5/model_best_fold5.pth'
    cfg.MODEL.WEIGHTS = pth_name
    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  
    predictor = DefaultPredictor(cfg)

    val_ds = DatasetCatalog.get('sartorius_val')

    metrics = score_all()
    print(metrics)
    # pth_scroe_table.loc[pth_name,"score"] = metrics

# 探查 best checkpoint
# pth_scroe_table.sort_values(by = "score").to_csv("./pth_score.csv")


# 与 Method2 绑定使用
# table.sort_values(by = "score").to_csv(Vis_Path + "/rex101_PreLiveCell_vis_fold5.csv".format(csv_name))
