# 此文件用于对detectron出来的模型进行  阈值搜索
# 此文件基于  Detectron2_Threshold_Search演化 是他的快速版本
# 2021年12月16日更新： 此版本根据 Detectron2_Threshold_Search_Fast.py 演化，用于 test29极其以后
import os
import detectron2
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog
import cv2
import pycocotools.mask as mask_util
import numpy as np
from pathlib import Path
from typing import Any, Iterator, List, Union
from tqdm.notebook import tqdm
import json


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

def score(pred, targ):
    # pred_masks = pred['instances'].pred_masks.cpu().numpy()
    pred_masks = pred

    enc_preds = [mask_util.encode(np.asarray(p, order='F')) for p in pred_masks]
    enc_targs = list(map(lambda x:x['segmentation'], targ['annotations']))
    # print(len(enc_targs)) For debug
    ious = mask_util.iou(enc_preds, enc_targs, [0]*len(enc_targs))
    # print(ious.shape)   For debug
    return iou_map(ious)


def score_all():
    for idx,item in enumerate(val_ds):
        # check_list = []
        # for elements in item["annotations"]:
        #     check_list.append(elements["category_id"])
        # print(len(np.unique(check_list)))
        print("{}/{}".format(idx+1,len(val_ds)))

        im =  cv2.imread(item['file_name'])
        pred = predictor(im)  

        res_store_matrix = np.zeros(shape= [19,19])
        for score_idx,score_threshold in enumerate(np.arange(5,100,5)):
            SCORE_THRESHOLDS = score_threshold/100
            take = pred['instances'].scores >= SCORE_THRESHOLDS
            pred_masks = pred['instances'].pred_masks[take]
            pred_masks = pred_masks.cpu().numpy()

            # print(pred_masks.max())
            # print(pred_masks.min())
            # print(np.unique(pred_masks[0,::]))
            for mask_idx,mask_threshold in enumerate(np.arange(5,100,5)):
                MASK_THRESHOLDS = mask_threshold
                masks_after_threshold = []
                # 此时的 masks只剩 true 和 false 了
                for mask in pred_masks:
                    # print(np.unique(mask))
                    # if mask >= MASK_THRESHOLDS[pred_class]: # skip predictions with small area
                    if mask.sum() >= MASK_THRESHOLDS:
                        masks_after_threshold.append(mask)
                
                if masks_after_threshold == []:
                    sc = 0
                else:
                    mask_clean = np.stack(masks_after_threshold,axis = 0)
                    sc = score(mask_clean, item)
                res_store_matrix[score_idx,mask_idx] = sc
        
        scores.append(res_store_matrix.tolist())
        
    return scores



cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.INPUT.MASK_FORMAT='bitmask'
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3 
# cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.TEST.DETECTIONS_PER_IMAGE = 1000

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
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  
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
##########################################################################################################

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3 
cfg.MODEL.WEIGHTS = '/storage/Kaggle_Cell_Segmentation/model/MaskRNN/test29/model_best_fold1.pth'  
predictor = DefaultPredictor(cfg)

dataDir=Path('../data/')
register_coco_instances('sartorius_val',{},'../data/2021_1cate_val_fold1.json', dataDir)

val_ds = DatasetCatalog.get('sartorius_val')
# 进行修正
# SCORE_THRESHOLDS = [.05, .05, .05]
# MASK_THRESHOLDS = [.6, .4, .3]
scores = []
# 再套一层  滑动score的得分

score_all()

# print(scores)

with open("./threshold_search_fast_fold1.json",'w') as outfile:
    json.dump(scores,outfile,indent= 4)