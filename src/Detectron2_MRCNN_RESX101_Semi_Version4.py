## 使用 ddp  将 LIVECELL数据集 然后做一个 PreTrain  
# 使用  resnet50 模型  其中参数细节 和 官方release版本尽量保持一致
# 使用  ddp 在 3张 titan上 跑30000个 ite
# Version3: 第三个版本的 Pretrain 前两个 第一个：使用的是默认设置  第二个： 使用的是不包含 deforma 和 anchor ratio的设置
#           这个版本 会在尽量不影响显存的基础上，进行预训练
# RESX101: 使用 resx101 作为 backbone
# Version4: 这个版本使用的是 Semi 查看半监督学习是否会给结果带来提升。  使用之前的 res50 x101 101 三个结果共同生成的  semi.json 文件 
#           如果确认这个版本有效，那么接下来我会考虑使用最新的结果来生成新的 semi

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
os.environ['NUMEXPR_MAX_THREADS'] = "12"
import detectron2
from pathlib import Path
import random, cv2
import matplotlib.pyplot as plt
import numpy as np
import pycocotools.mask as mask_util
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger
from detectron2.evaluation.evaluator import DatasetEvaluator
import albumentations as A
import detectron2.data.transforms as T
from detectron2.engine import BestCheckpointer
from detectron2.checkpoint import DetectionCheckpointer
from shutil import move,copyfile  # 用于复制文件
import torch
import json
import nni
import logging
import glob
import logging
import os
from collections import OrderedDict
import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from toolbox.livecell_coco_evaluator import COCOEvaluator
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.structures import polygons_to_bitmask
import pandas as pd

# from toolbox.starious_coco_evaluator.coco_evaluation import COCOEvaluator

SCORE_THRESHOLDS = [.25, .45, .65]
MIN_PIXELS = [30, 60, 30]
matrix = [[],[],[],[],[]]
matrix_fold_id = 0

####################################################################################################################################################################
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
## 注册  评估流程
def score(pred, targ):
    # pred_masks = pred['instances'].pred_masks.cpu().numpy()

    pred_class_labels = pd.Series(pred['instances'].pred_classes.cpu().numpy()).value_counts()
    pred_class = pred_class_labels.sort_values().index[-1]

    take_mojorities = pred['instances'].pred_classes == pred_class
    take = pred['instances'].scores >= SCORE_THRESHOLDS[pred_class]
    take = take_mojorities & take

    pred_masks = pred['instances'].pred_masks[take]
    pred_masks = pred_masks.cpu().numpy()

    masks_after_threshold = []
    for mask in pred_masks:
        if mask.sum() >= MIN_PIXELS[pred_class]: # skip predictions with small area
            masks_after_threshold.append(mask)
    
    if masks_after_threshold == []:
        return 0
    
    enc_preds = [mask_util.encode(np.asarray(p, order='F')) for p in masks_after_threshold]

    # enc_preds = [mask_util.encode(np.asarray(p, order='F')) for p in pred_masks]
    enc_targs = list(map(lambda x:x['segmentation'], targ))
    ious = mask_util.iou(enc_preds, enc_targs, [0]*len(enc_targs))
    return iou_map(ious)

class MAPIOUEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name):
        dataset_dicts = DatasetCatalog.get(dataset_name)
        self.annotations_cache = {item['image_id']:item['annotations'] for item in dataset_dicts}
            
    def reset(self):
        self.scores = []

    def process(self, inputs, outputs):
        for inp, out in zip(inputs, outputs):
            if len(out['instances']) == 0:
                self.scores.append(0)    
            else:
                targ = self.annotations_cache[inp['image_id']]
                self.scores.append(score(out, targ))

    def evaluate(self):
        global matrix
        global matrix_fold_id
        matrix[matrix_fold_id - 1].append(np.mean(self.scores))
        return {"MaP IoU": np.mean(self.scores)}

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return COCOEvaluator(dataset_name, cfg = cfg, distributed = True, output_dir = output_folder,TOPK_TYPE= 'livecell')
        # return MAPIOUEvaluator(dataset_name)

    def build_hooks(self):

        # copy of cfg
        cfg = self.cfg.clone()

        # build the original model hooks
        hooks = super().build_hooks()

        # add the best checkpointer hook
        hooks.insert(-1, BestCheckpointer(cfg.TEST.EVAL_PERIOD, 
                                        DetectionCheckpointer(self.model, cfg.OUTPUT_DIR),
                                        "MaP IoU",
                                        "max",
                                        file_prefix = "model_best_fold{}".format(matrix_fold_id)  # 为之后  预留
                                        ))
        return hooks

def setup(args,fold_id):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)

    dataDir_Train= Path('../data/train_semi_supervised/')
    dataDir_Test = Path('../data/')
    cfg.SEED = -1

    cfg.OUTPUT_DIR  ="/storage/Kaggle_Cell_Segmentation/model/MaskRNN/ResX101_Semi_Version4"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.INPUT.MASK_FORMAT='bitmask'
    register_coco_instances('sartorius_train',{}, '../data/semi_all_train.json', dataDir_Train)
    register_coco_instances('sartorius_test',{},'../data/2021_new_split_train_fold1.json', dataDir_Test)

    ## 注册训练过程
    # DATA
    cfg.DATASETS.TRAIN = ("sartorius_train" ,)
    cfg.DATASETS.TEST = ("sartorius_test",)
    cfg.DATALOADER.NUM_WORKERS = 8
    
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
    cfg.MODEL.PIXEL_MEAN = [128, 128, 128]
    cfg.MODEL.PIXEL_STD = [11.578, 11.578, 11.578]

    cfg.MODEL.RETINANET.NUM_CLASSES = 8
    cfg.MODEL.RETINANET.TOPK_CANDIDATES_TEST = 3000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 3000
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 3000
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 6000
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 12000

    cfg.TEST.DETECTIONS_PER_IMAGE = 1000
    ##########################################################################################################

    cfg.MODEL.WEIGHTS = '/storage/Kaggle_Cell_Segmentation/model/MaskRNN/ResX101Pretrained_Version3/model_best_fold0.pth'
    cfg.SOLVER.IMS_PER_BATCH = 6

    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 20000
    cfg.SOLVER.STEPS = [14000,17000]           
    cfg.SOLVER.CHECKPOINT_PERIOD = 20010
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.TEST.EVAL_PERIOD = len(DatasetCatalog.get('sartorius_train')) // cfg.SOLVER.IMS_PER_BATCH  # Once per epoch

    # cfg.SOLVER.AMP.ENABLED = True
        
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    global matrix
    global matrix_fold_id

    final_metric = []
    fold_id = 0

    cfg = setup(args,fold_id)
    matrix_fold_id = fold_id

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == '__main__':
    config_pth = model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
    args = default_argument_parser().parse_args()
    # print("Command Line Args:", args)
    args.num_gpus = 3
    args.config_file = config_pth
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )