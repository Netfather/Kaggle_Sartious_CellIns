## 使用 ddp  将 LIVECELL数据集 然后做一个 PreTrain  
# 使用  resnet50 模型  其中参数细节 和 官方release版本尽量保持一致
# 使用  ddp 在 3张 titan上 跑30000个 ite
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
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
# from toolbox.starious_coco_evaluator.coco_evaluation import COCOEvaluator

matrix = [[],[],[],[],[]]
matrix_fold_id = 0

## 注册  评估流程

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return COCOEvaluator(dataset_name,cfg,distributed= True)

    def build_hooks(self):

        # copy of cfg
        cfg = self.cfg.clone()

        # build the original model hooks
        hooks = super().build_hooks()

        # add the best checkpointer hook
        hooks.insert(-1, BestCheckpointer(cfg.TEST.EVAL_PERIOD, 
                                        DetectionCheckpointer(self.model, cfg.OUTPUT_DIR),
                                        "segm/AP",
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

    dataDir=Path('../data/LIVECell_dataset_2021/total_images/')
    cfg.SEED = -1

    cfg.OUTPUT_DIR  ="/storage/Kaggle_Cell_Segmentation/model/MaskRNN/Res50PretrainedLIVECELL"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.INPUT.MASK_FORMAT='bitmask'
    register_coco_instances('sartorius_train{}'.format(fold_id),{}, '../data/LIVECell_annotations_train.json', dataDir)
    register_coco_instances('sartorius_val{}'.format(fold_id),{},'../data/LIVECell_annotations_val.json', dataDir)
    register_coco_instances('sartorius_test{}'.format(fold_id),{},'../data/LIVECell_annotations_test.json', dataDir)



    ## 注册训练过程
    # DATA
    cfg.INPUT.MIN_SIZE_TRAIN = (440, 480, 520, 560, 580, 620)
    cfg.DATASETS.TRAIN = ("sartorius_train{}".format(fold_id) ,)
    cfg.DATASETS.TEST = ("sartorius_val{}".format(fold_id),)
    cfg.DATALOADER.NUM_WORKERS = 3
    

    # MODEL
    cfg.MODEL.WEIGHTS = "/storage/Kaggle_Cell_Segmentation/model/MaskRNN/Res50Pretrained/model_best_20000.pth"  # Let training initialize from model zoo

    # 优化器 学习率计划等
    cfg.SOLVER.IMS_PER_BATCH = 6
    cfg.SOLVER.BASE_LR = 0.02
    cfg.SOLVER.STEPS =  (35000, 40000)
    # 测试指标
    cfg.TEST.EVAL_PERIOD = 500  # Once per epoch
    cfg.SOLVER.MAX_ITER = 60000

    # 超高学习率 应该只 保持 不超过 一半
    # 然后直接降
    cfg.SOLVER.AMP.ENABLED = True
    cfg.SOLVER.CHECKPOINT_PERIOD = 30010 # 不需要保存每个 epoch  相应信息保存在best_checkpoint 中
    cfg.TEST.DETECTIONS_PER_IMAGE = 3000
    cfg.TEST.PRECISE_BN.ENABLED = False
    cfg.TEST.AUG.ENABLED = False



    # 模型细节 


    # 尝试修改 BN层 # ############################ 如下是对照 LIVECell 修改的结果
    # 这里可能会导致 显存过量  如无必要 建议放弃
    # cfg.MODEL.RESNETS.DEFORM_MODULATED = True
    # cfg.MODEL.RESNETS.DEFORM_NUM_GROUPS = 2
    # cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE = [False, False, False, True]


    cfg.MODEL.BACKBONE.FREEZE_AT = 0
    cfg.MODEL.RESNETS.NORM = "SyncBN"
    cfg.MODEL.RESNETS.STRIDE_IN_1X1 = False
    cfg.MODEL.FPN.NORM = "SyncBN"
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[4], [9], [17], [31], [64], [127]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.25, 0.5, 1.0, 2.0, 4.0]]
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NAME = "CascadeROIHeads"
    cfg.MODEL.ROI_BOX_HEAD.NORM = "SyncBN"
    cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = True
    cfg.MODEL.ROI_BOX_HEAD.NUM_CONV = 4 
    cfg.MODEL.ROI_BOX_HEAD.NUM_FC = 1
    cfg.MODEL.ROI_MASK_HEAD.NORM = "SyncBN"
    cfg.MODEL.ROI_MASK_HEAD.NUM_CONV= 8
    cfg.MODEL.RPN.IN_FEATURES = ["p2" ,"p2", "p3", "p4", "p5", "p6"]
    cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 3000
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 3000
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 6000
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 12000
    cfg.MODEL.RETINANET.NUM_CLASSES = 1
    cfg.MODEL.RETINANET.TOPK_CANDIDATES_TEST = 3000
    cfg.MODEL.PIXEL_MEAN = [128, 128, 128]
    cfg.MODEL.PIXEL_STD = [11.578, 11.578, 11.578]
# 尝试修改 BN层 # ############################ 如下是对照 LIVECell 修改的结果


    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    global matrix
    global matrix_fold_id

    final_metric = []

    for fold_id in range(1,2):

        cfg = setup(args,fold_id)
        matrix_fold_id = fold_id

        """
        If you'd like to do anything fancier than the standard training logic,
        consider writing your own training loop (see plain_train_net.py) or
        subclassing the trainer.
        """
        trainer = Trainer(cfg)
        trainer.resume_or_load(resume=True)
        trainer.train()


if __name__ == '__main__':
    config_pth = model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
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