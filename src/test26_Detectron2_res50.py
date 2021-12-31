## 使用 ddp  将 LIVECELL数据集 然后做一个 PreTrain  
# 使用  resnet x 101 模型  其中参数细节 和 官方release版本尽量保持一致
# 使用  ddp 在 3张 titan上 跑30000个 ite
# 预训练完成后 使用该权重进行训练
# test20: 1. 在 LiveCell上预训练
#         2. 使用LiveCell论文中的一些结构，但是 anchor_ratio 部分由于显存问题并没有加入，除此以外其他部分都已经加入
#         3. 使用 res50 的backbone，没有做任何修改，学习率设定为 0.001 
#     结论： 目前为止表现最好的模型，表明了 论文中部分结构的可信性
# test21: 1. 在 LiveCell上预训练
#         2. 使用LiveCell论文中的一些结构，但是 anchor_ratio 部分由于显存问题并没有加入，除此以外其他部分都已经加入
#         3. 使用 resx101 的backbone，没有做任何修改，学习率设定为 0.001 
#     结论： 有所提升，但是没有res50的增益那么大
# test22: 1. 在 LiveCell上预训练
#         2. 使用LiveCell论文中的一些结构，但是 anchor_ratio 部分由于显存问题并没有加入，除此以外其他部分都已经加入
#         3. 使用 res50 的backbone，没有做任何修改，学习率设定为 0.005
#         4. 为了加快迭代速度，将总 iteration 降低为 5000， multi-step-lr 做相应修改 
#     结论： 结果全面下降，说明 0.005的学习率不适合用来 Transfer
# test23: 1. 在 LiveCell上预训练
#         2. 使用LiveCell论文中的一些结构，但是 anchor_ratio 部分由于显存问题并没有加入，除此以外其他部分都已经加入
#         3. 使用 resx101 的backbone，没有做任何修改，学习率设定为 0.005
#         4. 为了加快迭代速度，将总 iteration 降低为 5000， multi-step-lr 做相应修改 
#     结论： 为了节约时间，此部分暂时没有送入测试
# test24: 1. 在 LiveCell上预训练
#         2. 使用LiveCell论文中的一些结构，但是 anchor_ratio 部分由于显存问题并没有加入，除此以外其他部分都已经加入
#         3. 使用 res50 的backbone，没有做任何修改，学习率设定为 0.0005
#         4. 为了加快迭代速度，将总 iteration 降低为 5000， multi-step-lr 做相应修改 
#     结论： 等待测试  测试暂缓，因为新的划分体现了无可比拟的优越性
# test25: 1. 在 LiveCell上预训练
#         2. 使用LiveCell论文中的全部结构，除了 FPN in feature 会对显存造成显著影响而未加入
#         3. 使用 res50 的backbone，没有做任何修改，学习率设定为 0.001
#         4. 为了加快迭代速度，将总 iteration 降低为 5000， multi-step-lr 做相应修改 
#         5. 使用新的 split 划分，本次使用的是 2021 种子 而且划分依据不再是细胞类型，而是 每张图中 annotation的数量
# test26: 1. 在 LiveCell上预训练
#         2. 使用LiveCell论文中的全部结构，除了 FPN in feature 会对显存造成显著影响而未加入
#         3. 使用 res50 的backbone，没有做任何修改，学习率设定为 0.001
#         4. 为了加快迭代速度，将总 iteration 降低为 5000， multi-step-lr 做相应修改 
#         5. 使用新的 split 划分，本次使用的是 2022 种子 而且划分依据不再是细胞类型，而是 每张图中 annotation的数量



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
import logging
import glob
import logging
import os
from collections import OrderedDict
import torch
import gc

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.structures import polygons_to_bitmask

# from toolbox.starious_coco_evaluator.coco_evaluation import COCOEvaluator

SCORE_THRESHOLDS = [.15, .3, .55]
MIN_PIXELS = [60, 140, 75]
matrix = [[],[],[],[],[]]
matrix_fold_id = 0

## 注册  评估流程

# Taken from https://www.kaggle.com/theoviel/competition-metric-map-iou
def precision_at(threshold, iou):
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) == 1  # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    return np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)

def score(pred, targ):
    # pred_masks = pred['instances'].pred_masks.cpu().numpy()

    pred_class = torch.mode(pred['instances'].pred_classes)[0]
    take = pred['instances'].scores >= SCORE_THRESHOLDS[pred_class]
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
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, ious)
        p = tp / (tp + fp + fn)
        prec.append(p)
    return np.mean(prec)

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
        return MAPIOUEvaluator(dataset_name)

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

    dataDir=Path('../data/')
    cfg.SEED = -1

    cfg.OUTPUT_DIR  ="/storage/Kaggle_Cell_Segmentation/model/MaskRNN/test26"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.INPUT.MASK_FORMAT='bitmask'
    register_coco_instances('sartorius_train{}'.format(fold_id),{}, '../data/2022_new_split_train_fold{}.json'.format(fold_id), dataDir)
    register_coco_instances('sartorius_val{}'.format(fold_id),{},'../data/2022_new_split_val_fold{}.json'.format(fold_id), dataDir)

    cfg.DATASETS.TRAIN = ('sartorius_train{}'.format(fold_id), )
    cfg.DATASETS.TEST = ('sartorius_val{}'.format(fold_id),)


    ## 注册训练过程
    # DATA
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
    # cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 3000
    # cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 3000
    # cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 6000
    # cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 12000
    cfg.MODEL.PIXEL_MEAN = [128, 128, 128]
    cfg.MODEL.PIXEL_STD = [11.578, 11.578, 11.578]
    ##########################################################################################################

    # cfg.MODEL.WEIGHTS = "/storage/Kaggle_Cell_Segmentation/model/MaskRNN/Res101XPretrained/model_0009999.pth"
    cfg.MODEL.WEIGHTS = '/storage/Kaggle_Cell_Segmentation/model/MaskRNN/Res50Pretrained_Version3/model_best_fold1.pth'  # Let training initialize from the pretrained model
    cfg.SOLVER.IMS_PER_BATCH = 6

    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 10000
    cfg.SOLVER.STEPS = [7000,8500]       
    cfg.SOLVER.CHECKPOINT_PERIOD = 10008
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.TEST.EVAL_PERIOD = len(DatasetCatalog.get('sartorius_train{}'.format(fold_id)))  // cfg.SOLVER.IMS_PER_BATCH  # Once per epoch
        
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args,fold_id):
    global matrix
    global matrix_fold_id

    final_metric = []

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
    del cfg,trainer
    gc.collect()


if __name__ == '__main__':
    config_pth = model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    args = default_argument_parser().parse_args()
    # print("Command Line Args:", args)
    args.num_gpus = 3
    args.config_file = config_pth
    for fold_id in range(1,6):
        launch(
            main,
            args.num_gpus,
            num_machines=args.num_machines,
            machine_rank=args.machine_rank,
            dist_url=args.dist_url,
            args=(args,fold_id),
        )