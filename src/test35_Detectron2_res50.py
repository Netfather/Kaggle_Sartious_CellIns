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
#     结论： 2021年12月16日更新：目前为止表现最好的模型，lb分数均匀分布在0.317到0.321之间，理论上做了ensamble之后应该会更好
# test26: 1. 在 LiveCell上预训练
#         2. 使用LiveCell论文中的全部结构，除了 FPN in feature 会对显存造成显著影响而未加入
#         3. 使用 res50 的backbone，没有做任何修改，学习率设定为 0.001
#         4. 保持迭代总长为 10000  2021年12月16日更新： 已经确认 10000的迭代总长是没有作用的
#         5. 使用新的 split 划分，本次使用的是 2022 种子 而且划分依据不再是细胞类型，而是 每张图中 annotation的数量
#     结论： 2021年12月16日更新：
#            2021年12月16日更新： 已经确认 10000的迭代总长是没有作用的，保持5000 iter总长不变
# test27: 1. 在 LiveCell上预训练
#         2. 使用LiveCell论文中的一些结构，但是 anchor_ratio 部分由于显存问题并没有加入，除此以外其他部分都已经加入
#         3. 使用 res50 的backbone，没有做任何修改，学习率设定为 0.001 
#         4. 为了加快迭代速度，将总 iteration 降低为 5000， multi-step-lr 做相应修改 
#         5. 使用新的 split 划分，本次使用的是 2021 种子 而且划分依据不再是细胞类型，而是 每张图中 annotation的数量
#         6. 修正原有的 每个fold的训练方式，现在将 fold_id 在 主函数中迭代，然后传入 launch中
#    结论： 这是 test25的对照实验，用于确认增益究竟是 新划分带来的，还是全部引入论文结构带来的。
#           2021年12月17日更新： 确认相关增益是论文中结构带来的，与其他的无关
# test28: 1. 在 LiveCell上预训练
#         2. 完全没有使用任何 livecell论文中的结构，保持默认设置不变
#         3. 使用 res50 的backbone，没有做任何修改，学习率设定为 0.001 
#         4. 为了加快迭代速度，将总 iteration 降低为 5000， multi-step-lr 做相应修改 
#         5. 使用新的 split 划分，本次使用的是 2021 种子 而且划分依据不再是细胞类型，而是 每张图中 annotation的数量
#         6. 修正原有的 每个fold的训练方式，现在将 fold_id 在 主函数中迭代，然后传入 launch中
#    结论： 这是 test25的对照实验，用于确认增益究竟是 新划分带来的，还是全部引入论文结构带来的。
#           2021年12月17日更新： 确认相关增益是论文中结构带来的，与其他的无关
# test29: 1. 在 LiveCell上预训练
#         2. 使用LiveCell论文中的全部结构，除了 FPN in feature 会对显存造成显著影响而未加入
#         3. 使用 res50 的backbone，没有做任何修改，学习率设定为 0.001 
#         4. 为了加快迭代速度，将总 iteration 降低为 5000， multi-step-lr 做相应修改 
#         5. 使用2021的 split 划分，本次使用的是 2021 种子 而且划分依据不再是细胞类型，而是 每张图中 annotation的数量
#         6. 修正原有的 每个fold的训练方式，现在将 fold_id 在 主函数中迭代，然后传入 launch中
#         7. 现在模型不再遵循之前的三分类方式，而是修正为和论文中一致的一分类，然后训练完成后，通过阈值搜索来确认最优置信分数区间和最小像素区间
#            为了匹配1分类模式，本次test的cv会遵循最原始方式，等本次test训练完成之后，再通过搜索确定阈值
#    结论： 2021年12月20日 更新： 完全没有用，不予考虑
# test30: 1. 在 LiveCell上预训练
#         2. 使用LiveCell论文中的全部结构，除了 FPN in feature 会对显存造成显著影响而未加入
#         3. 使用 resx101 的backbone，没有做任何修改，学习率设定为 0.001
#         4. 为了加快迭代速度，将总 iteration 降低为 5000， multi-step-lr 做相应修改 
#         5. 使用2021的 split 划分，本次使用的是 2021 种子 而且划分依据不再是细胞类型，而是 每张图中 annotation的数量
#         6. 修正原有的 每个fold的训练方式，现在将 fold_id 在 主函数中迭代，然后传入 launch中 
#         7. 将 cv评估部分 保持和本地的完全一致，现在他们用的是完全相同的函数和取样方法          
#    结论： 2021年12月21日更新： 和 res50 的test25结果相比， 本地cv 全面提升，但是 lb 几乎没有提升
#           基于此，可以认为 resx101 理应获得更不错的效果，具体的结果仍然有待商榷。
# test31: 1. 在 LiveCell上预训练
#         2. 使用LiveCell论文中的全部结构，除了 FPN in feature 会对显存造成显著影响而未加入
#         3. 使用 resx101 的backbone，没有做任何修改，学习率设定为 0.001
#         4. 为了加快迭代速度，将总 iteration 降低为 5000， multi-step-lr 做相应修改 
#         5. 使用2022的 split 划分，本次使用的是 2022 种子 而且划分依据不再是细胞类型，而是 每张图中 annotation的数量
#         6. 修正原有的 每个fold的训练方式，现在将 fold_id 在 主函数中迭代，然后传入 launch中
#         7. 根据kaggle讨论区指出的mask bug问题，这里使用bie同款无压缩的RLEjson文件进行考察
#              2021年12月21日更新： 经过确认 未压缩的RLE 和 压缩过的RLE 会产生不同的结果。 因此本次使用将使用 未压缩的bie同款RLE进行考察
#         8. 将 cv评估部分 保持和本地的完全一致，现在他们用的是完全相同的函数和取样方法     
#         9. 根据 test30的结果，经过验证，表明不同的backbone 应该使用不同的 置信分数阈值，特别是当resx101还修改了默认的结果的时候
#            目前 test31使用的是 经过搜索之后的 0.25 0.45 0.65 分数阈值   
#         10. cfg 中补入新的词条 该词条的缺失可能是导致训练过程中本地cv测不准的罪魁祸首
#         11. 修正 TOPK的容量， livecell数据集中的容量 和比赛的数据容量并不匹配， 修正为正确容量后
#    结论： 2021年12月22日更新： scroe的阈值 与 TOPK的容量息息相关 
#           2021年12月23日更新： 根据lb上的测试，以及local cv的反应，可以证明相关结果非常糟糕，基本可以确认 bie 的未压缩编码方式有问题，
#                               x101的结果相比 test26 大幅度下降，所以不再使用  bie的任何编码方式
# test32: 1. 在 LiveCell上预训练
#         2. 使用LiveCell论文中的全部结构，除了 FPN in feature 会对显存造成显著影响而未加入
#         3. 使用 resx101 的backbone，没有做任何修改，学习率设定为 0.001
#         4. 为了加快迭代速度，将总 iteration 降低为 5000， multi-step-lr 做相应修改 
#         5. 使用2022的 split 划分，本次使用的是 2022 种子 而且划分依据不再是细胞类型，而是 每张图中 annotation的数量
#         6. 经过测试 bie 版本的表现效果并不好， 修正为 官方推荐的版本
#         8. 将 cv评估部分 保持和本地的完全一致，现在他们用的是完全相同的函数和取样方法     
#         9. 根据 test30的结果，经过验证，表明不同的backbone 应该使用不同的 置信分数阈值，特别是当resx101还修改了默认的结果的时候
#            目前 test31使用的是 经过搜索之后的 0.25 0.45 0.65 分数阈值   
#            2021年12月22日更新： 阈值的变化来源于 TOPK 的修改， test31 的local cv 表现并不出彩，但是相关结果还是等待明天 有了完全的 cv 和 lb 再做决定
#               如果TOPK 和 livecell中保持一致，那么 阈值应当设置为 0.25 0.45 0.65
#               如果TOPK 和 默认的cfg保持一致，那么 阈值应当设置为 0.15 0.30 0.55
#    结论：  2021年12月24日更新： 结果对比test26没多大变化，说明当前已经几乎不可能再2有较好的提升了，考虑使用半监督 或者 降低学习率再试一试
# test33: 1. 在 LiveCell上预训练
#         2. 使用LiveCell论文中的全部结构，除了 FPN in feature 会对显存造成显著影响而未加入
#         3. 使用 resx101 的backbone，没有做任何修改，学习率设定为 5e-3
#             2021年12月23日更新： loss会在一开始疯狂下降，然后升高， 可能 0.001的学习率不适合，予以修正
#         4. 为了加快迭代速度，将总 iteration 降低为 3500, multi-step-lr 做相应修改 
#         5. 使用2021的 split 划分，本次使用的是 2021 种子 而且划分依据不再是细胞类型，而是 每张图中 annotation的数量
#         6. 经过测试 bie 版本的表现效果并不好， 修正为 官方推荐的版本
#         7. 2021年12月23日更新： 修复了官方的 coco_eva 的问题，现在可以正确使用了 
#            2021年12月24日更新： 现在计算的几乎百分百就是 local_cv的内容了，绝对正确   
#         8. 2021年12月24日更新： 目前统一一下： 所有的 res50 结构 统一使用 "default" 设置
#                                              所有的 resx101 结果 统一使用 "livecell" 设置
#
#    结论：
# test34: 1. 在 LiveCell上预训练  预训练权重为 Version5 版本
#         2. 使用LiveCell论文中的全部结构，除了 FPN in feature 会对显存造成显著影响而未加入
#         3. 使用 res50 的backbone，没有做任何修改，学习率设定为 5e-3
#             2021年12月23日更新： loss会在一开始疯狂下降，然后升高， 可能 0.001的学习率不适合，予以修正
#         4. 为了加快迭代速度，将总 iteration 降低为 3500, multi-step-lr 做相应修改 
#         5. 使用2021的 split 划分，本次使用的是 2021 种子 而且划分依据不再是细胞类型，而是 每张图中 annotation的数量
#         6. 2021年12月23日更新： 修复了官方的 coco_eva 的问题，现在可以正确使用了 
#            2021年12月24日更新： 现在计算的几乎百分百就是 local_cv的内容了 
#         7. 2021年12月24日更新： 目前统一一下： 之前没有修改 TOPK 容量的，权重统一为 "default" 设置
#                                              所有和livecell保持 TOPK 容量一致的结构 统一使用 "livecell" 设置
#
#    结论：
# test35: 1. 在 LiveCell 的 Version5 上预训练
#         2. 使用LiveCell论文中的全部结构，除了 FPN in feature 会对显存造成显著影响而未加入
#         3. 使用 res50 的backbone，没有做任何修改，学习率设定为 5e-3
#         4. 为了加快迭代速度，将总 iteration 降低为 3500， multi-step-lr 做相应修改 
#         5. 使用3407的 split 划分，本次使用的是 3407 种子 而且划分依据是细胞类型
#         6. 2021年12月23日更新： 修复了官方的 coco_eva 的问题，现在可以正确使用了 
#            2021年12月24日更新： 现在计算的几乎百分百就是 local_cv的内容了 
#         7. 2021年12月24日更新： 目前统一一下： 之前没有修改 TOPK 容量的，权重统一为 "default" 设置
#                                              所有和livecell保持 TOPK 容量一致的结构 统一使用 "livecell" 设置
#
#    结论：

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
os.environ['NUMEXPR_MAX_THREADS'] = "12"
import glob
import json
import logging
import os
import random
from collections import OrderedDict
from pathlib import Path
from shutil import copyfile, move  # 用于复制文件

import albumentations as A
import cv2
import detectron2
import detectron2.data.transforms as T
import detectron2.utils.comm as comm
import matplotlib.pyplot as plt
import nni
import numpy as np
import pandas as pd
import pycocotools.mask as mask_util
import torch
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import (BestCheckpointer, DefaultPredictor,
                               DefaultTrainer, default_argument_parser,
                               default_setup, hooks, launch)

from toolbox.livecell_coco_evaluator import COCOEvaluator
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.structures import polygons_to_bitmask
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode, Visualizer

SCORE_THRESHOLDS = [.25, .45, .65]
MIN_PIXELS = [60, 140, 75]
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

    cfg.OUTPUT_DIR  ="/storage/Kaggle_Cell_Segmentation/model/MaskRNN/test35"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.INPUT.MASK_FORMAT='bitmask'
    register_coco_instances('sartorius_train{}'.format(fold_id),{}, '../data/3407_new_split_train_fold{}.json'.format(fold_id), dataDir)
    register_coco_instances('sartorius_val{}'.format(fold_id),{},'../data/3407_new_split_val_fold{}.json'.format(fold_id), dataDir)

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
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 3000
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 3000
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 6000
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 12000

    cfg.MODEL.RETINANET.NUM_CLASSES = 8
    cfg.MODEL.RETINANET.TOPK_CANDIDATES_TEST = 3000

    cfg.MODEL.PIXEL_MEAN = [128, 128, 128]
    cfg.MODEL.PIXEL_STD = [11.578, 11.578, 11.578]

    cfg.TEST.DETECTIONS_PER_IMAGE = 3000
    ##########################################################################################################


    cfg.TEST.DETECTIONS_PER_IMAGE = 1000
    cfg.MODEL.RETINANET.NUM_CLASSES = 3
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3

    # cfg.MODEL.WEIGHTS = "/storage/Kaggle_Cell_Segmentation/model/MaskRNN/Res101XPretrained/model_0009999.pth"
    cfg.MODEL.WEIGHTS = '/storage/Kaggle_Cell_Segmentation/model/MaskRNN/Res50Pretrained_Version5/model_best_fold1.pth'   # Let training initialize from the pretrained model
    cfg.SOLVER.IMS_PER_BATCH = 6

    cfg.SOLVER.BASE_LR = 5e-3
    # 学习率到底是高 还是低？
    cfg.SOLVER.WARMUP_ITERS = 500
    cfg.SOLVER.MAX_ITER = 3500
    cfg.SOLVER.STEPS = [1500,2500]           
    cfg.SOLVER.CHECKPOINT_PERIOD = 5008 
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.MODEL.RETINANET.NUM_CLASSES = 3
    cfg.TEST.EVAL_PERIOD = len(DatasetCatalog.get('sartorius_train{}'.format(fold_id)))  // cfg.SOLVER.IMS_PER_BATCH  # Once per epoch

    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    global matrix
    global matrix_fold_id

    final_metric = []

    for fold_id in range(1,6):

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

