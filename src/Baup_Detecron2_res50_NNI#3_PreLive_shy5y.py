# 修复 数据集注册的问题
# 根据评论区的指引  单fold的 RNN模型是可以到达 0.30x的结果的  但是需要调节一下超参数 这次训练 超参数和默认保持一致 不乱改

import os
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

SCORE_THRESHOLDS = [.15, .3, .55]
MIN_PIXELS = [60, 120, 60]
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
        nni.report_intermediate_result(np.mean(self.scores))
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


def Train_Per_Fold(fold_id,args):
    global matrix
    global matrix_fold_id

    setup_logger()

    dataDir=Path('../data/')
    dataDir_livecell_shy5y = Path('../data/LIVECell_dataset_2021/SHSY5Y/')
    cfg = get_cfg()
    cfg.SEED = -1
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    # 更新 加入当前ex的id 和 trail id
    current_ex_id = nni.get_experiment_id()
    current_trail_id = nni.get_trial_id()
    # 生成路径
    os.makedirs(os.path.join("./",current_ex_id,current_trail_id), exist_ok=True)
    cfg.OUTPUT_DIR  = './{}/{}'.format(current_ex_id,current_trail_id)
    cfg.INPUT.MASK_FORMAT='bitmask'
    register_coco_instances('sartorius_train{}'.format(fold_id),{}, '../data/starious_bie_annotations_train_fold{}.json'.format(fold_id), dataDir)
    register_coco_instances('livecell_shy5y{}'.format(fold_id),{},'../data/LIVECell_shy5y_train_fold{}.json'.format(fold_id), dataDir_livecell_shy5y)
    register_coco_instances('sartorius_val{}'.format(fold_id),{},'../data/starious_bie_annotations_val_fold{}.json'.format(fold_id), dataDir)

    cfg.DATASETS.TRAIN = ('sartorius_train{}'.format(fold_id),'livecell_shy5y{}'.format(fold_id), )
    cfg.DATASETS.TEST = ('sartorius_val{}'.format(fold_id),)

    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = '/storage/Kaggle_Cell_Segmentation/model/MaskRNN/Res50Pretrained/model_best_fold1.pth'  # Let training initialize from the pretrained model
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = args["lr"]
    cfg.SOLVER.MAX_ITER = 10000
    cfg.SOLVER.STEPS = [7000,9000]    
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000  # Once per epoch
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  
    cfg.TEST.EVAL_PERIOD = (len(DatasetCatalog.get('sartorius_train{}'.format(fold_id))) +len(DatasetCatalog.get('livecell_shy5y{}'.format(fold_id))) ) // cfg.SOLVER.IMS_PER_BATCH  # Once per epoch
    # 测试指标

    # os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = Trainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()

    # 创建  fold 文件夹 并把相应的文件移入
    for f in ['fold{}'.format(fold_id)]:
        os.makedirs(cfg.OUTPUT_DIR + '/' + f, exist_ok=True)

    # 将 metrics.json 和 tf.events移入
    move(cfg.OUTPUT_DIR + '/metrics.json', cfg.OUTPUT_DIR + "/fold{}/metrics_fold{}.json".format(fold_id,fold_id))
    move(cfg.OUTPUT_DIR + '/model_best_fold{}.pth'.format(fold_id), cfg.OUTPUT_DIR + "/fold{}/model_best_fold{}.pth".format(fold_id,fold_id))
    files = glob.glob(os.path.join(cfg.OUTPUT_DIR,'events.out.tfevents.*'))
    for file in files:
        move(file, cfg.OUTPUT_DIR + "/fold{}".format(fold_id))


    # 读入数据
    print("Label:")
    print(matrix[matrix_fold_id - 1])
    return np.max(matrix[matrix_fold_id - 1])


_logger = logging.getLogger('CellInstance_ML')
if __name__ == '__main__':
    try:
        RCV_CONFIG = nni.get_next_parameter()
        _logger.debug(RCV_CONFIG)
        print(RCV_CONFIG)
        final_metric = []

        for fold in range(1,6):
            matrix_fold_id = fold
            metric = Train_Per_Fold(fold,RCV_CONFIG)
            final_metric.append(metric)

            print("Best metric on Fold {} is {}".format(fold,metric))
        nni.report_final_result(np.mean(final_metric))
    except Exception as exception:
        _logger.exception(exception)
        raise
    # 存储训练原文件

    



