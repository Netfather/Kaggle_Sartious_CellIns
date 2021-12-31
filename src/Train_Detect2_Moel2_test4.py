import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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

setup_logger()

dataDir=Path('../data/')
cfg = get_cfg()
cfg.SEED = 3407
cfg.OUTPUT_DIR  = '/storage/Kaggle_Cell_Segmentation/model/MaskRNN/test3'
cfg.INPUT.MASK_FORMAT='bitmask'
register_coco_instances('sartorius_train',{}, '../data/annotations_train_fold1.json', dataDir)
register_coco_instances('sartorius_val',{},'../data/annotations_val_fold1.json', dataDir)
metadata = MetadataCatalog.get('sartorius_train')

train_ds = DatasetCatalog.get('sartorius_train')

print(len(train_ds))

## 如下代码是为了拉出一张图片看看效果
# d = train_ds[42]
# # print(d)
# img = cv2.imread(d["file_name"])
# visualizer = Visualizer(img[:, :, ::-1], metadata=metadata)
# out = visualizer.draw_dataset_dict(d)
# cv2.imwrite("./test.jpg",out.get_image())

## 注册  评估流程

# Taken from https://www.kaggle.com/theoviel/competition-metric-map-iou
def precision_at(threshold, iou):
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) == 1  # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    return np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)

def score(pred, targ):
    pred_masks = pred['instances'].pred_masks.cpu().numpy()
    enc_preds = [mask_util.encode(np.asarray(p, order='F')) for p in pred_masks]
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
                                         file_prefix = "model_best"  # 为之后  预留
                                         ))
        return hooks

    @classmethod
    def build_train_loader(cls, cfg):
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        mapper_params = {'is_train': True,
         'augmentations': [detectron2.data.transforms.augmentation_impl.ResizeShortestEdge(min_size, max_size, sample_style),
                           detectron2.data.transforms.augmentation_impl.RandomFlip(horizontal=True, vertical=False),
                           detectron2.data.transforms.augmentation_impl.RandomFlip(horizontal=False, vertical=True)],
         'image_format': 'BGR',
         'use_instance_mask': True,
         'instance_mask_format': 'bitmask',
         'use_keypoint': False,
         'recompute_boxes': False}


        mapper = detectron2.data.dataset_mapper.DatasetMapper(**mapper_params)

        dataset  = detectron2.data.build.get_detection_dataset_dicts(
                    cfg.DATASETS.TRAIN,
                    filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
                    min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
                    if cfg.MODEL.KEYPOINT_ON
                    else 0,
                    proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
                )

        sampler = detectron2.data.samplers.distributed_sampler.TrainingSampler(len(dataset))

        return detectron2.data.build.build_detection_train_loader(dataset=dataset, mapper=mapper, sampler=sampler, 
                                                                          total_batch_size=cfg.SOLVER.IMS_PER_BATCH,aspect_ratio_grouping=False, num_workers=16 )

## 注册训练过程
# DATA
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("sartorius_train",)
cfg.DATASETS.TEST = ("sartorius_val",)
cfg.DATALOADER.NUM_WORKERS = 2

# MODEL
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo

# 优化器 学习率计划等
cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.0005 
cfg.SOLVER.MAX_ITER = len(DatasetCatalog.get('sartorius_train')) // cfg.SOLVER.IMS_PER_BATCH * 30
cfg.SOLVER.STEPS = [] # 在 Cosin中无用  
cfg.SOLVER.WEIGHT_DECAY = 0.01
cfg.SOLVER.AMP.ENABLED = True
cfg.SOLVER.CHECKPOINT_PERIOD = len(DatasetCatalog.get('sartorius_train')) // cfg.SOLVER.IMS_PER_BATCH * 31 # 不需要保存每个 epoch  相应信息保存在best_checkpoint 中

# 模型细节
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = .5

# 测试指标
cfg.TEST.EVAL_PERIOD = len(DatasetCatalog.get('sartorius_train')) // cfg.SOLVER.IMS_PER_BATCH  # Once per epoch

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = Trainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()


