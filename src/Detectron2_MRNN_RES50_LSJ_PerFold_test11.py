# 废弃 # 废弃 # 废弃 # 废弃 # 废弃 # 废弃 # 废弃 # 废弃 # 废弃 # 废弃 # 废弃 # 废弃 # 废弃 # 废弃 # 废弃 # 废弃 # 废弃 # 废弃 # 废弃 # 废弃 # 废弃 # 废弃 # 废弃 
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
import json
import numpy as np
import glob
# name = os.path.basename(__file__)
# print(name)

import logging

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    AMPTrainer,
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch,
)
import pycocotools.mask as mask_util
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm
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
from detectron2.config import LazyConfig, instantiate
from shutil import move,copyfile  # 用于复制文件
import torch
import json
from pathlib import Path
import detectron2
from detectron2.model_zoo import get_config,get_config_file
from detectron2.evaluation import COCOEvaluator

########如下只是超参数定义  实际训练只用到其中的部分
hyper_parameter_grouping = {
    "batch_size": 18,
    "base_lr" : 0.02,
    "momentum" : 0.9,
    "weight_decay" : 4e-5

}
######################################################
SCORE_THRESHOLDS = [.15, .3, .55]
MIN_PIXELS = [60, 120, 60]


Output_dir = r'../model/MaskRNN/test11'
for f in ['source_code']:
    os.makedirs(Output_dir + '/' + f, exist_ok=True)

## 注册  评估流程

# Taken from https://www.kaggle.com/theoviel/competition-metric-map-iou
def precision_at(threshold, iou):
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) >= 1  # Correct objects
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

class MAPIOUEvaluator(COCOEvaluator):
    def __init__(self, dataset_name):
        super().__init__(dataset_name,distributed=True)
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

def do_test(cfg, model, fold_id):
    if "evaluator" in cfg.dataloader:
        test_loader = detectron2.data.build.build_detection_test_loader(
            dataset=detectron2.data.build.get_detection_dataset_dicts(
                                                names='sartorius_val{}'.format(fold_id),
                                                filter_empty=False, 
                                                min_keypoints=0,
                                                proposal_files=None,  
                                                ), 
            mapper = detectron2.data.DatasetMapper(
                                                augmentations=[
                                                                detectron2.data.transforms.augmentation_impl.ResizeShortestEdge(800, 1333,"choice")
                                                              ], 
                                                image_format='BGR', 
                                                is_train=False,
                                                use_instance_mask = True,
                                                instance_mask_format = 'bitmask',
                                                use_keypoint = False,
                                                recompute_boxes = False), 
            num_workers=4, 
        )

        ret = inference_on_dataset(
            model, test_loader, MAPIOUEvaluator( 'sartorius_val{}'.format(fold_id))
        )
        print_csv_format(ret)
        return ret



def do_train(args, cfg,fold_id):
    """
    Args:
        cfg: an object with the following attributes:
            model: instantiate to a module
            dataloader.{train,test}: instantiate to dataloaders
            dataloader.evaluator: instantiate to evaluator for test set
            optimizer: instantaite to an optimizer
            lr_multiplier: instantiate to a fvcore scheduler
            train: other misc config defined in `configs/common/train.py`, including:
                output_dir (str)
                init_checkpoint (str)
                amp.enabled (bool)
                max_iter (int)
                eval_period, log_period (int)
                device (str)
                checkpointer (dict)
                ddp (dict)
    """
    model = instantiate(cfg.model)
    logger = logging.getLogger("detectron2")
    logger.info("Model:\n{}".format(model))
    model.to(cfg.train.device)


    cfg.optimizer.params.model = model

    optim = instantiate(cfg.optimizer)   # 如果要更换优化器 就在这里更换

    # train_loader = instantiate(cfg.dataloader.train)
    train_loader = detectron2.data.build.build_detection_train_loader(
        dataset=detectron2.data.build.get_detection_dataset_dicts(
                                                names='sartorius_train{}'.format(fold_id),
                                                filter_empty=True, 
                                                min_keypoints=0,
                                                proposal_files=None,  
                                                ), 
        mapper= detectron2.data.DatasetMapper(
                                                augmentations=[
                                                                detectron2.data.transforms.augmentation_impl.ResizeShortestEdge((480, 520, 560, 640, 672, 704, 736, 768, 800), 1333, "choice"),
                                                                detectron2.data.transforms.augmentation_impl.RandomFlip(horizontal=True, vertical=False),
                                                                detectron2.data.transforms.augmentation_impl.RandomFlip(horizontal=False, vertical=True)
                                                                ], 
                                                image_format='BGR', 
                                                is_train=True,
                                                use_instance_mask = True,
                                                instance_mask_format = 'bitmask',
                                                use_keypoint = False,
                                                recompute_boxes = False), 
        num_workers=4, 
        total_batch_size= hyper_parameter_grouping["batch_size"],
        aspect_ratio_grouping=False)

    

    model = create_ddp_model(model, **cfg.train.ddp)
    trainer = (AMPTrainer if cfg.train.amp.enabled else SimpleTrainer)(model, train_loader, optim)
    checkpointer = DetectionCheckpointer(
        model,
        cfg.train.output_dir,
        trainer=trainer,
    )
    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            # 如果要更换学习率计划 就在这里更换
            hooks.LRScheduler(scheduler= instantiate(cfg.lr_multiplier) ),
            hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
            if comm.is_main_process()
            else None,
            hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model,fold_id)),
            hooks.PeriodicWriter(
                default_writers(cfg.train.output_dir, cfg.train.max_iter),
                period=cfg.train.log_period,
            )
            if comm.is_main_process()
            else None,
            hooks.BestCheckpointer( cfg.train.eval_period, 
                                    checkpointer,
                                    "MaP IoU",
                                    "max",
                                    file_prefix = "model_best_fold{}".format(fold_id)  # 为之后  预留
                                    ),
        ]
    )

    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
    if args.resume and checkpointer.has_checkpoint():
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration
        start_iter = trainer.iter + 1
    else:
        start_iter = 0
    trainer.train(start_iter, cfg.train.max_iter)

def main(args,fold_id):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)

    dataDir=Path('../data/')
    register_coco_instances('sartorius_train{}'.format(fold_id),{}, '../data/starious_annotations_train_fold{}.json'.format(fold_id), dataDir)
    register_coco_instances('sartorius_val{}'.format(fold_id),{},'../data/starious_annotations_val_fold{}.json'.format(fold_id), dataDir)

    meta_iter = len(DatasetCatalog.get('sartorius_train{}'.format(fold_id))) // hyper_parameter_grouping["batch_size"]
    # 修正 cfg中一些 超参数
    cfg.train.amp.enabled=True
    cfg.train.checkpointer.max_to_keep= 2
    cfg.train.checkpointer.period= meta_iter * 51
    cfg.train.ddp.broadcast_buffers=False
    cfg.train.ddp.find_unused_parameters=False
    cfg.train.ddp.fp16_compression=True
    cfg.train.device='cuda'
    cfg.train.eval_period=    meta_iter
    # cfg.train.eval_period=    100  # Only For Debug
    cfg.train.init_checkpoint= '/storage/Kaggle_Cell_Segmentation/model/model_final_89a8d3.pkl'
    cfg.train.log_period= 20
    cfg.train.max_iter= meta_iter * 50
    cfg.train.output_dir='/storage/Kaggle_Cell_Segmentation/model/MaskRNN/test11'
    cfg.train.seed = -1

    # 优化器修正
    cfg.optimizer.lr = hyper_parameter_grouping["base_lr"]
    cfg.optimizer.momentum = hyper_parameter_grouping["momentum"]
    cfg.optimizer.weight_decay = hyper_parameter_grouping["weight_decay"]

    # 学习率计划修正
    cfg.lr_multiplier.warmup_length = 0.04
    cfg.lr_multiplier.warmup_factor = 0.001
    cfg.lr_multiplier.scheduler.milestones = [meta_iter * 20, meta_iter * 40]
    cfg.lr_multiplier.scheduler.values = [1,0.1,0.01]
    cfg.lr_multiplier.scheduler.num_updates = cfg.train.max_iter

    # 模型细节 修正
    cfg.model.roi_heads.num_classes = 3

    # 定义 评估集合

    default_setup(cfg, args)

    if args.eval_only:
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
        print(do_test(cfg, model,fold_id))
    else:
        do_train(args, cfg,fold_id)


    # 训练完成 将文件导出
    # 创建  fold 文件夹 并把相应的文件移入
    for f in ['fold{}'.format(fold_id)]:
        os.makedirs(cfg.train.output_dir + '/' + f, exist_ok=True)

    # 将 metrics.json 和 tf.events移入
    move(cfg.train.output_dir + '/metrics.json', cfg.train.output_dir + "/fold{}/metrics_fold{}.json".format(fold_id,fold_id))
    move(cfg.train.output_dir + '/model_best_fold{}.pth'.format(fold_id), cfg.train.output_dir + "/fold{}/model_best_fold{}.pth".format(fold_id,fold_id))
    files = glob.glob(os.path.join(cfg.train.output_dir,'events.out.tfevents.*'))
    for file in files:
        move(file, cfg.train.output_dir + "/fold{}".format(fold_id))


if __name__ == "__main__":


    LSJ_PTH = get_config_file("new_baselines/mask_rcnn_R_50_FPN_200ep_LSJ.py")
    args = default_argument_parser().parse_args()
    args.config_file = LSJ_PTH
    args.num_gpus = 3
    args.eval_only = False
    for fold_id in range(2,6):
    # fold_id = 1

        launch(
                main,
                args.num_gpus,
                num_machines=args.num_machines,
                machine_rank=args.machine_rank,
                dist_url=args.dist_url,
                args=(args,fold_id),
            )



    
