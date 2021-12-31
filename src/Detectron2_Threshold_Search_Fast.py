# 此文件用于对detectron出来的模型进行  阈值搜索
# 此文件基于  Detectron2_Threshold_Search演化 是他的快速版本
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


# 如下代码  通过测试 通过替换一部分的代码定义 可以成功让detectron2 输出 sigmoid的结果 便于进行阈值搜索
def BitMasks__init__(self, tensor: Union[torch.Tensor, np.ndarray]):
    device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
    tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device) # Original code: tensor = torch.as_tensor(tensor, dtype=torch.bool, device=device)
    assert tensor.dim() == 3, tensor.size()
    self.image_size = tensor.shape[1:]
    self.tensor = tensor

detectron2.structures.masks.BitMasks.__init__.__code__ = BitMasks__init__.__code__



def paste_masks_in_image(masks, boxes, image_shape, threshold=0.5):
    """
    Copy pasted from detectron2.layers.mask_ops.paste_masks_in_image and deleted thresholding of the mask
    """
    assert masks.shape[-1] == masks.shape[-2], "Only square mask predictions are supported"
    N = len(masks)
    if N == 0:
        return masks.new_empty((0,) + image_shape, dtype=torch.uint8)
    if not isinstance(boxes, torch.Tensor):
        boxes = boxes.tensor
    device = boxes.device
    assert len(boxes) == N, boxes.shape

    img_h, img_w = image_shape

    # The actual implementation split the input into chunks,
    # and paste them chunk by chunk.
    if device.type == "cpu":
        # CPU is most efficient when they are pasted one by one with skip_empty=True
        # so that it performs minimal number of operations.
        num_chunks = N
    else:
        # GPU benefits from parallelism for larger chunks, but may have memory issue
        num_chunks = int(np.ceil(N * img_h * img_w * BYTES_PER_FLOAT / GPU_MEM_LIMIT))
        assert (
            num_chunks <= N
        ), "Default GPU_MEM_LIMIT in mask_ops.py is too small; try increasing it"
    chunks = torch.chunk(torch.arange(N, device=device), num_chunks)

    img_masks = torch.zeros(
        N, img_h, img_w, device=device, dtype=torch.float32
    )
    for inds in chunks:
        masks_chunk, spatial_inds = _do_paste_mask(
            masks[inds, None, :, :], boxes[inds], img_h, img_w, skip_empty=device.type == "cpu"
        )
        img_masks[(inds,) + spatial_inds] = masks_chunk
    return img_masks

# print(detectron2.layers.mask_ops.paste_masks_in_image.__code__)
detectron2.layers.mask_ops.paste_masks_in_image.__code__ = paste_masks_in_image.__code__





def precision_at(threshold, iou):
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) >= 1  # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    return np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)

def score(pred, targ):
    # pred_masks = pred['instances'].pred_masks.cpu().numpy()
    pred_masks = pred

    enc_preds = [mask_util.encode(np.asarray(p, order='F')) for p in pred_masks]
    enc_targs = list(map(lambda x:x['segmentation'], targ['annotations']))
    # print(len(enc_targs)) For debug
    ious = mask_util.iou(enc_preds, enc_targs, [0]*len(enc_targs))
    # print(ious.shape)   For debug
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, ious)
        p = tp / (tp + fp + fn)
        prec.append(p)
    return np.mean(prec)


def score_all():
    for idx,item in enumerate(val_ds):
        # check_list = []
        # for elements in item["annotations"]:
        #     check_list.append(elements["category_id"])
        # print(len(np.unique(check_list)))
        print("{}/{}".format(idx+1,len(val_ds)))

        im =  cv2.imread(item['file_name'])
        pred = predictor(im)  
        pred_class_index = pred['instances'].pred_classes[0]
        # print(pred_class_index)

        pred_class = torch.mode(pred['instances'].pred_classes)[0]

        res_store_matrix = np.zeros(shape= [19,19])
        for score_idx,score_threshold in enumerate(np.arange(5,100,5)):
            SCORE_THRESHOLDS = [score_threshold/100,score_threshold/100,score_threshold/100]
            take = pred['instances'].scores >= SCORE_THRESHOLDS[pred_class]
            pred_masks = pred['instances'].pred_masks[take]
            pred_masks = pred_masks.cpu().numpy()

            # print(pred_masks.max())
            # print(pred_masks.min())
            # print(np.unique(pred_masks[0,::]))
            for mask_idx,mask_threshold in enumerate(np.arange(5,100,5)):
                MASK_THRESHOLDS = [mask_threshold/100,mask_threshold/100,mask_threshold/100]
                masks_after_threshold = []
                # 此时的 masks只剩 true 和 false 了
                for mask in pred_masks:
                    # print(np.unique(mask))
                    # if mask >= MASK_THRESHOLDS[pred_class]: # skip predictions with small area
                    masks_after_threshold.append( np.where(mask > MASK_THRESHOLDS[pred_class],1,0).astype(np.uint8) )  
                
                if masks_after_threshold == []:
                    sc = 0
                else:
                    mask_clean = np.stack(masks_after_threshold,axis = 0)
                    sc = score(mask_clean, item)
                res_store_matrix[score_idx,mask_idx] = sc
        
        scores[pred_class_index].append(res_store_matrix.tolist())
        
    return scores



cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))

## 注册训练过程
# DATA
fold_id = 0
cfg.INPUT.MIN_SIZE_TRAIN = (440, 480, 520, 580, 620)
cfg.INPUT.MIN_SIZE_TEST = 800
cfg.DATASETS.TRAIN = ("sartorius_train{}".format(fold_id),)
cfg.DATASETS.TEST = ("sartorius_val{}".format(fold_id),)
cfg.DATALOADER.NUM_WORKERS = 4


# 模型细节
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  


##############################根据LIVECELL稍作修改
# cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.25, 0.5, 1.0, 2.0, 4.0]]
cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = True

cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 3000
cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 3000
cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 6000
cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 12000

# cfg.MODEL.RETINANET.NUM_CLASSES = 3
cfg.MODEL.RETINANET.TOPK_CANDIDATES_TEST = 3000

cfg.MODEL.PIXEL_MEAN = [128, 128, 128]
cfg.MODEL.PIXEL_STD = [11.578, 11.578, 11.578]

cfg.TEST.DETECTIONS_PER_IMAGE = 3000

###########################根据LIVECELL稍作修改


cfg.INPUT.MASK_FORMAT='bitmask'
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3 
cfg.MODEL.WEIGHTS = '/storage/Kaggle_Cell_Segmentation/model/MaskRNN/test16/fold3/model_best_fold3.pth'  
cfg.TEST.DETECTIONS_PER_IMAGE = 1000
predictor = DefaultPredictor(cfg)
dataDir=Path('../data/')
register_coco_instances('sartorius_val',{},'../data/starious_annotations_val_fold3.json', dataDir)

val_ds = DatasetCatalog.get('sartorius_val')
# 进行修正
# SCORE_THRESHOLDS = [.05, .05, .05]
# MASK_THRESHOLDS = [.6, .4, .3]
scores = [[],[],[]]
# 再套一层  滑动score的得分

score_all()

# print(scores)

with open("./threshold_search_fast_fold3.json",'w') as outfile:
    json.dump(scores,outfile,indent= 4)