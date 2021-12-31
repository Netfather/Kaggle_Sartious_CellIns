import os

from torch.nn.functional import fold

from Detectron2_MRCNN_RESX101_Pretrained_Version3 import main
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
import gc

###########################################################################################################################################################
# 2021年12月17日更新： 实现 1toCat 函数 将json文件中的  0 - 7分类修正为 1分类问题 以便于训练
# 函数传入一个列表， 列表中存放需要修正的json文件的地址
def to1Cat(json_pth_lists):
    pth_out_name = [
        'livecell_1cat_annotations_train.json',
        'livecell_1cat_annotations_val.json',
        'livecell_1cat_annotations_test.json'
    ]
    for idx,pth in enumerate(json_pth_lists):
        # 读入对应的 json文件
        try:
            with open(pth) as f:
                data_json = json.loads(f.read())
                # 展示一下 data_json 字典中的键值
                print(data_json.keys())
                # 修改 cate_id 一共对应 两个部分  
                # 1. 字典键值中的  categories 字段
                data_json['categories'] = [{'name':'cell', 'id':1}]
                # 2. 修改 annotations 中 所有的 segmentations 字段
                for element_dic in data_json['annotations']:
                    element_dic['category_id'] = 1
                # 将改写完的文件转存
                with open(pth_out_name[idx], 'w', encoding='utf-8') as f:
                    json.dump(data_json, f, ensure_ascii=True, indent=4)
                del data_json
                gc.collect()
        except:
            raise("Open Json file failed")
###########################################################################################################################################################
# 2021年12月20日更新： https://www.kaggle.com/c/sartorius-cell-instance-segmentation/discussion/295603 
# 根据上面链接的描述，这个discussion指出压缩过的 RLE编码可能存在某些bug 导致漏检，为了检查这个问题，将
# 根据 bie 的  all文件 将 2021 和 2022 种子文件进行手动拆分，将 bie 的 all文件拆分成 未压缩的 bie 划分形式
def SplitBie(bie_json_pth,Standard_json_pth,fold_id):
    '''
    bie_json_pth: 传入 bie 全划分的地址
    Standard_json_pth： 传入 标准 经过压缩的 bie 划分地址
    fold_id: 传入的划分是第几个fold_id
    Return: 返回无，在程序内部将划分好的bie划分进行写入即可
    '''
    try:
        with open(bie_json_pth) as f:
            bie_json = json.loads(f.read())
        with open(Standard_json_pth) as f:
            sta_json = json.loads(f.read())
        # 检查二者的key是否相等
        print(bie_json.keys())
        print(sta_json.keys())
        # 新建两个字典 用于存放
        train_json = dict()
        val_json = dict()
        # 写入 cate 类别
        train_json['categories'] = bie_json['categories']
        val_json['categories'] = bie_json['categories']
        # 读取 sta_json中的图片
        sta_image_list = []
        for item_dict in sta_json['images']:
            sta_image_list.append(item_dict['id'])
        print(len(sta_image_list))
        # 根据这个列表对 bie_json 进行分类
        train_annotations = []
        train_images = []
        val_annotations = []
        val_images = []
        for item_dict in bie_json['images']:
            if (item_dict['id'] in sta_image_list):
                train_images.append(item_dict)
            else:
                val_images.append(item_dict)
        for item_dict in bie_json['annotations']:
            if (item_dict['image_id'] in sta_image_list):
                train_annotations.append(item_dict)
            else:
                val_annotations.append(item_dict)
        train_json['images'] = train_images
        val_json['images'] = val_images
        train_json['annotations'] = train_annotations
        val_json['annotations'] = val_annotations
        # 写为json 文件
        train_file_name = '2022_bie_split_train_fold{}.json'.format(fold_id)
        val_file_name = '2022_bie_split_val_fold{}.json'.format(fold_id)
        with open(train_file_name, 'w', encoding='utf-8') as f:
            json.dump(train_json, f, ensure_ascii=True, indent=4)
        with open(val_file_name, 'w', encoding='utf-8') as f:
            json.dump(val_json, f, ensure_ascii=True, indent=4)
    except:
        raise("Error: Fail to open json files!!")





###########################################################################################################################################################



if __name__ == '__main__':
    list = [19,15,12]
    a = np.array(list) > 18
    print(a)
