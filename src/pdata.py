# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         pdata
# Description:  此文件用于检查一下数据集的完整性 以及观测一下数据集的分布情况
# 更详细的数据分布 请参看  https://www.kaggle.com/gunesevitan/sartorius-cell-instance-segmentation-eda/notebook
# Author:       Administrator
# Date:         2021/11/16
# -------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import os
from tqdm.notebook import tqdm
import cv2

## 1. 关于train.csv的数据
# 相关内容可以在竞赛网站的 data 选项卡中观察得到
# id - Unique ID of the image
# annotation - Run length encoded segmentation masks
# width - Width of the image
# height - Height of the image
# cell_type - Type of the cell line
# plate_time - Plate creation time
# sample_date - Timestamp of the sample
# sample_id - Unique ID of the sample
# elapsed_timedelta - Time since first image taken of sample

df_train = pd.read_csv('../data/train.csv')
df_train.drop(columns=['elapsed_timedelta'], inplace=True)
print(f'Training Set Shape: {df_train.shape} - {df_train["id"].nunique()} Images - Memory Usage: {df_train.memory_usage().sum() / 1024 ** 2:.2f} MB')

print(df_train.head)

# 之所以 这么长 是因为 所有的标记区域全部是独立计算的  将id去重之后可以观察得到正确的 长度
# print(np.unique(df_train["id"]).shape)
# (606,) 去重之后的长度为 606 完美对应 ../data/train 中的数据

## 2. 加载半监督学习数据
# 考虑到冲榜 必定需要加载 半监督学习文件夹中的所有数据， 因此提前将数据加入到 df_train 中 并进行保存

def parse_filename(filename):
    image_id = filename.split('.')[0]
    cell_type = filename.split('[')[0]
    filename_split = filename.split('_')
    plate_time = filename_split[-3]
    sample_date = filename_split[-4]
    sample_id = '_'.join(filename_split[:3]) + '_' + '_'.join(filename_split[-2:]).split('.')[0]

    return image_id, cell_type, plate_time, sample_date, sample_id


train_semi_supervised_images = os.listdir('../data/train_semi_supervised/')
for filename in tqdm(train_semi_supervised_images):
    image_id, cell_type, plate_time, sample_date, sample_id = parse_filename(filename)
    sample = {
        'id': image_id,
        'annotation': np.nan,
        'width': 704,
        'height': 520,
        'cell_type': cell_type,
        'plate_time': plate_time,
        'sample_date': sample_date,
        'sample_id': sample_id
    }
    df_train = df_train.append(sample, ignore_index=True)

df_train['cell_type'] = df_train['cell_type'].str.rstrip('s')
print(
    f'Training Set Shape: {df_train.shape} - {df_train["id"].nunique()} Images - Memory Usage: {df_train.memory_usage().sum() / 1024 ** 2:.2f} MB')

# df_train.to_csv("../data/train_plus_semi_supervised.csv")


## 关于 标记 mask  和 肾脏分割一样  标记是通过rle 编码来进行的  这里引入两个之前用到的 非常有用的函数

def decode_rle_mask(rle_mask, shape):

    """
    Decode run-length encoded segmentation mask string into 2d array

    Parameters
    ----------
    rle_mask (str): Run-length encoded segmentation mask string
    shape (tuple): Height and width of the mask

    Returns
    -------
    mask [numpy.ndarray of shape (height, width)]: Decoded 2d segmentation mask
    """

    rle_mask = rle_mask.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (rle_mask[0:][::2], rle_mask[1:][::2])]
    starts -= 1
    ends = starts + lengths

    mask = np.zeros((shape[0] * shape[1]), dtype=np.uint8)
    for start, end in zip(starts, ends):
        mask[start:end] = 1

    mask = mask.reshape(shape[0], shape[1])
    return mask

# 注释： 可以使用如下的方式提取出 纯粹的 train 和 纯粹的 semi 部分
for image_id in tqdm(df_train.loc[~df_train['annotation'].isnull(), 'id'].unique()):

    image = cv2.imread(f'../data/train/{image_id}.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    df_train.loc[df_train['id'] == image_id, 'image_mean'] = np.mean(image)
    df_train.loc[df_train['id'] == image_id, 'image_std'] = np.std(image)

    for rle_mask in df_train.loc[df_train['id'] == image_id, 'annotation']:
        mask = decode_rle_mask(rle_mask, (520, 704))
        df_train.loc[(df_train['id'] == image_id) & (df_train['annotation'] == rle_mask), 'mask_area'] = np.sum(mask) # mask_area指的是当前这个实例
        # 有多少个标记像素

for image_id in tqdm(df_train.loc[df_train['annotation'].isnull(), 'id'].unique()):
    image = cv2.imread(f'../data/train_semi_supervised/{image_id}.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    df_train.loc[df_train['id'] == image_id, 'image_mean'] = np.mean(image)
    df_train.loc[df_train['id'] == image_id, 'image_std'] = np.std(image)

annotation_counts = df_train.loc[~df_train['annotation'].isnull()].groupby('id')['annotation'].count()
df_train['annotation_count'] = df_train['id'].map(annotation_counts)  # annotation_count指的是每张图 总共有多少个 标记像素
# 例如 一张图被重复了 40个横行 那么这个地方的值就是40
# 此版本  包含额外信息  图片的 平均值和 std 被额外记录  同时 mask 的标签数量总和 也被记录
df_train.to_csv('train_processed.csv', index=False)

## 后续更加具体的分析 请参看 kaggle notebook 这里做一下总结
# 首先图片不管是训练数据 还是  半监督数据  都是不平衡的
###  Train
# cort 细胞 占据 320
#shsy5y 细胞  占据155
# astro细胞  占据131
###  半监督数据集中
# cort 细胞 占据 1008
#shsy5y 细胞  占据496
# astro细胞  占据468
# 可以发现 cort细胞 平均比总细胞

# 然后是 标签数据 个数的分析
# 对于所有标记的细胞类型，  考察其在一整张图上的 个数  注意： 由于提供的数据是 包含重叠区域的，因此重叠区域是包含在这个统计中的
# annotation_count
# ----------------
# cort Mean: 33.6781  -  Median: 30.0000  -  Std: 16.4964 - Min: 4.0000 -  Max: 108.0000
# shsy5y Mean: 337.3290  -  Median: 324.0000  -  Std: 149.5954 - Min: 49.0000 -  Max: 790.0000
# astro Mean: 80.3206  -  Median: 73.0000  -  Std: 64.1304 - Min: 5.0000 -  Max: 594.0000


# 对于 每个子项 因为每个细胞划分是单独 列为表格中的一个横行的 通过分析数据中 mask_area部分 可以很好的知道 更细致的分布
#mask_area
# ---------
# cort Mean: 240.1645  -  Median: 208.0000  -  Std: 139.1664 - Min: 33.0000 -  Max: 2054.0000
# shsy5y Mean: 224.4963  -  Median: 193.0000  -  Std: 133.9388 - Min: 30.0000 -  Max: 2254.0000
# astro Mean: 905.8057  -  Median: 665.0000  -  Std: 855.1877 - Min: 37.0000 -  Max: 13327.0000

## 可以看出 对于 cort 和  shsy5y 这两种细胞是最难以区分的
# df_train = pd.read_csv('./train_processed.csv')
# # 但是有一个问题  同一张图 是否会有 不同的 细胞类型 在一张图中？》
#
# # 下面进行测试
# for item in np.unique(df_train["id"]):
#     print(item)
#     temp = df_train.loc[ (df_train['id'] == item) & (~df_train['annotation'].isnull()), "cell_type"].unique()
#     print(temp)

# 这些图片是单独存在的 也就是一种图片里 只存在一种细胞 因此可能还需要一个分类网络先进行分类 判断图片是哪个类
