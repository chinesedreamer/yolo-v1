import os

#
# path and dataset parameter
#

#voc2007  datapath
#trainval
PASCAL_PATH = os.path.join('/media/xupengtao/新加卷/datasets/目标检测/PASCAL_VOC/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007')
#test
PASCAL_TEST_PATH=os.path.join('/media/xupengtao/新加卷/datasets/目标检测/PASCAL_VOC/VOCtest_06-Nov-2007/VOCdevkit/VOC2007')
#存pikle文件
CACHE_PATH = os.path.join('cache')


OUTPUT_DIR = os.path.join('output')

#WEIGHTS_FILE = None   #从头训练
#WEIGHTS_FILE = os.path.join(OUTPUT_DIR, '2019_08_18_10_38','yolo-1000')#从自己之前的模型训练，测试
WEIGHTS_FILE = os.path.join('weights', 'YOLO_small.ckpt')#从YOLO_small 模型训练

CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']

FLIPPED = True  #翻转增加数据


#
# model parameter
#

IMAGE_SIZE = 448

CELL_SIZE = 7

BOXES_PER_CELL = 2


OBJECT_SCALE = 1.0
NOOBJECT_SCALE = 1.0
CLASS_SCALE = 2.0
COORD_SCALE = 5.0


#
# train parameter
#

GPU = '0'

LEARNING_RATE = 0.0001

DECAY_STEPS = 30000

DECAY_RATE = 0.1

STAIRCASE = True

BATCH_SIZE = 10

MAX_ITER = 15000

SUMMARY_ITER = 10

SAVE_ITER = 1000


#
# test parameter
#

THRESHOLD = 0.2

IOU_THRESHOLD = 0.4
