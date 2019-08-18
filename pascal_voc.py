import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import pickle
import copy
import config as cfg


class pascal_voc():
    def __init__(self, phase, rebuild=False):
        if phase=='train':
            self.data_path = cfg.PASCAL_PATH
        else:
            self.data_path = cfg.PASCAL_TEST_PATH
        self.cache_path = cfg.CACHE_PATH
        self.batch_size = cfg.BATCH_SIZE
        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.classes = cfg.CLASSES
        self.C=len(self.classes)
        self.class_to_ind = dict(zip(self.classes, range(len(self.classes))))#类名对应数字编号
        self.flipped = cfg.FLIPPED#  是否翻转增加数据
        self.phase = phase#train   test
        self.rebuild = rebuild#是否重新构造cache_file
        self.cursor = 0#记录数据是否被用完
        self.epoch = 1#数据被用了第几轮
        self.gt_labels = None
        self.prepare()

    def get(self):
        images = np.zeros(
            (self.batch_size, self.image_size, self.image_size, 3))
        labels = np.zeros(
            (self.batch_size, self.cell_size, self.cell_size, 5+self.C))
        count = 0
        while count < self.batch_size:
            imname = self.gt_labels[self.cursor]['imname']
            flipped = self.gt_labels[self.cursor]['flipped']#是否翻转图片
            images[count, :, :, :] = self.image_read(imname, flipped)
            labels[count, :, :, :] = self.gt_labels[self.cursor]['label']
            count += 1
            self.cursor += 1
            if self.cursor >= len(self.gt_labels):#超过数据总量，shuffle后再来一轮
                np.random.shuffle(self.gt_labels)
                self.cursor = 0
                self.epoch += 1
        return images, labels

    def image_read(self, imname, flipped=False):
        image = cv2.imread(imname)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = (image / 255.0) * 2.0 - 1.0#归一化
        if flipped:
            image = image[:, ::-1, :]#图片翻转
        return image

    def prepare(self):
        gt_labels = self.load_labels()
        # 'imname': imname,
        # 'label': label,[self.cell_size, self.cell_size, self.C+5]
        # 'flipped': False
        if self.flipped:
            print('Appending horizontally-flipped training examples ...')#水平翻转加倍数据
            gt_labels_cp = copy.deepcopy(gt_labels)
            for idx in range(len(gt_labels_cp)):
                gt_labels_cp[idx]['flipped'] = True
                # 左右镜像需要将坐标x倒序排列，先倒序
                gt_labels_cp[idx]['label'] =\
                    gt_labels_cp[idx]['label'][:, ::-1, :]#第二维倒序   可以简单的认为将坐标原点移到了右边  相当于翻转
                for i in range(self.cell_size):
                    for j in range(self.cell_size):
                        # 倒序后，坐标值需做相应镜像
                        if gt_labels_cp[idx]['label'][i, j, 0] == 1:
                            #x坐标镜像
                            gt_labels_cp[idx]['label'][i, j, 1] = \
                                self.image_size - 1 -\
                                gt_labels_cp[idx]['label'][i, j, 1]
            gt_labels += gt_labels_cp
        np.random.shuffle(gt_labels)
        self.gt_labels = gt_labels
        return gt_labels

    def load_labels(self):
        cache_file = os.path.join(
            self.cache_path, 'pascal_' + self.phase + '_gt_labels.pkl')

        if os.path.isfile(cache_file) and not self.rebuild:
            print('Loading gt_labels from: ' + cache_file)
            with open(cache_file, 'rb') as f:
                gt_labels = pickle.load(f)
            return gt_labels

        print('Processing gt_labels from: ' + self.data_path)

        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

        if self.phase == 'train':
            txtname = os.path.join(
                self.data_path, 'ImageSets', 'Main', 'trainval.txt')
        else:
            txtname = os.path.join(
                self.data_path, 'ImageSets', 'Main', 'test.txt')
        with open(txtname, 'r') as f:
            self.image_index = [x.strip() for x in f.readlines()]#图片编号

        gt_labels = []
        for index in self.image_index:
            label, num = self.load_pascal_annotation(index)#从xml文件加载
            if num == 0:
                continue
            imname = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')
            gt_labels.append({'imname': imname,
                              'label': label,
                              'flipped': False})
        print('Saving gt_labels to: ' + cache_file)
        with open(cache_file, 'wb') as f:
            pickle.dump(gt_labels, f)
        return gt_labels

    def load_pascal_annotation(self, index):
        """
        从xml文件中得到label[self.cell_size, self.cell_size, self.C+5]和目标个数.
        """

        imname = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')
        im = cv2.imread(imname)
        h_ratio = 1.0 * self.image_size / im.shape[0]
        w_ratio = 1.0 * self.image_size / im.shape[1]


        label = np.zeros((self.cell_size, self.cell_size, self.C+5))#  有无目标1   边界框4    类别C
        filename = os.path.join(self.data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')

        for obj in objs:
            bbox = obj.find('bndbox')
            #找到目标框
            x1 = max(min((float(bbox.find('xmin').text) - 1) * w_ratio, self.image_size - 1), 0)
            y1 = max(min((float(bbox.find('ymin').text) - 1) * h_ratio, self.image_size - 1), 0)
            x2 = max(min((float(bbox.find('xmax').text) - 1) * w_ratio, self.image_size - 1), 0)
            y2 = max(min((float(bbox.find('ymax').text) - 1) * h_ratio, self.image_size - 1), 0)
            #类别编号
            cls_ind = self.class_to_ind[obj.find('name').text.lower().strip()]#.lower()转为小写字母   .strip()移除字符串头尾指定的字符(默认为空格或换行符)或字符序列
            #目标框   未作归一化
            boxes = [(x2 + x1) / 2.0, (y2 + y1) / 2.0, x2 - x1, y2 - y1] #相对于整个图片的中心坐标，和wh值
            #判断该中心点在哪个单元格里
            x_ind = int(boxes[0] * self.cell_size / self.image_size)
            y_ind = int(boxes[1] * self.cell_size / self.image_size)
            # 只能接受最先提供给label的一类物体，后面重复的则忽略    也就是说一个单元格只能对应一个目标，不足之处
            if label[y_ind, x_ind, 0] == 1:
                continue
            label[y_ind, x_ind, 0] = 1   #有目标
            label[y_ind, x_ind, 1:5] = boxes
            label[y_ind, x_ind, 5 + cls_ind] = 1

        return label, len(objs)