import tensorflow as tf
import cv2
import numpy as np
import config as cfg

def leak_relu(x, alpha=0.1):#激活函数
    return tf.maximum(alpha * x, x)
class Yolo():
    def __init__(self,train=False):
        self.image_size=cfg.IMAGE_SIZE
        self.batch_size=cfg.BATCH_SIZE
        # detection params
        self.S = cfg.CELL_SIZE  # cell size
        self.B = cfg.BOXES_PER_CELL  # boxes_per_cell
        self.classes = cfg.CLASSES#classes
        self.C = len(self.classes) # number of classes
        self.idx1 = self.S * self.S * self.C  # 从这个索引，边界框置信度  前面是类别
        self.idx2 = self.idx1 + self.S * self.S * self.B  # 从这个索引   边界框预测
        # 每个单元格左上的x,y坐标
        self.x_offset = np.transpose(np.reshape(np.array([np.arange(self.S)]*self.S*self.B),
                                                [self.B, self.S, self.S]), [1, 2, 0])
        #arange函数用于创建等差数组   S1--->【0,1,2,.....,S-1】
        #np.array    变成S2*B行【0,1,2,.....,S-1】   二维数组-->[B*S2,S1]
        #reshape       三维数组  [B,S2,S1]
        #transpose    维度转化   【S2，S1，B】   第二维是【0,1,2,.....,S-1】，第一、三维相同
        self.y_offset = np.transpose(self.x_offset, [1, 0, 2])#S1,S2,B
        #最终得到  （0，0） （1，0）等一系列坐标，相同的坐标有B个
        self.threshold = cfg.THRESHOLD # confidence scores threhold
        self.iou_threshold = cfg.IOU_THRESHOLD
        #  the maximum number of boxes to be selected by non max suppression
        self.max_output_size = 10
        self.sess = tf.Session()
        self._build_net()  #建立网络
        if train:
            self.class_scale=cfg.CLASS_SCALE#分类损失比例
            self.object_scale=cfg.OBJECT_SCALE
            self.noobject_scale=cfg.NOOBJECT_SCALE
            self.coord_scale=cfg.COORD_SCALE
            self.labels = tf.placeholder(
                tf.float32,
                [None, self.S, self.S, 5 + self.C])
            self._loss(self.predicts, self.labels)
            self.loss = tf.losses.get_total_loss()
            tf.summary.scalar('loss', self.loss)

    def _build_net(self):
        """build the network"""
        print("Start to build the network ...")
        self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3])
        net = self._conv_layer(self.images, 1, 64, 7, 2)
        net = self._maxpool_layer(net, 1, 2, 2)
        net = self._conv_layer(net, 2, 192, 3, 1)
        net = self._maxpool_layer(net, 2, 2, 2)
        net = self._conv_layer(net, 3, 128, 1, 1)
        net = self._conv_layer(net, 4, 256, 3, 1)
        net = self._conv_layer(net, 5, 256, 1, 1)
        net = self._conv_layer(net, 6, 512, 3, 1)
        net = self._maxpool_layer(net, 6, 2, 2)
        net = self._conv_layer(net, 7, 256, 1, 1)
        net = self._conv_layer(net, 8, 512, 3, 1)
        net = self._conv_layer(net, 9, 256, 1, 1)
        net = self._conv_layer(net, 10, 512, 3, 1)
        net = self._conv_layer(net, 11, 256, 1, 1)
        net = self._conv_layer(net, 12, 512, 3, 1)
        net = self._conv_layer(net, 13, 256, 1, 1)
        net = self._conv_layer(net, 14, 512, 3, 1)
        net = self._conv_layer(net, 15, 512, 1, 1)
        net = self._conv_layer(net, 16, 1024, 3, 1)
        net = self._maxpool_layer(net, 16, 2, 2)
        net = self._conv_layer(net, 17, 512, 1, 1)
        net = self._conv_layer(net, 18, 1024, 3, 1)
        net = self._conv_layer(net, 19, 512, 1, 1)
        net = self._conv_layer(net, 20, 1024, 3, 1)
        net = self._conv_layer(net, 21, 1024, 3, 1)
        net = self._conv_layer(net, 22, 1024, 3, 2)
        net = self._conv_layer(net, 23, 1024, 3, 1)
        net = self._conv_layer(net, 24, 1024, 3, 1)
        net = self._flatten(net)
        net = self._fc_layer(net, 25, 512, activation=leak_relu)
        net = self._fc_layer(net, 26, 4096, activation=leak_relu)
        net = self._fc_layer(net, 27, self.S*self.S*(self.C+5*self.B))
        self.predicts = net
    def _conv_layer(self, x, id, num_filters, filter_size, stride):
        # 输入,层数,卷积核个数，卷积核大小，步长
        """Conv layer"""
        in_channels = x.get_shape().as_list()[-1]
        # 权重初始化  标准化    开始初始化stddev=0.1,从零开始训练模型时出现Nan,后来将stddev改为了0.01,即减少了权重间差异,避免Nan的出现
        weight = tf.Variable(tf.truncated_normal([filter_size, filter_size,
                                                  in_channels, num_filters], stddev=0.01))
        #偏置初始化  零
        bias = tf.Variable(tf.zeros([num_filters,]))
        # padding, note: not using padding="SAME"
        pad_size = filter_size // 2#平方
        #在w,H通道padding     Batch_size,W,H,Channels
        pad_mat = np.array([[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]])
        x_pad = tf.pad(x, pad_mat)
        conv = tf.nn.conv2d(x_pad, weight, strides=[1, stride, stride, 1], padding="VALID")
        output = leak_relu(tf.nn.bias_add(conv, bias))
        print("    Layer %d: type=Conv, num_filter=%d, filter_size=%d, stride=%d, output_shape=%s" \
                  % (id, num_filters, filter_size, stride, str(output.get_shape())))
        return output

    def _fc_layer(self, x, id, num_out, activation=None):
        """fully connected layer"""
        num_in = x.get_shape().as_list()[-1]
        weight = tf.Variable(tf.truncated_normal([num_in, num_out], stddev=0.1))
        bias = tf.Variable(tf.zeros([num_out,]))
        output = tf.nn.xw_plus_b(x, weight, bias)
        if activation:
            output = activation(output)
        print("    Layer %d: type=Fc, num_out=%d, output_shape=%s" \
                  % (id, num_out, str(output.get_shape())))
        return output

    def _maxpool_layer(self, x, id, pool_size, stride):
        output = tf.nn.max_pool(x, [1, pool_size, pool_size, 1],
                                strides=[1, stride, stride, 1], padding="SAME")
        print("    Layer %d: type=MaxPool, pool_size=%d, stride=%d, output_shape=%s" \
                  % (id, pool_size, stride, str(output.get_shape())))
        return output

    def _flatten(self, x):
        """flatten the x"""
        tran_x = tf.transpose(x, [0, 3, 1, 2])  # channle first mode   Batch_sizes,C,W,H
        nums = np.product(x.get_shape().as_list()[1:])#C,W,H相乘
        return tf.reshape(tran_x, [-1, nums])#flatten

    def calc_iou(self, boxes1, boxes2, scope='iou'):
        """calculate ious
        Args:
          boxes1: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
          boxes2: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4] ===> (x_center, y_center, w, h)
        Return:
          iou: 4-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        """
        with tf.variable_scope(scope):
            # transform (x_center, y_center, w, h) to (x1, y1, x2, y2)
            boxes1_t = tf.stack([boxes1[..., 0] - boxes1[..., 2] / 2.0,
                                 boxes1[..., 1] - boxes1[..., 3] / 2.0,
                                 boxes1[..., 0] + boxes1[..., 2] / 2.0,
                                 boxes1[..., 1] + boxes1[..., 3] / 2.0],
                                axis=-1)

            boxes2_t = tf.stack([boxes2[..., 0] - boxes2[..., 2] / 2.0,
                                 boxes2[..., 1] - boxes2[..., 3] / 2.0,
                                 boxes2[..., 0] + boxes2[..., 2] / 2.0,
                                 boxes2[..., 1] + boxes2[..., 3] / 2.0],
                                axis=-1)

            # calculate the left up point & right down point
            lu = tf.maximum(boxes1_t[..., :2], boxes2_t[..., :2])
            rd = tf.minimum(boxes1_t[..., 2:], boxes2_t[..., 2:])

            # intersection
            intersection = tf.maximum(0.0, rd - lu)
            inter_square = intersection[..., 0] * intersection[..., 1]

            # calculate the boxs1 square and boxs2 square
            square1 = boxes1[..., 2] * boxes1[..., 3]
            square2 = boxes2[..., 2] * boxes2[..., 3]

            union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

        return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)  #确保0,1之间
    def _loss(self, predicts, labels, scope='loss'):
        with tf.variable_scope(scope):
            predict_classes = tf.reshape(
                predicts[:, :self.idx1],
                [self.batch_size, self.S, self.S, self.C])#类别
            predict_scales = tf.reshape(
                predicts[:, self.idx1:self.idx2],
                [self.batch_size, self.S, self.S, self.B])#置信度
            predict_boxes = tf.reshape(
                predicts[:, self.idx2:],
                [self.batch_size, self.S, self.S, self.B, 4])#预测框  相对于单元格的中心坐标，和相对于整个图片开方的wh值   都在0，1之间
            # convert the x, y to the coordinates relative to the top left point of the image
            # the predictions of w, h are the square root
            offset_x = tf.reshape(
                tf.constant(self.x_offset, dtype=tf.float32),
                [1, self.S, self.S, self.B])
            offset_y = tf.reshape(
                tf.constant(self.y_offset, dtype=tf.float32),
                [1, self.S, self.S, self.B])
            predict_boxes_tran = tf.stack([(predict_boxes[: ,:, :, :, 0] + tf.tile(offset_x, [self.batch_size, 1, 1, 1]) )/ self.S ,
                              (predict_boxes[:, :,:, :, 1] + tf.tile(offset_y, [self.batch_size, 1, 1, 1])) / self.S,
                              tf.square(predict_boxes[:,:, :, :, 2]) ,
                              tf.square(predict_boxes[:,:, :, :, 3]) ], axis=-1)  #相对于整个图片的中心坐标，和wh值  归一化的
            # 提取标签边框的置信度IOU    实际上是0或1   0无目标  1有目标
            response = tf.reshape(
                labels[..., 0],# labels-->[Batch_size,S,S,C+5]  labels[...,0]--->[batch_size, S, S, 1]
                [self.batch_size, self.S, self.S, 1])
            #  提取标签边框
            boxes = tf.reshape(
                labels[..., 1:5],
                [self.batch_size, self.S, self.S, 1, 4])
            # 标签边框维度4复制了boxes_per_cell次,(batch, 7, 7, 1, 4)->(batch, 7, 7, 2, 4)
            # 边框归一化在这里，和yolo的voc_label生成的数据格式一致
            boxes = tf.tile(
                boxes, [1, 1, 1, self.B, 1]) / self.image_size  # #相对于整个图片的中心坐标，和wh值   归一化
            classes = labels[..., 5:]#类别

            # 计算的结果用作标签的边框置信度（IOU），在线计算，而实际的标签边框置信度IOU(response)，是用作有无目标的过滤器
            iou_predict_truth = self.calc_iou(predict_boxes_tran, boxes)#[Batch_size,S,S,B]

            # calculate I tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]   目标tensor
            # 求出各自边框中的最大值(batch, S, S, B)->(batch, S, S,1 )
            object_mask = tf.reduce_max(iou_predict_truth, 3, keepdims=True)
            object_mask = tf.cast(
                (iou_predict_truth >= object_mask), tf.float32) * response#有目标的单元格为1，否则为0

            # calculate no_I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]  取反
            noobject_mask = tf.ones_like(
                object_mask, dtype=tf.float32) - object_mask

            boxes_tran = tf.stack(
                [boxes[..., 0] * self.S - tf.tile(offset_x, [self.batch_size, 1, 1, 1]),
                 boxes[..., 1] * self.S - tf.tile(offset_y, [self.batch_size, 1, 1, 1]),
                 tf.sqrt(boxes[..., 2]),
                 tf.sqrt(boxes[..., 3])], axis=-1) #相对于单元格的中心坐标，和相对于整个图片开方的wh值

            # class_loss
            class_delta = response * (predict_classes - classes)#[batch,S,S,C]
            class_loss = tf.reduce_mean(#均方差损失
                tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]),#只留batch维度
                name='class_loss') * self.class_scale

            # object_loss
            object_delta = object_mask * (predict_scales - iou_predict_truth)
            object_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(object_delta), axis=[1, 2, 3]),
                name='object_loss') * self.object_scale

            # noobject_loss
            noobject_delta = noobject_mask * predict_scales#相当于减0
            noobject_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(noobject_delta), axis=[1, 2, 3]),
                name='noobject_loss') * self.noobject_scale

            # coord_loss
            coord_mask = tf.expand_dims(object_mask, 4)
            boxes_delta = coord_mask * (predict_boxes - boxes_tran)
            coord_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(boxes_delta), axis=[1, 2, 3, 4]),
                name='coord_loss') * self.coord_scale

            tf.losses.add_loss(class_loss)
            tf.losses.add_loss(object_loss)
            tf.losses.add_loss(noobject_loss)
            tf.losses.add_loss(coord_loss)

            tf.summary.scalar('class_loss', class_loss)
            tf.summary.scalar('object_loss', object_loss)
            tf.summary.scalar('noobject_loss', noobject_loss)
            tf.summary.scalar('coord_loss', coord_loss)

            tf.summary.histogram('boxes_delta_x', boxes_delta[..., 0])
            tf.summary.histogram('boxes_delta_y', boxes_delta[..., 1])
            tf.summary.histogram('boxes_delta_w', boxes_delta[..., 2])
            tf.summary.histogram('boxes_delta_h', boxes_delta[..., 3])
            tf.summary.histogram('iou', iou_predict_truth)

    #测试
    def detect_from_file(self, image_file, imshow=True, deteted_boxes_file="boxes.txt",
                     detected_image_file="detected_image.jpg"):
        """Do detection given a image file"""
        # read image
        image = cv2.imread(image_file)
        img_h, img_w, _ = image.shape
        predicts = self._detect_from_image(image)
        predict_boxes = self._interpret_predicts(predicts, img_h, img_w)
        self.show_results(image, predict_boxes, imshow, deteted_boxes_file, detected_image_file)

    def _detect_from_image(self, image):
        """Do detection given a cv image"""
        img_resized = cv2.resize(image, (448, 448))
        img_RGB = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_resized_np = np.asarray(img_RGB)
        _images = np.zeros((1, 448, 448, 3), dtype=np.float32)#加batch_szie=1
        _images[0] = (img_resized_np / 255.0) * 2.0 - 1.0#归一化，放止梯度爆炸等
        predicts = self.sess.run(self.predicts, feed_dict={self.images: _images})[0]#进入网络
        return predicts

    def _interpret_predicts(self, predicts, img_h, img_w):
        """Interpret the predicts and get the detetction boxes"""
        idx1 = self.S*self.S*self.C
        idx2 = idx1 + self.S*self.S*self.B
        # class prediction
        class_probs = np.reshape(predicts[:idx1], [self.S, self.S, self.C])
        # confidence
        confs = np.reshape(predicts[idx1:idx2], [self.S, self.S, self.B])
        # boxes -> (x, y, w, h)
        boxes = np.reshape(predicts[idx2:], [self.S, self.S, self.B, 4])

        # convert the x, y to the coordinates relative to the top left point of the image
        boxes[:, :, :, 0] += self.x_offset
        boxes[:, :, :, 1] += self.y_offset
        boxes[:, :, :, :2] /= self.S

        # the predictions of w, h are the square root
        boxes[:, :, :, 2:] = np.square(boxes[:, :, :, 2:])
        #上面操作是归一化到图片，下面乘以图片真实比例
        # multiply the width and height of image
        boxes[:, :, :, 0] *= img_w
        boxes[:, :, :, 1] *= img_h
        boxes[:, :, :, 2] *= img_w
        boxes[:, :, :, 3] *= img_h

        # class-specific confidence scores [S, S, B, C]
        #[s,s,b,1]*[s,s,1,c]--->[s,s,b,c]  得到每个预测框的class
        scores = np.expand_dims(confs, -1) * np.expand_dims(class_probs, 2)

        scores = np.reshape(scores, [-1, self.C]) # [S*S*B, C]
        boxes = np.reshape(boxes, [-1, 4])        # [S*S*B, 4]

        # filter the boxes when score < threhold
        scores[scores < self.threshold] = 0.0  #测试低于阙值置0，而不丢弃

        # non max suppression
        self._non_max_suppression(scores, boxes)

        # report the boxes
        predict_boxes = [] # (class, x, y, w, h, scores)
        max_idxs = np.argmax(scores, axis=1)
        for i in range(len(scores)):
            max_idx = max_idxs[i]
            if scores[i, max_idx] > 0.0:
                predict_boxes.append((self.classes[max_idx], boxes[i, 0], boxes[i, 1],
                                      boxes[i, 2], boxes[i, 3], scores[i, max_idx]))
        return predict_boxes

    def _non_max_suppression(self, scores, boxes):
        """Non max suppression"""
        # for each class
        for c in range(self.C):
            sorted_idxs = np.argsort(scores[:, c])#从小到大排序索引    找到该类别scores最大的一个框
            last = len(sorted_idxs) - 1#S*S*B-1
            while last > 0:#循环检测
                #如果最大的得分也很低，则表明没有这一类别的目标
                if scores[sorted_idxs[last], c] < 1e-6:
                    break
                for i in range(last):
                    if scores[sorted_idxs[i], c] < 1e-6:
                        continue
                    #如果与较大得分的框重合过多，该类别得分置0  只用得分较大的那个去预测  一对一
                    if self._iou(boxes[sorted_idxs[i]], boxes[sorted_idxs[last]]) > self.iou_threshold:
                        scores[sorted_idxs[i], c] = 0.0
                last -= 1



    def show_results(self, image, results, imshow=True, deteted_boxes_file=None,
                     detected_image_file=None):
        """Show the detection boxes"""
        img_cp = image.copy()
        if deteted_boxes_file:
            f = open(deteted_boxes_file, "w")
        #  draw boxes
        for i in range(len(results)):
            x = int(results[i][1])
            y = int(results[i][2])
            w = int(results[i][3]) // 2#//平方
            h = int(results[i][4]) // 2
            print("   class: %s, [x, y, w, h]=[%d, %d, %d, %d], confidence=%f" % (results[i][0],
                            x, y, w, h, results[i][-1]))

            cv2.rectangle(img_cp, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(img_cp, (x - w, y - h - 20), (x + w, y - h), (125, 125, 125), -1)#留一部分写字
            cv2.putText(img_cp, results[i][0] + ' : %.2f' % results[i][5], (x - w + 5, y - h - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            if deteted_boxes_file:
                f.write(results[i][0] + ',' + str(x) + ',' + str(y) + ',' +
                        str(w) + ',' + str(h)+',' + str(results[i][5]) + '\n')
        if imshow:
            cv2.imshow('YOLO_small detection', img_cp)
            cv2.waitKey(0)
        if detected_image_file:
            cv2.imwrite(detected_image_file, img_cp)
        if deteted_boxes_file:
            f.close()
    def _iou(self, box1, box2):
        """Compute the iou of two boxes"""

        inter_w = np.minimum(box1[0]+0.5*box1[2], box2[0]+0.5*box2[2]) - \
                  np.maximum(box1[0]-0.5*box2[2], box2[0]-0.5*box2[2])#重合部分w
        inter_h = np.minimum(box1[1]+0.5*box1[3], box2[1]+0.5*box2[3]) - \
                  np.maximum(box1[1]-0.5*box2[3], box2[1]-0.5*box2[3])#重合部分h
        if inter_h < 0 or inter_w < 0:#tensor不能作为判断条件
            inter = 0
        else:
            inter = inter_w * inter_h
        union = box1[2]*box1[3] + box2[2]*box2[3] - inter
        return inter / union