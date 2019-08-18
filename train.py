import os
import argparse
import datetime
import tensorflow as tf
import config as cfg
import yolo
import pascal_voc


class TrainNet():
    def __init__(self,net,data):
        self.net=net
        self.data=data
        self.weights_file=cfg.WEIGHTS_FILE
        self.max_iter=cfg.MAX_ITER
        self.initial_learning_rate=cfg.LEARNING_RATE
        self.decay_steps=cfg.DECAY_STEPS#开始将DECAY_STEPS写成了DECAY_RATE,导致学习率为0    以后注意，显示训练中学习率，防止学习率为0，导致loss不下降，不学习
        self.decay_rate = cfg.DECAY_RATE
        #staircase: 默认为False，此时学习率随迭代轮数的变化是连续的(指数函数)；为 True 时，global_step/decay_steps 会转化为整数，此时学习率便是阶梯函数
        self.staircase=cfg.STAIRCASE
        self.summary_iter=cfg.SUMMARY_ITER
        self.save_iter=cfg.SAVE_ITER
        self.output_dir=os.path.join(cfg.OUTPUT_DIR,datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.save_cfg()

        self.variable_to_restore=tf.global_variables()
        self.saver=tf.train.Saver(self.variable_to_restore)
        self.ckpt_file=os.path.join(self.output_dir,'yolo')
        # merge_all可以将所有summary全部保存到磁盘，以便tensorboard显示。如果没有特殊要求，一般用这一句就可一显示训练时的各种信息了。
        self.summary_op=tf.summary.merge_all()
        self.writer=tf.summary.FileWriter(self.output_dir,flush_secs=60)

        self.global_step=tf.train.create_global_step()
        #学习率变化
        self.learning_rate = tf.train.exponential_decay(
            self.initial_learning_rate, self.global_step, self.decay_steps,
            self.decay_rate, self.staircase, name='learning_rate')
        self.optimizer=tf.train.GradientDescentOptimizer(
            learning_rate=self.learning_rate)
        self.train_op=tf.contrib.slim.learning.create_train_op(
            self.net.loss,self.optimizer,global_step=self.global_step)

        gpu_options=tf.GPUOptions()
        config=tf.ConfigProto(gpu_options=gpu_options)
        self.sess=tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        if self.weights_file is not None:
            print('Restoring weights from: '+self.weights_file)
            self.saver.restore(self.sess,self.weights_file)

        self.writer.add_graph(self.sess.graph)

    def train(self):
        for step in range(1,self.max_iter+1):
            images,labels=self.data.get()# label:[self.batch_size, self.cell_size, self.cell_size, 5+self.C]
            feed_dict={self.net.images:images,
                       self.net.labels:labels}
            if step % self.summary_iter==0:
                if step % (self.summary_iter*10)==0:
                    summary_str,loss,_=self.sess.run(
                        [self.summary_op,self.net.loss,self.train_op],
                        feed_dict=feed_dict)
                    log_str = "{} Epoch: {}, Step: {}, Learning rate: {},Loss: {:5.3f}" \
                        .format(
                        datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
                        self.data.epoch,#数据第几遍遍历
                        int(step),
                        round(self.learning_rate.eval(session=self.sess), 6),
                        loss)
                    print(log_str)
                else:
                    summary_str, _ = self.sess.run(
                        [self.summary_op, self.train_op],
                        feed_dict=feed_dict)
                    self.writer.add_summary(summary_str, step)
            else:
                self.sess.run(self.train_op, feed_dict=feed_dict)
            if step % self.save_iter == 0:
                print('{} Saving checkpoint file to: {}'.format(
                    datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
                    self.output_dir))
                self.saver.save(
                    self.sess, self.ckpt_file, global_step=self.global_step)


    def save_cfg(self):#存configure文件
        with open(os.path.join(self.output_dir, 'config.txt'), 'w') as f:
            cfg_dict = cfg.__dict__
            for key in sorted(cfg_dict.keys()):
                if key[0].isupper():#是否为大写字母
                    cfg_str = '{}: {}\n'.format(key, cfg_dict[key])
                    f.write(cfg_str)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0', type=str)
    args = parser.parse_args()

    if args.gpu is not None:
        cfg.GPU = args.gpu



    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU

    yolo_net = yolo.Yolo(train=True)
    pascal = pascal_voc.pascal_voc(phase='train')

    trian_net = TrainNet(yolo_net, pascal)

    print('Start training ...')
    trian_net.train()
    print('Done training.')


if __name__ == '__main__':


    main()