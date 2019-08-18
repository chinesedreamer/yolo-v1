import yolo
import config as cfg
import tensorflow as tf
yolo_net = yolo.Yolo()
print("Start to load weights from file:%s" % (cfg.WEIGHTS_FILE))
saver = tf.train.Saver()
saver.restore(yolo_net.sess, cfg.WEIGHTS_FILE)
yolo_net.detect_from_file("./test/bus.jpg")
