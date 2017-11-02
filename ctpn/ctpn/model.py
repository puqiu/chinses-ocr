#from cfg import Config as cfg
import tensorflow as tf
import sys
import os
from cfg import Config
from other import resize_im
sys.path.append('ctpn')
from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg

from lib.fast_rcnn.test import  test_ctpn
from lib.fast_rcnn.nms_wrapper import nms



def load_tf_model():
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    # init session
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    # load network
    net = get_network("VGGnet_test")
    # load model
    #print ('Loading network {:s}... '.format("VGGnet_test")),
    saver = tf.train.Saver()
    saver.restore(sess, os.path.join(os.getcwd(),"ctpn/checkpoints/model_final_tf13.ckpt"))
    #print (' done.')
    return sess,saver,net
sess,saver,net = load_tf_model()
def ctpn(img):
    scale, max_scale = Config.SCALE,Config.MAX_SCALE
    img,f = resize_im(img,scale=scale,max_scale=max_scale)
    scores, boxes = test_ctpn(sess, net, img)
    return scores, boxes,img
