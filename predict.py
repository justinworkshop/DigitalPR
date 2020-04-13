# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 15:59:50 2020

@author: wei.zheng
"""

import sys
import tensorflow as tf
from PIL import Image,ImageFilter
import matplotlib.pyplot as plt
import numpy as np

def predictint(imvalue):
    """
    This function returns a predicted integer.
    """
    
    # Define the model (same as when creating the model file)
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]), name="w")
    b = tf.Variable(tf.zeros([10]), name="b")
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:

       
        saver.restore(sess, "tmp_mnist/model.ckpt")
        print ("Model restored.")

        prediction=tf.argmax(y,1)
     
        return prediction.eval(feed_dict={x: [imvalue]}, session=sess)


def imageprepare(argv):
    """
    This function returns the numpy values.
    """
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    img = im.resize((28,28), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
    data = img.getdata()
    print("data:",data)
    data = (255.0-data)/255.0
    new_data = np.reshape(data,(1,28*28))

    return new_data
    #print(tva)

	
def main(argv=None):
    """
    Main function.
    """
    path = "map/testmap0.jpg"
    
    imvalue = imageprepare(path)
    
    imvalue = np.array(imvalue)
    print("imvalue.shape:",imvalue.shape)
    print("----------------------------")
    predint = predictint(imvalue)
    print ("result:",predint[0]) 
    
if __name__ == "__main__":
    main()