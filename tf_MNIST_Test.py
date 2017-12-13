# -*- coding: utf-8 -*-
'''
Created on 2017. 9. 8.

@author: danny
'''
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf
from PIL import Image, ImageFilter, ImageOps #pip3 install Pillow
#opencv-python
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt # imshow / show 보일려면 pyplot을 붙여야 한다. 
import pandas

sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

sess.run(tf.initialize_all_variables())
y = tf.nn.softmax(tf.matmul(x,W) + b)

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Second layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Densely Connected Layer

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# Train and Evaluate the Model

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())

saver = tf.train.Saver()
saver.restore(sess, 'E:\Pywork\WorkSpace\MNIST\model_dropout\mnist_model_dropout')

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})) #keep_prob : dropout 비율

src_path = "E:/Pywork/WorkSpace/MNIST/test/"

def Estimate(filename):
    imgread = Image.open(src_path+filename).convert("L") # convert RGB, CMYK, L(256단계 흑백 이미지), 1(단색 이미지) 
    img = Image.eval(imgread, lambda x:256-x)
    #img = Image.eval(imgread, lambda x : 0 if x>=128 else 255)
    
    width = float(img.size[0])
    height = float(img.size[1])
    #newImage = Image.new('L', (28, 28), (255))
    newImage = Image.new('L', (28, 28), (0))
   
    if width > height:
        # 폭이 더 큰 경우 처리 로직
        nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
         
        # 20/20 이미지로 변환하고
        img = img.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN) 
         
        #thr2 = cv2.adaptiveThreshold(newImage, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11,2)
        # 이진화 작업
         
        wtop = int(round(((28 - nheight) / 2), 0))  #
        newImage.paste(img, (4, wtop))  # 리사이즈된 이미지를 흰색 바탕의 캔버스에 붙여 넣는다
         
    else:
        # 높이가 더 큰경우에 처리 로직
        nwidth = int(round((20.0 / height * width), 0))
        if (nwidth == 0):
            nwidth = 1
        # resize and sharpen
        img = img.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))
        newImage.paste(img, (wleft, 4))
     
         
    #newImage = ImageOps.solarize(newImage, threshold=128)
    newImage = Image.eval(newImage, lambda x : 255 if x>=28 else 0) #binary
    tv = list(newImage.getdata()) 
    #tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    tva = [(x * 1.0 / 255.0) for x in tv]
    
    return (np.asfarray(tva).reshape((1,784)))
 
import pytesseract
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname = 'c:/Windows/Fonts/malgun.ttf').get_name()
rc('font', family = font_name)

while True:
    filename = input("Image 파일명을 입력하세요 (형식 xxxx.jpg): ")

    if os.path.exists(src_path+filename):
        predicted_vector = sess.run(y_conv, feed_dict={x: Estimate(filename), keep_prob: 1.0}) #model.predict(Estimate(filename), batch_size=1, verbose=1)
        predicted_class = predicted_vector.argmax(axis=-1)
        fig = plt.figure()

        gray = cv2.imread(src_path+filename) 
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        #gray = cv2.resize(gray, (28, 28))
        subplot2 = fig.add_subplot(2,2,1)
        subplot2.imshow(gray, cmap='Greys')#,interpolation = 'nearest')
        
        subplot3 = fig.add_subplot(2,2,2)
        subplot3.imshow(Estimate(filename).reshape(28,28), cmap='Greys')#,interpolation = 'nearest')

        pred = predicted_vector[0]
        subplot = fig.add_subplot(2,2,3)
        subplot.set_xticks(list(range(10)))
        subplot.set_ylim(0,1)
        subplot.bar(list(range(10)), pred, align='center')
        
       
        print("Prediction: ", sess.run(tf.argmax(y_conv,1), feed_dict={x: Estimate(filename), keep_prob: 1.0}))
        print("OCR: ",pytesseract.image_to_string(Image.open(src_path+filename),lang='eng').replace(" ",""))
        #plt.imshow(Estimate(filename).reshape(28,28), cmap='Greys', interpolation='nearest')
        #plt.show()

        #plt.title("손글씨 결과 ")
        plt.show()
        
    else:
        True
 
    if input("Continue? (y/n) : ") == "n":
        break
     
    
