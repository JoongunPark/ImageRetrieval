########################################################################################
# Davi Frossard, 2016                                                                  #
# VGG16 implementation in TensorFlow                                                   #
# Details:                                                                             #
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
#                                                                                      #
# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md     #
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow      #
########################################################################################

import numpy as np
from os import listdir
from os.path import join
from scipy.misc import imread, imresize
from PIL import Image
import pickle
import tensorflow as tf
import random

class vgg16:
    def __init__(self, imgs, weights=None, sess=None):
        self.imgs = imgs
        self.convlayers()
        self.fc_layers()
#        self.probs_attribute = tf.nn.sigmoid(self.fc4l1)
        self.probs_category = tf.nn.softmax(self.fc4l2)
	self.trainning()
        if weights is not None and sess is not None:
            self.load_weights(weights, sess)


    def convlayers(self):
        self.parameters = []

        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = self.imgs-mean

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

    def fc_layers(self):
        # fc1
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            fc1w = tf.Variable(tf.truncated_normal([shape, 4096],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                 trainable=True, name='biases')
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
            self.fc1 = tf.nn.relu(fc1l)
            self.parameters += [fc1w, fc1b]

        # fc2
        with tf.name_scope('fc2') as scope:
            fc2w = tf.Variable(tf.truncated_normal([4096, 4096],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                 trainable=True, name='biases')
            fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
            self.fc2 = tf.nn.relu(fc2l)
            self.parameters += [fc2w, fc2b]

        # fc3
        with tf.name_scope('fc3') as scope:
            fc3w = tf.Variable(tf.truncated_normal([4096, 48],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc3b = tf.Variable(tf.constant(1.0, shape=[48], dtype=tf.float32),
                                 trainable=True, name='biases')
            fc3l = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)
            self.fc3 = tf.nn.relu(fc3l)
            self.parameters += [fc3w, fc3b]
	
        # fc4-1
        with tf.name_scope('fc41') as scope:
            fc4w1 = tf.Variable(tf.truncated_normal([48,1000],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc4b1 = tf.Variable(tf.constant(1.0, shape=[1000], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.fc4l1 = tf.nn.bias_add(tf.matmul(self.fc3, fc4w1), fc4b1)
            self.parameters += [fc4w1, fc4b1]

        # fc4-2
        with tf.name_scope('fc42') as scope:
            fc4w2 = tf.Variable(tf.truncated_normal([48,50],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc4b2 = tf.Variable(tf.constant(1.0, shape=[50], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.fc4l2 = tf.nn.bias_add(tf.matmul(self.fc3, fc4w2), fc4b2)
            self.parameters += [fc4w2, fc4b2]

    def load_weights(self, weight_file, sess):
#        weights = np.load(weight_file)
#        keys = sorted(weights.keys())
	print ('Load weights...')
    
	#initialize before load pretrained model
#	sess.run(tf.initialize_all_variables())

#        for i, k in enumerate(keys):
#	
#	    #remove f8 layer 
#	    if i > 29:
#	        break
#            sess.run(self.parameters[i].assign(weights[k]))

        saver = tf.train.Saver()
        saver.restore(sess, "fine-tunning-suffler_0000061.ckpt")

	print ('Load complete.')


    def trainning(self):

        weight = open('attr_file.txt', 'r')
	attr_weight = [float(i) for i in weight.readline().split()[1:1000]]
	weight.close()

        #train step
        #cross_entropy = -tf.reduce_sum(category_*tf.log(self.probs_category))

	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.fc4l2, category_))
        #cross_entorpy2 = -tf.reduce_sum(attribute_*tf.log(self.probs_attribute))

	# for weight entropy
	#cross_entropy2 = tf.contrib.losses.sigmoid_cross_entropy(self.probs_attribute, attribute_, attr_weight, label_smoothing=0,scope=None)
	wneg = tf.constant(0.0033268346541)
	cross_entropy2 = tf.mul(tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(self.fc4l1, attribute_, 299.58602364)), wneg)

	self.loss = tf.add(cross_entropy, cross_entropy2) 
	#self.loss = cross_entropy	

        self.train_step = tf.train.AdamOptimizer(0.00004).minimize(self.loss)
        
        correct_prediction = tf.equal(tf.arg_max(self.probs_category,1), tf.arg_max(category_,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

    def trainImage(self, sess, batch1, batch2, batch3):

#	batch_imgs = np.reshape(batch1, (-1, 224, 224, 3))
#	batch_category = np.reshape(batch2, (-1, 50))
#	batch_attribute = np.reshape(batch3, (-1, 1000))
        self.train_step.run(session=sess, feed_dict={vgg.imgs: batch1 ,category_: batch2, attribute_: batch3}) 

    def evalImage(self, sess, img, category_label, attribute_label):
        loss = self.loss.eval(session=sess, feed_dict={vgg.imgs: [img1],category_:category_label, attribute_:attribute_label})
        print "loss" , loss 



if __name__ == '__main__':
    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    category_ = tf.placeholder(tf.float32, [None, 50])
    attribute_ = tf.placeholder(tf.float32, [None, 1000])

    vgg = vgg16(imgs, 'vgg16_weights.npz', sess)


    # Change dataset directory path
#    data_dir = 'query'
#    datalist = [join(data_dir, f) for f in listdir(data_dir)]
#
#    res = np.zeros((1, 4096))

    suffler = 0#-1 

    for e in range(2,100):

	suffler += 1
	suffler %= 20 	

        list_eval = open('deepfashion/list_eval_partition.txt', 'r')
        category = open('deepfashion/list_category_img.txt', 'r')
        attribute = open('deepfashion/list_attr_img.txt', 'r')
    
        num_images = int(list_eval.readline().strip())
        list_eval.readline()
    
        category.readline()
        category.readline()
        attribute.readline()
        attribute.readline()
    
    #    f = open('query_features', 'w')
    #    f.write("256 4096\n")
    #    features = []
    #    datalist = sorted(datalist)
	batch = [[],[],[]]
	batch_counter = 0

#    	parsed_eval = list_eval.readline().split()
#        parsed_category = category.readline().split()
#        parsed_attribute = attribute.readline().split()
#
	train_counter = 0
        for index in range(0,num_images):
    
    		parsed_eval = list_eval.readline().split()
        	parsed_category = category.readline().split()
        	parsed_attribute = attribute.readline().split()
    	
    		try:	
    		    filename = parsed_eval[0]
    		    imtype = parsed_eval[1]
    		except:
    		    break
    
    		if imtype=="train":
		    
		    train_counter+=1
		    if train_counter % 20 != suffler:
		        continue

		    print index

    		    img1 = Image.open(filename)
             	    img1 = img1.resize((224,224), Image.BILINEAR)
        	    # Convert Image object to ndarray
        	    img1 = np.array(img1.getdata()).reshape(img1.size[0], img1.size[1], 3)
    
    		    a = [0] * 50
    		    a[int(parsed_category[1])-1] = 1
    		
#		    batch[0].append([img1])
#		    batch[1].append([a])	
#		    batch[2].append(parsed_attribute[1:1001])
#
                    vgg.trainImage(sess, [img1], [a], [parsed_attribute[1:1001]])
#                    vgg.evalImage(sess, img1, [a], [parsed_attribute[1:1001]])
		    
    #		    category_array = np.zeros((1, 50))   				 
    #		    attribute_arrary = np.array(parsed_attribute[1:1001])
    
    
    #                else:
    #		    img1 = Image.open(filename)
    #         	    img1 = img1.resize((224,224), Image.BILINEAR)
    #    		    # Convert Image object to ndarray
    #    		    img1 = np.array(img1.getdata()).reshape(img1.size[0], img1.size[1], 3)
    #
    #		    a = [0] * 50
    #		    a[int(parsed_category[1])-1] = 1
    #
    #                    vgg.evalImage(sess, img1, [a], [parsed_attribute[1:1001]])
    #	
    #		try:
    #			img1.load()
    #		except IOError as e:
    #			print e
    		
    		 
    
    #		train_accuracy = accuracy.eval(feed_dict={vgg.imgs: [img1],y_:batch[1],keep_prob:1.0}) 
    #		print "step %d, training accuracy %g" % (i,train_accuracy) 
    
    #		# Extract image descriptor in layer fc2/Relu. If you want, change fc2 to fc1
    #		layer = sess.graph.get_tensor_by_name('fc2/Relu:0')
    #		layer2 = sess.graph.get_tensor_by_name('fc3/Relu:0')
    #
    #		# Run the session for feature extract at 'fc2/Relu' layer
    #		[_feature, _feature2] = sess.run([layer, layer2], feed_dict={vgg.imgs: [img1]})
    #
    #		# Convert tensor variable into numpy array
    #		# It is 4096 dimension vector
    #		feature = np.array(_feature) 
    #		feature2 = np.array(_feature2) 
    #		print feature
    #		print feature2
    
    		# Write the code to save descriptor
    		# ....
    #		feat = feature[0]
    #		for i in feat:
    #			f.write(str(i))
    #		f.write("\n")
        list_eval.close()
	category.close()
	attribute.close()

        saver = tf.train.Saver() 
        save_path = saver.save(sess, "./fine-tunning-suffler_000004"+str(e)+".ckpt")
	print str(e), "is done"
