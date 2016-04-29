# -*- coding: utf-8 -*-
"""
Created on April 29 10:43:29 2016

@author: Rob Romijnders
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
from bn_class import *

                            
"""Hyperparameters"""
# The graph is build with conv-pool blocks. One list as below denotes the settings
# for a conv-pool block as in [number_filters, kernel_size, pool_stride]
filt_1 = [30,5,3]       #Configuration for conv1 in [num_filt,kern_size,pool_stride]
filt_2 = [12,5,3]
num_fc_1 = 30        #Number of neurons in hully connected layer
max_iterations = 2000#Max iterations
batch_size = 50     # Batch size
dropout = 0.5       #Dropout rate in the fully connected layer
learning_rate = 1e-3
num_classes = 2     # Number of classes. Will be useful for multiple labels

"""Load the data"""
music = True
if music:
    #Load the csv. Due to the appending in Matlab, the first row is faulty
  data = np.loadtxt('data_music.csv',delimiter=',',skiprows=1)
else:
  pass

#Set up some indices for random split of train and testset
train = 0.7     #ratio for trainset
val = 0.85      #ratio for validation
N = data.shape[0]
ind_stop_train = int(train*N)
ind_stop_val = int(val*N)
ind = np.random.permutation(N)


# The first column contains the target labels
X_train = data[ind[:ind_stop_train],1:]
X_val = data[ind[ind_stop_train:ind_stop_val],1:]
X_test = data[ind[ind_stop_val:],1:]
N = X_train.shape[0]
Ntest = X_test.shape[0]
D = X_train.shape[1]
y_train = data[ind[:ind_stop_train],0]
y_val = data[ind[ind_stop_train:ind_stop_val],0]
y_test = data[ind[ind_stop_val:],0]
print('For training, we have %s observations with %s dimensions'%(N,D))

#Proclaim the epochs
epochs = np.floor(batch_size*max_iterations / N)
print('Train with approximately %d epochs' %(epochs))

# Nodes for the input variables (placeholders)
x = tf.placeholder("float", shape=[None, D], name = 'Input_data')
y_ = tf.placeholder(tf.int64, shape=[None], name = 'Ground_truth')
keep_prob = tf.placeholder("float", name = 'dropout_keep_prob')
bn_train = tf.placeholder(tf.bool)          #Boolean value to guide batchnorm
# More explanation on bn_train is below

# Define functions for initializing variables and standard layers
#For now, this seems superfluous, but in extending the code
#to many more layers, this will keep our code
#read-able
def weight_variable(shape, name):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name = name)

def bias_variable(shape, name):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name = name)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')



with tf.name_scope("Reshaping_data") as scope:
  x_image = tf.reshape(x, [-1,D,1,1])


"""Build the graph"""

with tf.name_scope("Conv1") as scope:
  W_conv1 = weight_variable([filt_1[1], 1, 1, filt_1[0]], 'Conv_Layer_1')
  b_conv1 = bias_variable([filt_1[0]], 'bias_for_Conv_Layer_1')
  a_conv1 = conv2d(x_image, W_conv1) + b_conv1
  h_conv1 = tf.nn.relu(a_conv1)

with tf.name_scope('max_pool1') as scope:
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, filt_1[2], 1, 1],
                        strides=[1, filt_1[2], 1, 1], padding='VALID')
                        #width is now (128-4)/2+1
    width_pool1 = int(np.floor((D-filt_1[2])/filt_1[2]))+1 
    size1 = tf.shape(h_pool1)       #Debugging purposes
    
    
with tf.name_scope("Conv2") as scope:
  W_conv2 = weight_variable([filt_2[1], 1, filt_1[0], filt_2[0]], 'Conv_Layer_2')
  b_conv2 = bias_variable([filt_2[0]], 'bias_for_Conv_Layer_2')
  a_conv2 = conv2d(h_pool1, W_conv2) + b_conv2
  h_conv2 = a_conv2
 # h_conv2 = tf.nn.relu(a_conv2) #ReLU after batchnorm
  
with tf.name_scope('max_pool2') as scope:
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, filt_2[2], 1, 1],
                        strides=[1, filt_2[2], 1, 1], padding='VALID')
                        #width is now (128-4)/2+1
    width_pool2 = int(np.floor((width_pool1-filt_2[2])/filt_2[2]))+1 
    size2 = tf.shape(h_pool2)       #Debugging purposes
  
with tf.name_scope('Batch_norm1') as scope:
# ewma is the decay for which we update the moving average of the 
# mean and variance in the batch-norm layers
# The placeholder bn_train denotes wether we are in train or testtime. 
# - In traintime, we update the mean and variance according to the statistics
#    of the batch
#  - In testtime, we use the moving average of the mean and variance. We do NOT
#     update 
    ewma = tf.train.ExponentialMovingAverage(decay=0.99)                  
    bn_conv1 = ConvolutionalBatchNormalizer(filt_2[0], 0.001, ewma, True)           
    update_assignments = bn_conv1.get_assigner() 
    a_bn1 = bn_conv1.normalize(h_pool2, train=bn_train) 
    h_bn1 = tf.nn.relu(a_bn1) 


with tf.name_scope("Fully_Connected1") as scope:
# Now we proces the final information with a fully connected layer. We convert
# both activations over all channels into one 1D tensor per sample.
# We have "filt_2[0]" channels and "width_pool2" activations per channel.
# Hence we use "width_pool2*filt_2[0]" i this first line
  W_fc1 = weight_variable([width_pool2*filt_2[0], num_fc_1], 'Fully_Connected_layer_1')
  b_fc1 = bias_variable([num_fc_1], 'bias_for_Fully_Connected_Layer_1')
  h_flat = tf.reshape(h_bn1, [-1, width_pool2*filt_2[0]])
  h_flat = tf.nn.dropout(h_flat,keep_prob)
  h_fc1 = tf.nn.relu(tf.matmul(h_flat, W_fc1) + b_fc1)
 
  
with tf.name_scope("Output_layer") as scope:
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
  W_fc2 = tf.Variable(tf.truncated_normal([num_fc_1, num_classes], stddev=0.1),name = 'W_fc2')
  b_fc2 = tf.Variable(tf.constant(0.1, shape=[num_classes]),name = 'b_fc2')
  h_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  size3 = tf.shape(h_fc2)       #Debugging purposes

with tf.name_scope("SoftMax") as scope:
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(h_fc2,y_)
  cost = tf.reduce_sum(loss) / batch_size
  loss_summ = tf.scalar_summary("cross entropy_loss", cost)
with tf.name_scope("train") as scope:
    tvars = tf.trainable_variables()
    grads = tf.gradients(cost, tvars)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = zip(grads, tvars)
    train_step = optimizer.apply_gradients(gradients)
    # The following block plots for every trainable variable
    #  - Histogram of the entries of the Tensor
    #  - Histogram of the gradient over the Tensor
    #  - Histogram of the grradient-norm over the Tensor
    numel = tf.constant([[0]])
    for gradient, variable in gradients:
      if isinstance(gradient, ops.IndexedSlices):
        grad_values = gradient.values
      else:
        grad_values = gradient
      
      numel +=tf.reduce_sum(tf.size(variable))  
        
      h1 = tf.histogram_summary(variable.name, variable)
      h2 = tf.histogram_summary(variable.name + "/gradients", grad_values)
      h3 = tf.histogram_summary(variable.name + "/gradient_norm", clip_ops.global_norm([grad_values]))
with tf.name_scope("Evaluating_accuracy") as scope:
    correct_prediction = tf.equal(tf.argmax(h_fc2,1), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    accuracy_summary = tf.scalar_summary("accuracy", accuracy)

""" Note on argmax and softmax"""
#In the two blocks of code above, we use softmax to generate a final disctribution.
#We use argmax to evaluate the accuracy. Both functions are superfluous for the
#binary case. However, we code in this way to allow for multiple labels in 
#future implementations.  
   
   

#Define one op to call all summaries    
merged = tf.merge_all_summaries()

# For now, we collect performances in a Numpy array.
# In future releases, I hope TensorBoard allows for more
# flexibility in plotting
perf_collect = np.zeros((4,int(np.floor(max_iterations /200))))

with tf.Session() as sess:
  writer = tf.train.SummaryWriter("/home/rob/Dropbox/ml_projects/music/log_tb", sess.graph_def)

  sess.run(tf.initialize_all_variables())
  
  step = 0      # Step is a counter for filling the numpy array perf_collect
  for i in range(max_iterations):
    batch_ind = np.random.choice(N,batch_size,replace=False)
    if i==0:
        # Use this line to check before-and-after test accuracy
        result = sess.run([accuracy], feed_dict={ x: X_test, y_: y_test, keep_prob: 1.0, bn_train : False})
        acc_test_before = result[0]

    if i%200 == 0:
      #Check training performance
      result = sess.run([accuracy,cost],feed_dict = { x: X_train, y_: y_train, keep_prob: 1.0, bn_train : False})
      perf_collect[1,step] = result[0]
      perf_collect[3,step] = result[1]
        
      #Check validation performance
      result = sess.run([accuracy,merged,cost], feed_dict={ x: X_val, y_: y_val, keep_prob: 1.0, bn_train : False})
      acc = result[0]
      perf_collect[0,step] = acc
      perf_collect[2,step] = result[2]
      
      #Write information to TensorBoard
      summary_str = result[1]
      writer.add_summary(summary_str, i)
      writer.flush()  #Don't forget this command! It makes sure Python writes the summaries to the log-file
      print(" Validation accuracy at %s out of %s is %s" % (i,max_iterations, acc))
      step +=1  
    sess.run(train_step,feed_dict={x:X_train[batch_ind], y_: y_train[batch_ind], keep_prob: dropout, bn_train : True})

  result = sess.run([accuracy,numel], feed_dict={ x: X_test, y_: y_test, keep_prob: 1.0, bn_train : False})
  acc_test = result[0]
  print('The network has %s trainable parameters'%(result[1]))
  
"""Additional plots"""
print('The accuracy on the test data is %.3f, before training was %.3f' %(acc_test,acc_test_before))
plt.figure()
plt.plot(perf_collect[0],label='Valid accuracy')
plt.plot(perf_collect[1],label = 'Train accuracy')
plt.legend()
plt.show()

plt.figure()
plt.plot(perf_collect[2],label='Valid cost')
plt.plot(perf_collect[3],label = 'Train cost')
plt.legend()
plt.show()

# We can now open TensorBoard. Run the following line from your terminal
# Change the log-dir to your own settings
# tensorboard --logdir=/home/rob/Dropbox/ml_projects/music/log_tb