import math
import numpy as np
import tensorflow as tf
from tensorflow.python import control_flow_ops

def batch_norm(x, n_out, phase_train, scope='bn'):
  """
  Batch normalization on convolutional maps.
  Args:
    x:           Tensor, 4D BHWD input maps
    n_out:       integer, depth of input maps
    phase_train: boolean tf.Varialbe, true indicates training phase
    scope:       string, variable scope
  Return:
    normed:      batch-normalized maps
  """
  with tf.variable_scope(scope):
    beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                       name='beta', trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                        name='gamma', trainable=True)
    batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.5)

    def mean_var_with_update():
      ema_apply_op = ema.apply([batch_mean, batch_var])
      with tf.control_dependencies([ema_apply_op]):
        return tf.identity(batch_mean), tf.identity(batch_var)
    mean, var = tf.cond(phase_train,
                        mean_var_with_update,
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
  return normed