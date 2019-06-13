import itertools
import numpy as np
from scipy.special import comb

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import BatchNormalization


class ProjectLayer(tf.keras.layers.Layer):

    def __init__(self, input_num, subnet_num, l1_proj=0.0, method="random"):
        super(ProjectLayer, self).__init__()
        self.input_num = input_num
        self.subnet_num = subnet_num
        self.l1_proj = l1_proj
        self.method = method
        self.trainable = True

        if self.method == "random":
            self.kernel_iniializer = tf.keras.initializers.RandomNormal()

        elif self.method == "comb":
            init = np.zeros((self.input_num, self.subnet_num), dtype=np.float32)
            cct = 0
            kmax = 0
            while cct < self.subnet_num:
                kmax += 1
                cct += comb(self.input_num, kmax)
            itr = itertools.combinations(range(self.input_num), 1)
            for k in range(2, kmax + 1):
                itr = itertools.chain(itr, itertools.combinations(range(self.input_num), k))
            for i, v in enumerate(itr):
                if i < self.subnet_num:
                    init[np.array(v), i] = 1.0
                else:
                    pass
            self.kernel_iniializer = tf.keras.initializers.Constant(init)

        elif self.method == "orthogonal":
            self.kernel_iniializer = tf.keras.initializers.orthogonal()

        elif self.method == "gam":
            self.kernel_iniializer = tf.keras.initializers.constant(np.eye(self.input_num))
            self.trainable = False

    def build(self, input_shape=None):
        self.proj_weights = self.add_weight(name="proj_weights",
                                            shape=[self.input_num, self.subnet_num],
                                            dtype=tf.float32,
                                            initializer=self.kernel_iniializer,
                                            trainable=self.trainable,
                                            regularizer=tf.keras.regularizers.l1(self.l1_proj))

    def call(self, inputs, training=False):
        output = tf.matmul(inputs, self.proj_weights)
        return output


class Subnetwork(tf.keras.layers.Layer):

    def __init__(self, subnet_arch=[10, 6], activation_func=tf.tanh, smooth_lambda=0.0, bn_flag=False):
        super(Subnetwork, self).__init__()
        self.dense = []
        self.subnet_arch = subnet_arch
        self.activation_func = activation_func
        self.smooth_lambda = smooth_lambda
        self.bn_flag = bn_flag

    def build(self, input_shape=None):
        for nodes in self.subnet_arch:
            self.dense.append(layers.Dense(nodes, activation=self.activation_func))
        self.output_layer = layers.Dense(1, activation=tf.identity)
        self.subnet_bn = BatchNormalization(momentum=0.0, epsilon=1e-10, center=False, scale=False)

    def call(self, inputs, training=False):
        with tf.GradientTape() as t1:
            t1.watch(inputs)
            with tf.GradientTape() as t2:
                t2.watch(inputs)
                x = inputs
                for dense_layer in self.dense:
                    x = dense_layer(x)
                self.output_original = self.output_layer(x)
            self.grad1 = t2.gradient(self.output_original, inputs)
        self.grad2 = t1.gradient(self.grad1, inputs)

        if self.bn_flag:
            output = self.subnet_bn(self.output_original, training=training)
        else:
            _ = self.subnet_bn(self.output_original, training=training)
            output = self.output_original
        self.smooth_penalty = tf.reduce_mean(tf.square(self.grad2)) / tf.sqrt(self.subnet_bn.moving_variance)
        return output


class SubnetworkBlock(tf.keras.layers.Layer):

    def __init__(self, subnet_num, subnet_arch=[10, 6], activation_func=tf.tanh, smooth_lambda=0.0, bn_flag=True):
        super(SubnetworkBlock, self).__init__()
        self.subnet_num = subnet_num
        self.subnet_arch = subnet_arch
        self.activation_func = activation_func
        self.smooth_lambda = smooth_lambda
        self.bn_flag = bn_flag

    def build(self, input_shape=None):
        self.subnets = []
        for subnet in range(self.subnet_num):
            self.subnets.append(Subnetwork(self.subnet_arch,
                                           self.activation_func,
                                           self.smooth_lambda,
                                           self.bn_flag))
        self.built = True

    def call(self, inputs, training=False):
        self.smooth_penalties = []
        self.subnet_outputs = []
        self.subnet_inputs = tf.split(inputs, self.subnet_num, 1)
        for i in range(self.subnet_num):
            subnet = self.subnets[i]
            subnet_output = subnet(self.subnet_inputs[i], training=training)
            self.subnet_outputs.append(subnet_output)
            self.smooth_penalties.append(subnet.smooth_penalty)

        output = tf.reshape(tf.squeeze(tf.stack(self.subnet_outputs, 1)), [-1, self.subnet_num])
        self.smooth_loss = self.smooth_lambda * tf.reduce_sum(self.smooth_penalties)
        return output


class OutputLayer(tf.keras.layers.Layer):

    def __init__(self, subnet_num, l1_subnet=0.001):
        super(OutputLayer, self).__init__()
        self.l1_subnet = l1_subnet
        self.subnet_num = subnet_num

    def build(self, input_shape=None):
        self.output_weights = self.add_weight(name="output_weights",
                                              shape=[self.subnet_num, 1],
                                              initializer=tf.keras.initializers.RandomNormal(),
                                              regularizer=tf.keras.regularizers.l1(self.l1_subnet),
                                              trainable=True)
        self.subnet_swicher = self.add_weight(name="switcher",
                                              shape=[self.subnet_num, 1],
                                              initializer=tf.ones_initializer(),
                                              trainable=False)
        self.output_bias = self.add_weight(name="output_bias",
                                           shape=[1],
                                           initializer=tf.zeros_initializer(),
                                           trainable=True)

    def call(self, inputs, training=False):
        output = (tf.matmul(inputs, self.subnet_swicher * self.output_weights) + self.output_bias)
        return output
