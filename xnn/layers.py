import itertools
import numpy as np
import tensorflow as tf
from scipy.special import comb
from tensorflow.keras import layers


class ProjectLayer(tf.keras.layers.Layer):

    def __init__(self, index_list, subnet_num, l1_proj=0.0, method="random"):
        super(ProjectLayer, self).__init__()
        self.index_list = index_list
        self.input_num = len(index_list)
        self.subnet_num = subnet_num
        self.l1_proj = l1_proj
        self.method = method
        self.trainable = True

        if self.method == "random":
            self.kernel_iniializer = tf.keras.initializers.GlorotNormal()

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
        self.built = True

    def call(self, inputs, training=False):
        output = tf.matmul(tf.gather(inputs, self.index_list, axis=1), self.proj_weights)
        return output


class CategNet(tf.keras.layers.Layer):

    def __init__(self, depth, bn_flag=True, cagetnet_id=0):
        super(CategNet, self).__init__()
        self.bn_flag = bn_flag
        self.depth = depth
        self.cagetnet_id = cagetnet_id

    def build(self, input_shape=None):
        self.categ_bias = self.add_weight(name="cate_bias_" + str(self.cagetnet_id),
                                                 shape=[self.depth, 1],
                                                 initializer=tf.zeros_initializer(),
                                                 trainable=True)
        self.moving_mean = self.add_weight(name="mean" + str(self.cagetnet_id), shape=[1], initializer=tf.zeros_initializer(), trainable=False)
        self.moving_norm = self.add_weight(name="norm" + str(self.cagetnet_id), shape=[1], initializer=tf.ones_initializer(), trainable=False)
        self.built = True

    def call(self, inputs, training=False):

        dummy = tf.one_hot(indices=tf.cast(inputs[:, 0], tf.int32), depth=self.depth)
        self.output_original = tf.matmul(dummy, self.categ_bias)

        if training:
            self.subnet_mean = tf.reduce_mean(self.output_original, 0)
            self.subnet_norm = tf.maximum(tf.math.reduce_std(self.output_original, 0), 1e-10)
            self.moving_mean.assign(self.subnet_mean)
            self.moving_norm.assign(self.subnet_norm)
        else:
            self.subnet_mean = self.moving_mean
            self.subnet_norm = self.moving_norm

        if self.bn_flag:
            output = (self.output_original - self.subnet_mean) / (self.subnet_norm)
        else:
            output = self.output_original
        return output


class CategNetBlock(tf.keras.layers.Layer):

    def __init__(self, meta_info, categ_variable_list=[], categ_index_list=[], bn_flag=True):
        super(CategNetBlock, self).__init__()
        self.meta_info = meta_info
        self.categ_variable_list = categ_variable_list
        self.categ_index_list = categ_index_list
        self.bn_flag = bn_flag

    def build(self, input_shape=None):
        self.categnets = []
        for i, key in enumerate(self.categ_variable_list):
            self.categnets.append(CategNet(depth=len(self.meta_info[key]['values']), bn_flag=self.bn_flag, cagetnet_id=i))
        self.built = True

    def call(self, inputs, training=False):
        output = 0
        if len(self.categ_variable_list) > 0:
            self.categ_output = []
            for i, key in enumerate(self.categ_variable_list):
                self.categ_output.append(self.categnets[i](tf.gather(inputs, [self.categ_index_list[i]], axis=1), training=training))
            output = tf.reshape(tf.squeeze(tf.stack(self.categ_output, 1)), [-1, len(self.categ_variable_list)])
        return output


class Subnetwork(tf.keras.layers.Layer):

    def __init__(self, subnet_arch=[10, 6], activation_func=tf.tanh, smooth_lambda=0.0, bn_flag=False, subnet_id=0):
        super(Subnetwork, self).__init__()
        self.dense = []
        self.subnet_arch = subnet_arch
        self.activation_func = activation_func
        self.smooth_lambda = smooth_lambda
        self.bn_flag = bn_flag
        self.subnet_id = subnet_id

    def build(self, input_shape=None):
        for nodes in self.subnet_arch:
            self.dense.append(layers.Dense(nodes, activation=self.activation_func,
                                           kernel_initializer=tf.keras.initializers.GlorotNormal()))
        self.output_layer = layers.Dense(1, activation=tf.identity,
                                         kernel_initializer=tf.keras.initializers.GlorotNormal())
        self.moving_mean = self.add_weight(name="mean" + str(self.subnet_id), shape=[1], initializer=tf.zeros_initializer(), trainable=False)
        self.moving_norm = self.add_weight(name="norm" + str(self.subnet_id), shape=[1], initializer=tf.ones_initializer(), trainable=False)
        self.built = True

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

        if training:
            self.subnet_mean = tf.reduce_mean(self.output_original, 0)
            self.subnet_norm = tf.maximum(tf.math.reduce_std(self.output_original, 0), 1e-10)
            self.moving_mean.assign(self.subnet_mean)
            self.moving_norm.assign(self.subnet_norm)
        else:
            self.subnet_mean = self.moving_mean
            self.subnet_norm = self.moving_norm

        if self.bn_flag:
            output = (self.output_original - self.subnet_mean) / (self.subnet_norm)
        else:
            output = self.output_original
        self.smooth_penalty = tf.reduce_mean(tf.square(self.grad2)) / self.subnet_norm
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
        for i in range(self.subnet_num):
            self.subnets.append(Subnetwork(self.subnet_arch,
                                           self.activation_func,
                                           self.smooth_lambda,
                                           self.bn_flag,
                                           subnet_id=i))
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
                                              initializer=tf.keras.initializers.GlorotNormal(),
                                              regularizer=tf.keras.regularizers.l1(self.l1_subnet),
                                              trainable=True)
        self.output_switcher = self.add_weight(name="switcher",
                                               shape=[self.subnet_num, 1],
                                               initializer=tf.ones_initializer(),
                                               trainable=False)
        self.output_bias = self.add_weight(name="output_bias",
                                           shape=[1],
                                           initializer=tf.zeros_initializer(),
                                           trainable=True)
        self.built = True

    def call(self, inputs, training=False):
        output = (tf.matmul(inputs, self.output_switcher * self.output_weights) + self.output_bias)
        return output
