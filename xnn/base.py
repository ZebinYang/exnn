import os
import numpy as np
import matplotlib.lines as mlines
from matplotlib import pyplot as plt

import tensorflow as tf
from .layers import ProjectLayer, SubnetworkBlock, OutputLayer


class BaseNet(tf.keras.Model):
    """
    Abstract Class.

    """

    def __init__(self, input_num,
                 input_dummy_num=0,
                 subnet_num=10,
                 subnet_arch=[10, 6],
                 task="Regression",
                 proj_method="orthogonal",
                 activation_func=tf.tanh,
                 bn_flag=True,
                 lr_bp=0.001,
                 l1_proj=0.001,
                 l1_subnet=0.001,
                 smooth_lambda=0.00001,
                 batch_size=100,
                 training_epochs=10000,
                 tuning_epochs=500,
                 beta_threshold=0.05,
                 verbose=False,
                 val_ratio=0.2,
                 early_stop_thres=1000):

        super(BaseNet, self).__init__()

        # Parameter initiation
        self.input_num = input_num
        self.input_dummy_num = input_dummy_num
        self.subnet_num = subnet_num
        self.subnet_arch = subnet_arch
        self.task = task
        self.proj_method = proj_method
        self.activation_func = activation_func
        self.bn_flag = bn_flag

        self.lr_bp = lr_bp
        self.l1_proj = l1_proj
        self.l1_subnet = l1_subnet
        self.smooth_lambda = smooth_lambda
        self.batch_size = batch_size
        self.beta_threshold = beta_threshold
        self.tuning_epochs = tuning_epochs
        self.training_epochs = training_epochs

        self.verbose = verbose
        self.val_ratio = val_ratio
        self.early_stop_thres = early_stop_thres

        # build
        self.proj_layer = ProjectLayer(input_num=self.input_num,
                                       subnet_num=self.subnet_num,
                                       l1_proj=self.l1_proj,
                                       method=self.proj_method)
        self.subnet_blocks = SubnetworkBlock(subnet_num=self.subnet_num,
                                             subnet_arch=self.subnet_arch,
                                             activation_func=self.activation_func,
                                             smooth_lambda=self.smooth_lambda,
                                             bn_flag=self.bn_flag)
        self.output_layer = OutputLayer(subnet_num=self.subnet_num, l1_subnet=self.l1_subnet)
        self.output_bias_dummy = self.add_weight(name="output_bias_dummy",
                                                 shape=[self.input_dummy_num, 1],
                                                 initializer=tf.zeros_initializer(),
                                                 trainable=True)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_bp)
        if self.task == "Regression":
            self.loss_fn = tf.keras.losses.MeanSquaredError()
        elif self.task == "Classification":
            self.loss_fn = tf.keras.losses.BinaryCrossentropy()
        else:
            raise ValueError('The task type is not supported')

    def call(self, inputs, training=False):

        self.XX = inputs[:, :self.input_num]
        self.XD = inputs[:, self.input_num:]
        self.proj_outputs = self.proj_layer(self.XX, training=training)
        self.subnet_outputs = self.subnet_blocks(self.proj_outputs, training=training)
        
        if self.task == "Regression":
            output = self.output_layer(self.subnet_outputs) + tf.matmul(self.XD, self.output_bias_dummy)
        elif self.task == "Classification":
            output = tf.nn.sigmoid(self.output_layer(self.subnet_outputs) + tf.matmul(self.XD, self.output_bias_dummy))
        else:
            raise ValueError('The task type is not supported')
        
        return output

    @tf.function
    def predict(self, x):
        return self.apply(tf.cast(x, tf.float32), training=False)
    
    @tf.function
    def evaluate(self, x, y, training=False):
        return self.loss_fn(y, self.apply(tf.cast(x, tf.float32), training=training))

    @tf.function
    def train_step_init(self, inputs, labels):
        pass

    @tf.function
    def train_step_finetune(self, inputs, labels):
        pass

    def get_active_subnets(self):
        if self.bn_flag:
            beta = self.output_layer.output_weights.numpy() * self.output_layer.subnet_swicher.numpy()
        else:
            subnet_norm = [self.subnet_blocks.subnets[i].moving_norm.numpy()[0] for i in range(self.subnet_num)]
            beta = self.output_layer.output_weights.numpy() * np.array([subnet_norm]).reshape([-1, 1]) * self.output_layer.subnet_swicher.numpy()

        subnets_scale = (np.abs(beta) / np.sum(np.abs(beta))).reshape([-1])
        sorted_index = np.argsort(subnets_scale)
        active_index = sorted_index[subnets_scale[sorted_index].cumsum()>self.beta_threshold][::-1]
        return active_index, beta, subnets_scale

    def fit(self, train_x, train_y):

        self.err_val = []
        self.err_train = []
        tr_x = train_x[:int(round((1 - self.val_ratio) * train_x.shape[0])), :]
        tr_y = train_y[:int(round((1 - self.val_ratio) * train_x.shape[0])), :]
        val_x = train_x[int(round((1 - self.val_ratio) * train_x.shape[0])):, :]
        val_y = train_y[int(round((1 - self.val_ratio) * train_x.shape[0])):, :]

        # 1. Training
        if self.verbose:
            print("Initial training.")

        last_improvement = 0
        best_validation = np.inf
        train_size = tr_x.shape[0]
        for epoch in range(self.training_epochs):
            shuffle_index = np.arange(tr_x.shape[0])
            np.random.shuffle(shuffle_index)
            tr_x = tr_x[shuffle_index]
            tr_y = tr_y[shuffle_index]

            for iterations in range(train_size // self.batch_size):
                offset = (iterations * self.batch_size) % train_size
                batch_xx = tr_x[offset:(offset + self.batch_size), :]
                batch_yy = tr_y[offset:(offset + self.batch_size)]
                self.train_step_init(tf.cast(batch_xx, tf.float32), batch_yy)

#             self.err_train.append(self.evaluate(tr_x, tr_y, training=True))
#             self.err_val.append(self.evaluate(val_x, val_y, training=True))
#             if self.verbose & (epoch % 1 == 0):
#                 print("Training epoch: %d, train loss: %0.5f, val loss: %0.5f" %
#                       (epoch + 1, self.err_train[-1], self.err_val[-1]))

#             if self.err_val[-1] < best_validation:
#                 best_validation = self.err_val[-1]
#                 last_improvement = epoch
#             if epoch - last_improvement > self.early_stop_thres:
#                 print("Early stop at epoch %d, With Testing Error: %0.5f" % (epoch + 1, self.err_val[-1]))
#                 break

        # 2. pruning
        if self.verbose:
            print("Subnetwork pruning.")

        active_index, _, _ = self.get_active_subnets()
        scal_factor = np.zeros((self.subnet_num, 1))
        scal_factor[active_index] = 1
        self.output_layer.subnet_swicher.assign(tf.constant(scal_factor, dtype=tf.float32))

        # 3. fine tune
        if self.verbose:
            print("Fine tuning.")
            
        for epoch in range(self.tuning_epochs):
            shuffle_index = np.arange(tr_x.shape[0])
            np.random.shuffle(shuffle_index)
            tr_x = tr_x[shuffle_index]
            tr_y = tr_y[shuffle_index]

            for iterations in range(train_size // self.batch_size):
                offset = (iterations * self.batch_size) % train_size
                batch_xx = tr_x[offset:(offset + self.batch_size), :]
                batch_yy = tr_y[offset:(offset + self.batch_size)]
                self.train_step_finetune(tf.cast(batch_xx, tf.float32), batch_yy)

#             self.err_train.append(self.evaluate(tr_x, tr_y, training=True))
#             self.err_val.append(self.evaluate(val_x, val_y, training=True))
#             if self.verbose & (epoch % 1 == 0):
#                 print("Tuning epoch: %d, train loss: %0.5f, val loss: %0.5f" %
#                       (epoch + 1, self.err_train[-1], self.err_val[-1]))

#         self.evaluate(train_x, train_y, training=True)
        # record the key values in the network
        self.subnet_input_min = []
        self.subnet_input_max = []
        for i in range(self.subnet_num):
            min_ = np.dot(train_x[:,:self.input_num], self.proj_layer.get_weights()[0])[:, i].min()
            max_ = np.dot(train_x[:,:self.input_num], self.proj_layer.get_weights()[0])[:, i].max()
            self.subnet_input_min.append(min_)
            self.subnet_input_max.append(max_)

        self.tr_x = tr_x
        self.val_x = val_x
        self.tr_y = tr_y
        self.val_y = val_y

    def visualize(self, folder="./results", name="demo", dummy_name=None, save_eps=False):
        if not os.path.exists(folder):
            os.makedirs(folder)
        save_path = folder + name

        input_size = self.input_num
        active_index, beta, subnets_scale = self.get_active_subnets()
        max_ids = len(active_index)
        max_ids += 1 if self.input_dummy_num else 0

        coef_index = self.proj_layer.proj_weights.numpy()
        dummy_gamma = self.output_bias_dummy.numpy()

        f = plt.figure(figsize=(12, int(len(active_index) * 4.5)))
        for i, indice in enumerate(active_index):

            subnet = self.subnet_blocks.subnets[indice]
            min_ = self.subnet_input_min[indice]
            max_ = self.subnet_input_max[indice]
            subnets_inputs = np.linspace(min_, max_, 1000).reshape([-1, 1])
            subnets_outputs = np.sign(beta[indice]) * subnet.apply(tf.cast(tf.constant(subnets_inputs), tf.float32)).numpy()

            if coef_index[np.argmax(np.abs(coef_index[:, indice])), indice] < 0:
                coef_index[:, indice] = - coef_index[:, indice]
                subnets_inputs = - subnets_inputs

            ax1 = f.add_subplot(np.int(max_ids), 2, i * 2 + 1)
            ax1.plot(subnets_inputs, subnets_outputs)
            yint = np.round(np.linspace(
                np.min(subnets_outputs), np.max(subnets_outputs), 6), 2)
            ax1.set_yticks(yint)
            ax1.set_yticklabels(["{0: .2f}".format(j) for j in yint])
            ax1.set_ylim([np.min(subnets_outputs) - (np.max(subnets_outputs) - np.min(subnets_outputs))*0.1, 
                      np.max(subnets_outputs) + (np.max(subnets_outputs) - np.min(subnets_outputs))*0.25])
            ax1.text(0.25, 0.9,'Scale: ' + str(np.round(100 * subnets_scale[indice], 1)) + "%",
                  fontsize=16,  horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)

            ax2 = f.add_subplot(np.int(max_ids), 2, i * 2 + 2)
            ax2.bar(np.arange(input_size), coef_index.T[indice, :input_size])
            ax2.set_xticks(np.arange(input_size))
            ax2.set_xticklabels(["X" + str(j + 1) for j in range(input_size)])

            yint = np.round(np.linspace(
                np.min(coef_index.T[indice, :]), np.max(coef_index.T[indice, :]), 6), 2)
            ax2.set_yticks(yint)
            ax2.set_yticklabels(["{0: .2f}".format(j) for j in yint])
            if i == 0:
                ax1.set_title("Ridge Functions", fontsize=24)
                ax2.set_title("Projection Indexes", fontsize=24)

        if self.input_dummy_num > 0:
            ax3 = f.add_subplot(np.int(max_ids), 1, np.int(max_ids))
            ax3.bar(np.arange(self.input_dummy_num),
                    dummy_gamma[:, 0] - dummy_gamma[-1, 0])
            ax3.set_xticks(np.arange(self.input_dummy_num))
            ax3.set_xticklabels(dummy_name)

        if max_ids > 0:
            f.savefig("%s.png" % save_path, bbox_inches='tight', dpi=100)
            if save_eps:
                f.savefig("%s.eps" % save_path, bbox_inches='tight', dpi=100)
