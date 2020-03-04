import os
import numpy as np
import tensorflow as tf
import matplotlib.lines as mlines
from matplotlib import pyplot as plt
from abc import ABCMeta, abstractmethod
from sklearn.model_selection import train_test_split

from .layers import ProjectLayer, SubnetworkBlock, OutputLayer, CategNetBlock


class BaseNet(tf.keras.Model, metaclass=ABCMeta):
    """
    Abstract Class.

    """
    @abstractmethod
    def __init__(self, meta_info,
                 subnet_num=10,
                 subnet_arch=[10, 6],
                 task_type="Regression",
                 proj_method="orthogonal",
                 activation_func=tf.tanh,
                 bn_flag=True,
                 lr_bp=0.001,
                 l1_proj=0.001,
                 l1_subnet=0.001,
                 smooth_lambda=0.00001,
                 batch_size=1000,
                 training_epochs=2000,
                 tuning_epochs=500,
                 beta_threshold=0.05,
                 verbose=False,
                 val_ratio=0.2,
                 early_stop_thres=1000,
                 random_state=0):

        super(BaseNet, self).__init__()

        # Parameter initiation
        self.meta_info = meta_info
        self.subnet_num = subnet_num
        self.subnet_arch = subnet_arch
        self.task_type = task_type
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
        self.random_state = random_state
        
        np.random.seed(random_state)
        tf.random.set_seed(random_state)

        self.categ_variable_num = 0
        self.numerical_input_num = 0
        self.categ_variable_list = []
        self.categ_index_list = []
        self.noncateg_index_list = []
        self.noncateg_variable_list = []
        self.variables_names = list(self.meta_info.keys())
        for i, (key, item) in enumerate(self.meta_info.items()):
            if item['type'] == "target":
                continue
            if item['type'] == "categorical":
                self.categ_variable_num += 1
                self.categ_variable_list.append(key)
                self.categ_index_list.append(i)
            else:
                self.numerical_input_num +=1
                self.noncateg_index_list.append(i)
                self.noncateg_variable_list.append(key)

        self.subnet_num = min(self.subnet_num, self.numerical_input_num)
        # build
        self.proj_layer = ProjectLayer(index_list=list(self.noncateg_index_list),
                                       subnet_num=self.subnet_num,
                                           l1_proj=self.l1_proj,
                                           method=self.proj_method)
        
        self.categ_blocks = CategNetBlock(meta_info=self.meta_info, 
                                         categ_variable_list=self.categ_variable_list, 
                                         categ_index_list=self.categ_index_list,
                                         bn_flag=self.bn_flag)

        self.subnet_blocks = SubnetworkBlock(subnet_num=self.subnet_num,
                                             subnet_arch=self.subnet_arch,
                                             activation_func=self.activation_func,
                                             smooth_lambda=self.smooth_lambda,
                                             bn_flag=self.bn_flag)
        self.output_layer = OutputLayer(subnet_num=self.subnet_num + self.categ_variable_num, l1_subnet=self.l1_subnet)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_bp)
        if self.task_type == "Regression":
            self.loss_fn = tf.keras.losses.MeanSquaredError()
        elif self.task_type == "Classification":
            self.loss_fn = tf.keras.losses.BinaryCrossentropy()
        else:
            raise ValueError('The task type is not supported')

    def call(self, inputs, training=False):

        self.proj_outputs = self.proj_layer(inputs, training=training)
        self.categ_outputs = self.categ_blocks(inputs, training=training)
        self.subnet_outputs = self.subnet_blocks(self.proj_outputs, training=training)
        
        concat_list = []
        if self.numerical_input_num > 0:
            concat_list.append(self.subnet_outputs)
        if self.categ_variable_num > 0:
            concat_list.append(self.categ_outputs)

        if self.task_type == "Regression":
            output = self.output_layer(tf.concat(concat_list, 1))
        elif self.task_type == "Classification":
            output = tf.nn.sigmoid(self.output_layer(tf.concat(concat_list, 1)))
        else:
            raise ValueError('The task type is not supported')
        
        return output

    @tf.function
    def predict_graph(self, x):
        return self.__call__(tf.cast(x, tf.float32), training=False)

    def predict(self, x):
        return self.predict_graph(x).numpy()
    
    @tf.function
    def evaluate_graph(self, x, y, training=False):
        return self.loss_fn(y, self.__call__(tf.cast(x, tf.float32), training=training))

    def evaluate(self, x, y, training=False):
        return self.evaluate_graph(x, y, training=training).numpy()

    @tf.function
    def train_step_init(self, inputs, labels):
        pass

    @tf.function
    def train_step_finetune(self, inputs, labels):
        pass
                                       
    def get_active_subnets(self, beta_threshold=0):
        if self.bn_flag:
            beta = self.output_layer.output_weights.numpy()
        else:
            subnet_norm = [self.subnet_blocks.subnets[i].moving_norm.numpy()[0] for i in range(self.numerical_input_num)]
            categ_norm = [self.categ_blocks.categnets[i].moving_norm.numpy()[0]for i in range(self.categ_variable_num)]
            beta = self.output_layer.output_weights.numpy() * np.hstack([subnet_norm, categ_norm]).reshape([-1, 1])
        beta = beta * self.output_layer.output_switcher.numpy()
        subnets_scale = (np.abs(beta) / np.sum(np.abs(beta))).reshape([-1])
        sorted_index = np.argsort(subnets_scale)
        active_index = sorted_index[subnets_scale[sorted_index].cumsum()>beta_threshold][::-1]

        active_me_index = []
        active_categ_index = []
        for i in active_index:
            if i in range(self.subnet_num):
                active_me_index.append(i)
            elif i in range(self.subnet_num, self.subnet_num + self.categ_variable_num):
                active_categ_index.append(i)
        return active_me_index, active_categ_index, beta, subnets_scale

    def fit(self, train_x, train_y):

        self.err_val = []
        self.err_train = []
    
        n_samples = train_x.shape[0]
        indices = np.arange(n_samples)
        if self.task_type == "Regression":
            tr_x, val_x, tr_y, val_y, tr_idx, val_idx = train_test_split(train_x, train_y, indices, test_size=self.val_ratio, 
                                          random_state=self.random_state)
        elif self.task_type == "Classification":
            tr_x, val_x, tr_y, val_y, tr_idx, val_idx = train_test_split(train_x, train_y, indices, test_size=self.val_ratio, 
                                      stratify=train_y, random_state=self.random_state)
        self.tr_idx = tr_idx
        self.val_idx = val_idx

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

            self.err_train.append(self.evaluate(tr_x, tr_y, training=False))
            self.err_val.append(self.evaluate(val_x, val_y, training=False))
            if self.verbose & (epoch % 1 == 0):
                print("Training epoch: %d, train loss: %0.5f, val loss: %0.5f" %
                      (epoch + 1, self.err_train[-1], self.err_val[-1]))

            if self.err_val[-1] < best_validation:
                best_validation = self.err_val[-1]
                last_improvement = epoch
            if epoch - last_improvement > self.early_stop_thres:
                if self.verbose:
                    print("Early stop at epoch %d, With Testing Error: %0.5f" % (epoch + 1, self.err_val[-1]))
                break

        # 2. pruning
        if self.verbose:
            print("Subnetwork pruning.")

        active_me_index, active_categ_index, _, _ = self.get_active_subnets(self.beta_threshold)
        scal_factor = np.zeros((self.subnet_num + self.categ_variable_num, 1))
        scal_factor[active_me_index] = 1
        scal_factor[active_categ_index] = 1
        self.output_layer.output_switcher.assign(tf.constant(scal_factor, dtype=tf.float32))

        # 3. fine tune
        if self.verbose:
            print("Fine tuning.")
            
        last_improvement = 0
        best_validation = np.inf
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

            self.err_train.append(self.evaluate(tr_x, tr_y, training=False))
            self.err_val.append(self.evaluate(val_x, val_y, training=False))
            if self.verbose & (epoch % 1 == 0):
                print("Tuning epoch: %d, train loss: %0.5f, val loss: %0.5f" %
                      (epoch + 1, self.err_train[-1], self.err_val[-1]))
            
            if self.err_val[-1] < best_validation:
                best_validation = self.err_val[-1]
                last_improvement = epoch
            if epoch - last_improvement > self.early_stop_thres:
                if self.verbose:
                    print("Early stop at epoch %d, With Testing Error: %0.5f" % (epoch + 1, self.err_val[-1]))
                break

        # record the key values in the network
        self.subnet_input_min = []
        self.subnet_input_max = []
        for i in range(self.subnet_num):
            min_ = np.dot(train_x[:,self.noncateg_index_list], self.proj_layer.get_weights()[0])[:, i].min()
            max_ = np.dot(train_x[:,self.noncateg_index_list], self.proj_layer.get_weights()[0])[:, i].max()
            self.subnet_input_min.append(min_)
            self.subnet_input_max.append(max_)


    def visualize(self, folder="./results/", name="demo", save_png=False, save_eps=False):
        if not os.path.exists(folder):
            os.makedirs(folder)
        save_path = folder + name

        input_size = self.numerical_input_num
        coef_index = self.proj_layer.proj_weights.numpy()
        active_index, active_categ_index, beta, subnets_scale = self.get_active_subnets()
        max_ids = len(active_index) + len(active_categ_index)

        f = plt.figure(figsize=(12, int(max_ids * 4.5)))
        for i, indice in enumerate(active_index):

            subnet = self.subnet_blocks.subnets[indice]
            min_ = self.subnet_input_min[indice]
            max_ = self.subnet_input_max[indice]
            subnets_inputs = np.linspace(min_, max_, 1000).reshape([-1, 1])
            subnets_outputs = np.sign(beta[indice]) * subnet.__call__(tf.cast(tf.constant(subnets_inputs), tf.float32)).numpy()

            if coef_index[np.argmax(np.abs(coef_index[:, indice])), indice] < 0:
                coef_index[:, indice] = - coef_index[:, indice]
                subnets_inputs = - subnets_inputs

            ax1 = f.add_subplot(np.int(max_ids), 2, i * 2 + 1)
            ax1.plot(subnets_inputs, subnets_outputs)
            
            xint = np.round(np.linspace(np.min(subnets_inputs), np.max(subnets_inputs), 5), 2)
            ax1.set_xticks(xint)
            ax1.set_xticklabels(["{0: .2f}".format(j) for j in xint], fontsize=14)
            
            yint = np.round(np.linspace(np.min(subnets_outputs), np.max(subnets_outputs), 6), 2)
            ax1.set_yticks(yint)
            ax1.set_yticklabels(["{0: .2f}".format(j) for j in yint], fontsize=14)
            ax1.set_ylim([np.min(subnets_outputs) - (np.max(subnets_outputs) - np.min(subnets_outputs))*0.1, 
                      np.max(subnets_outputs) + (np.max(subnets_outputs) - np.min(subnets_outputs))*0.25])
            ax1.text(0.25, 0.9,'IR: ' + str(np.round(100 * subnets_scale[indice], 1)) + "%",
                  fontsize=24, horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)

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
                ax2.set_title("Projection Indices", fontsize=24)

        if self.categ_variable_num > 0:
            for indice in active_categ_index:
                dummy_name = self.categ_variable_list[indice - self.numerical_input_num]
                dummy_gamma = self.categ_blocks.categnets[indice - self.numerical_input_num].categ_bias.numpy()
                norm = self.categ_blocks.categnets[indice - self.numerical_input_num].moving_norm.numpy()
                ax3 = f.add_subplot(np.int(max_ids), 1, np.int(max_ids))
                ax3.bar(np.arange(len(self.meta_info[dummy_name]['values'])), np.sign(beta[indice]) * dummy_gamma[:, 0] / norm)
                ax3.set_xticks(np.arange(len(self.meta_info[dummy_name]['values'])))
                ax3.set_xticklabels(self.meta_info[self.categ_variable_list[indice - self.numerical_input_num]]['values'], fontsize=14)

                yint = np.round(np.linspace(np.min(np.sign(beta[indice]) * dummy_gamma[:, 0] / norm),
                           np.max(np.sign(beta[indice]) * dummy_gamma[:, 0] / norm), 6), 2)
                ax3.set_yticks(yint)
                ax3.set_yticklabels(["{0: .2f}".format(j) for j in yint], fontsize=14)
                ax3.set_title(dummy_name + " (" + str(np.round(100 * subnets_scale[indice], 1)) + "%)", fontsize=20)

        if max_ids > 0:
            if save_png:
                f.savefig("%s.png" % save_path, bbox_inches='tight', dpi=100)
            if save_eps:
                f.savefig("%s.eps" % save_path, bbox_inches='tight', dpi=100)
