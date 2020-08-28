import os
import numpy as np
from itertools import *
import tensorflow as tf
from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib import lines as mlines
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
                 l2_smooth=0.00001,
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
        self.subnet_num = subnet_num
        self.subnet_arch = subnet_arch
        self.task_type = task_type
        self.proj_method = proj_method
        self.activation_func = activation_func
        self.bn_flag = bn_flag

        self.lr_bp = lr_bp
        self.l1_proj = l1_proj
        self.l1_subnet = l1_subnet
        self.l2_smooth = l2_smooth
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

        self.dummy_values_ = {}
        self.cfeature_num_ = 0
        self.nfeature_num_ = 0
        self.cfeature_list_ = []
        self.nfeature_list_ = []
        self.cfeature_index_list_ = []
        self.nfeature_index_list_ = []
        
        self.feature_list_ = []
        self.feature_type_list_ = []
        for idx, (feature_name, feature_info) in enumerate(meta_info.items()):
            if feature_info["type"] == "target":
                continue
            if feature_info["type"] == "categorical":
                self.cfeature_num_ += 1
                self.cfeature_list_.append(feature_name)
                self.cfeature_index_list_.append(idx)
                self.feature_type_list_.append("categorical")
                self.dummy_values_.update({feature_name:meta_info[feature_name]["values"]})
            else:
                self.nfeature_num_ += 1
                self.nfeature_list_.append(feature_name)
                self.nfeature_index_list_.append(idx)
                self.feature_type_list_.append("continuous")
            self.feature_list_.append(feature_name)

        # build
        self.subnet_num = min(self.subnet_num, self.nfeature_num_)
        self.proj_layer = ProjectLayer(index_list=self.nfeature_index_list_,
                               subnet_num=self.subnet_num,
                               l1_proj=self.l1_proj,
                               method=self.proj_method)
        
        self.categ_blocks = CategNetBlock(feature_list=self.feature_list_,
                               cfeature_index_list=self.cfeature_index_list_,
                               dummy_values=self.dummy_values_, 
                               bn_flag=self.bn_flag)

        self.subnet_blocks = SubnetworkBlock(subnet_num=self.subnet_num,
                                 subnet_arch=self.subnet_arch,
                                 activation_func=self.activation_func,
                                 l2_smooth=self.l2_smooth,
                                 bn_flag=self.bn_flag)
        self.output_layer = OutputLayer(subnet_num=self.subnet_num + self.cfeature_num_, l1_subnet=self.l1_subnet)

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
        if self.nfeature_num_ > 0:
            concat_list.append(self.subnet_outputs)
        if self.cfeature_num_ > 0:
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
          
    @property
    def projection_indices_(self):
        """Return the projection indices.
        Returns
        -------
        projection_indices_ : ndarray of shape (d, )
        """
        projection_indices = np.array([])
        if self.nfeature_num_ > 0:
            active_sim_subnets = [item["indice"] for key, item in self.active_subnets_.items()]
            projection_indices = self.proj_layer.proj_weights.numpy()[:, active_sim_subnets]
            return projection_indices

    @property
    def orthogonality_measure_(self):
        """Return the orthogonality measure (the lower, the better).
        Returns
        -------
        orthogonality_measure_ : float scalar
        """
        ortho_measure = np.nan
        if self.nfeature_num_ > 0:
            ortho_measure = np.linalg.norm(np.dot(self.projection_indices_.T,
                                      self.projection_indices_) - np.eye(self.projection_indices_.shape[1]))
            if self.projection_indices_.shape[1] > 1:
                ortho_measure /= self.projection_indices_.shape[1]
        return ortho_measure
        
    @property
    def importance_ratios_(self):
        """Return the estimator importance ratios (the higher, the more important the feature).
        Returns
        -------
        importance_ratios_ : ndarray of shape (n_estimators,)
            The estimator importances.
        """
        importance_ratios_ = {**self.active_subnets_, **self.active_dummy_subnets_}
        return importance_ratios_

    @property
    def active_subnets_(self):
        """
        Return the information of sim subnetworks
        """
        if self.bn_flag:
            beta = self.output_layer.output_weights.numpy()
        else:
            subnet_norm = [self.subnet_blocks.subnets[i].moving_norm.numpy()[0] for i in range(self.subnet_num)]
            categ_norm = [self.categ_blocks.categnets[i].moving_norm.numpy()[0]for i in range(self.cfeature_num_)]
            beta = self.output_layer.output_weights.numpy() * np.hstack([subnet_norm, categ_norm]).reshape([-1, 1])
            
        beta = beta * self.output_layer.output_switcher.numpy()
        importance_ratio = (np.abs(beta) / np.sum(np.abs(beta))).reshape([-1])
        sorted_index = np.argsort(importance_ratio)
        active_index = sorted_index[importance_ratio[sorted_index].cumsum() > 0][::-1]
        active_subnets = {"Subnet " + str(indice + 1):{"type":"sim_net",
                                          "indice":indice,
                                          "rank":idx,
                                          "beta":self.output_layer.output_weights.numpy()[indice],
                                          "ir":importance_ratio[indice]}
                      for idx, indice in enumerate(active_index) if indice in range(self.subnet_num)}

        return active_subnets

    @property
    def active_dummy_subnets_(self):
        """
        Return the information of active categorical features
        """
        if self.bn_flag:
            beta = self.output_layer.output_weights.numpy()
        else:
            subnet_norm = [self.subnet_blocks.subnets[i].moving_norm.numpy()[0] for i in range(self.subnet_num)]
            categ_norm = [self.categ_blocks.categnets[i].moving_norm.numpy()[0]for i in range(self.cfeature_num_)]
            beta = self.output_layer.output_weights.numpy() * np.hstack([subnet_norm, categ_norm]).reshape([-1, 1])
            
        beta = beta * self.output_layer.output_switcher.numpy()
        importance_ratio = (np.abs(beta) / np.sum(np.abs(beta))).reshape([-1])
        sorted_index = np.argsort(importance_ratio)
        active_index = sorted_index[importance_ratio[sorted_index].cumsum() > 0][::-1]

        active_dummy_subnets = {self.cfeature_list_[indice - self.subnet_num]:{"type":"dummy_net",
                                                       "indice":indice,
                                                       "rank":idx,
                                                       "beta":self.output_layer.output_weights.numpy()[indice],
                                                       "ir":importance_ratio[indice]}
                      for idx, indice in enumerate(active_index) if indice in range(self.subnet_num, self.subnet_num + self.cfeature_num_)}
        return active_dummy_subnets

    def estimate_density(self, x):
        
        density, bins = np.histogram(x, bins=10, density=True)
        return density, bins

    def get_active_subnets(self, beta_threshold=0):
        if self.bn_flag:
            beta = self.output_layer.output_weights.numpy()
        else:
            subnet_norm = [self.subnet_blocks.subnets[i].moving_norm.numpy()[0] for i in range(self.subnet_num)]
            categ_norm = [self.categ_blocks.categnets[i].moving_norm.numpy()[0]for i in range(self.cfeature_num_)]
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
            elif i in range(self.subnet_num, self.subnet_num + self.cfeature_num_):
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

        self.evaluate(tr_x, tr_y, training=True) # update the batch normalization using all the training data
        active_me_index, active_categ_index, _, _ = self.get_active_subnets(self.beta_threshold)
        scal_factor = np.zeros((self.subnet_num + self.cfeature_num_, 1))
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
        self.dummy_density_ = {}
        self.subnet_input_density = []
        self.evaluate(tr_x, tr_y, training=True) # update the batch normalization using all the training data
        for i in range(self.subnet_num):
            xb = np.dot(tr_x[:,self.nfeature_index_list_], self.proj_layer.get_weights()[0])[:, i]
            min_ = xb.min()
            max_ = xb.max()
            self.subnet_input_min.append(min_)
            self.subnet_input_max.append(max_)
            self.subnet_input_density.append(self.estimate_density(xb))
        for idx in range(self.cfeature_num_):
            feature_name = self.cfeature_list_[idx]
            feature_indice = self.cfeature_index_list_[idx]

            unique, counts = np.unique(tr_x[:, feature_indice], return_counts=True)
            density = np.zeros((len(self.dummy_values_[feature_name])))
            density[unique.astype(int)] = counts / tr_x.shape[0]
            self.dummy_density_.update({feature_name:{"density":{"values":self.dummy_values_[feature_name],
                                            "scores":density}}})


    def visualize(self, folder="./results/", name="demo", save_png=False, save_eps=False):

        input_size = self.nfeature_num_
        coef_index = self.proj_layer.proj_weights.numpy()
        active_index, active_categ_index, beta, subnets_scale = self.get_active_subnets()
        max_ids = len(active_index) + len(active_categ_index)

        fig = plt.figure(figsize=(12, int(max_ids * 4.5)))
        for i, indice in enumerate(active_index):

            subnet = self.subnet_blocks.subnets[indice]
            min_ = self.subnet_input_min[indice]
            max_ = self.subnet_input_max[indice]
            subnets_inputs = np.linspace(min_, max_, 1000).reshape([-1, 1])
            subnets_outputs = np.sign(beta[indice]) * subnet.__call__(tf.cast(tf.constant(subnets_inputs), tf.float32)).numpy()

            if coef_index[np.argmax(np.abs(coef_index[:, indice])), indice] < 0:
                coef_index[:, indice] = - coef_index[:, indice]
                subnets_inputs = - subnets_inputs

            ax1 = fig.add_subplot(np.int(max_ids), 2, i * 2 + 1)
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

            ax2 = fig.add_subplot(np.int(max_ids), 2, i * 2 + 2)
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

        if self.cfeature_num_ > 0:
            for indice in active_categ_index:
                feature_name = self.cfeature_list_[indice - self.subnet_num]
                dummy_gamma = self.categ_blocks.categnets[indice - self.subnet_num].categ_bias.numpy()
                norm = self.categ_blocks.categnets[indice - self.subnet_num].moving_norm.numpy()
                ax3 = fig.add_subplot(np.int(max_ids), 1, np.int(max_ids))
                ax3.bar(np.arange(len(self.dummy_values_[feature_name])), np.sign(beta[indice]) * dummy_gamma[:, 0] / norm)
                ax3.set_xticks(np.arange(len(self.dummy_values_[feature_name])))
                ax3.set_xticklabels(self.dummy_values_[feature_name], fontsize=14)

                yint = np.round(np.linspace(np.min(np.sign(beta[indice]) * dummy_gamma[:, 0] / norm),
                           np.max(np.sign(beta[indice]) * dummy_gamma[:, 0] / norm), 6), 2)
                ax3.set_yticks(yint)
                ax3.set_yticklabels(["{0: .2f}".format(j) for j in yint], fontsize=14)
                ax3.set_title(feature_name + " (" + str(np.round(100 * subnets_scale[indice], 1)) + "%)", fontsize=20)

        plt.show()
        if max_ids > 0:
            save_path = folder + name
            if save_eps:
                if not os.path.exists(folder):
                    os.makedirs(folder)
                fig.savefig("%s.eps" % save_path, bbox_inches="tight", dpi=100)
            if save_png:
                if not os.path.exists(folder):
                    os.makedirs(folder)
                fig.savefig("%s.png" % save_path, bbox_inches="tight", dpi=100)
                
                
    def visualize_new(self, cols_per_row=3, subnet_num=10**5, dummy_subnet_num=10**5, show_indices=10**5,
                  folder="./results/", name="demo", save_png=False, save_eps=False):

        input_size = min(self.nfeature_num_, show_indices)
        coef_index = self.proj_layer.proj_weights.numpy()
        projection_indices = self.projection_indices_[:, :subnet_num]
        active_subnets = list(islice(self.active_subnets_.items(), subnet_num))  
        active_dummy_subnets = list(islice(self.active_dummy_subnets_.items(), dummy_subnet_num))
        max_ids = len(active_subnets) + len(active_dummy_subnets)
        fig = plt.figure(figsize=(8 * cols_per_row, 4.6 * int(np.ceil(max_ids / cols_per_row))))
        outer = gridspec.GridSpec(int(np.ceil(max_ids / cols_per_row)), cols_per_row, wspace=0.15, hspace=0.25)

        if projection_indices.shape[1] > 0:
            xlim_min = - max(np.abs(projection_indices.min() - 0.1), np.abs(projection_indices.max() + 0.1))
            xlim_max = max(np.abs(projection_indices.min() - 0.1), np.abs(projection_indices.max() + 0.1))
        for idx, (key, item) in enumerate(active_subnets):
            
            indice = item["indice"]
            inner = outer[idx].subgridspec(2, 2, wspace=0.15, height_ratios=[6, 1], width_ratios=[3, 1])
            ax1_main = fig.add_subplot(inner[0, 0])
            subnet = self.subnet_blocks.subnets[indice]
            min_ = self.subnet_input_min[indice]
            max_ = self.subnet_input_max[indice]
            density, bins = self.subnet_input_density[indice]
            xgrid = np.linspace(min_, max_, 1000).reshape([-1, 1])
            ygrid = np.sign(item["beta"]) * subnet.__call__(tf.cast(tf.constant(xgrid), tf.float32)).numpy()

            if coef_index[np.argmax(np.abs(coef_index[:, indice])), indice] < 0:
                coef_index[:, indice] = - coef_index[:, indice]
                xgrid = - xgrid

            ax1_main.plot(xgrid, ygrid, color="red")
            ax1_main.set_xticklabels([])
            ax1_main.set_title("SIM " + str(idx + 1) + 
                         " (IR: " + str(np.round(100 * item["ir"], 2)) + "%)", fontsize=16)
            fig.add_subplot(ax1_main)

            ax1_density = fig.add_subplot(inner[1, 0])  
            xint = ((np.array(bins[1:]) + np.array(bins[:-1])) / 2).reshape([-1, 1]).reshape([-1])
            ax1_density.bar(xint, density, width=xint[1] - xint[0])
            ax1_main.get_shared_x_axes().join(ax1_main, ax1_density)
            ax1_density.set_yticklabels([])
            fig.add_subplot(ax1_density)
            
            ax2 = fig.add_subplot(inner[:, 1])
            if input_size <= 20:
                rects = ax2.barh(np.arange(input_size), [beta for beta in coef_index.T[indice, :input_size].ravel()][::-1])
                ax2.set_yticks(np.arange(input_size))
                ax2.set_yticklabels(["X" + str(idx + 1) for idx in range(input_size)][::-1])
                ax2.set_xlim(xlim_min, xlim_max)
                ax2.set_ylim(-1, len(coef_index.T[indice, :input_size]))
                ax2.axvline(0, linestyle="dotted", color="black")
            else:
                right = np.round(np.linspace(0, np.round(len(coef_index.T[indice, :input_size].ravel()) * 0.45).astype(int), 5))
                left = len(coef_index.T[indice, :input_size].ravel()) - 1 - right
                input_ticks = np.unique(np.hstack([left, right])).astype(int)

                rects = plt.barh(np.arange(len(coef_index.T[indice, :input_size].ravel())),
                            [beta for beta in coef_index.T[indice, :input_size].ravel()][::-1])
                ax2.set_yticks(input_ticks)
                ax2.set_yticklabels(["X" + str(idx + 1) for idx in input_ticks][::-1])
                ax2.set_xlim(xlim_min, xlim_max)
                ax2.set_ylim(-1, len(coef_index.T[indice, :input_size].ravel()))
                ax2.axvline(0, linestyle="dotted", color="black")
            fig.add_subplot(ax2)
        
        for idx, (key, item) in enumerate(active_dummy_subnets):

            indice = item["indice"]
            feature_name = self.cfeature_list_[indice - self.subnet_num]
            norm = self.categ_blocks.categnets[indice - self.subnet_num].moving_norm.numpy()
            dummy_values = self.dummy_density_[feature_name]["density"]["values"]
            dummy_scores = self.dummy_density_[feature_name]["density"]["scores"]
            dummy_coef = self.categ_blocks.categnets[indice - self.subnet_num].categ_bias.numpy()
            dummy_coef = np.sign(item["beta"]) * dummy_coef[:, 0] / norm

            ax_main = fig.add_subplot(outer[len(self.active_subnets_) + idx])
            ax_density = ax_main.twinx()
            ax_density.bar(np.arange(len(dummy_values)), dummy_scores, width=0.6)
            ax_density.set_ylim(0, dummy_scores.max() * 1.2)
            ax_density.set_yticklabels([])

            input_ticks = (np.arange(len(dummy_values)) if len(dummy_values) <= 6 else 
                              np.linspace(0.1 * len(dummy_coef), len(dummy_coef) * 0.9, 4).astype(int))
            input_labels = [dummy_values[i] for i in input_ticks]
            if len("".join(list(map(str, input_labels)))) > 30:
                input_labels = [str(dummy_values[i])[:4] for i in input_ticks]

            ax_main.set_xticks(input_ticks)
            ax_main.set_xticklabels(input_labels)
            ax_main.set_ylim(- np.abs(dummy_coef).max() * 1.2, np.abs(dummy_coef).max() * 1.2)
            ax_main.plot(np.arange(len(dummy_values)), dummy_coef, color="red", marker="o")
            ax_main.axhline(0, linestyle="dotted", color="black")
            ax_main.set_title(feature_name +
                             " (IR: " + str(np.round(100 * item["ir"], 2)) + "%)", fontsize=16)
            ax_main.set_zorder(ax_density.get_zorder() + 1)
            ax_main.patch.set_visible(False)

        plt.show()
        if max_ids > 0:
            save_path = folder + name
            if save_png:
                if not os.path.exists(folder):
                    os.makedirs(folder)
                f.savefig("%s.png" % save_path, bbox_inches='tight', dpi=100)
            if save_eps:
                if not os.path.exists(folder):
                    os.makedirs(folder)
                f.savefig("%s.eps" % save_path, bbox_inches='tight', dpi=100)