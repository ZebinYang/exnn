import tensorflow as tf
from .base import BaseNet


class ExNN(BaseNet):
    """
    Enhanced explainable neural network (ExNN) based on sparse, orthogonal and smooth constraints.

    ExNN is based on our paper (Yang et al. 2020 TNNLS) with the following implementation details:

    1. Categorical variables should be first converted by one-hot encoding, and we directly link each of the dummy variables as a bias term to final output.

    2. The weights of projection layer are forced to be orthogonal, which is separately optimized via Cayley Transform.

    3. A normalization procedure is implemented for each of the subnetwork outputs, for identifiability considerations and improving the performance of L1 sparsity constraint on the scaling layer.

    4. The roughness penalty for subnetworks are implemented via calculating the 2-order gradients from the output to the input of each subnetwork.

    5. We train the network and early stop if no improvement occurs in certain epochs.

    6. The subnetworks whose scaling factors are close to zero are pruned for parsimony consideration.

    7. The pruned network will then be fine-tuned.

    Parameters
    ----------
    :type subnet_num: int
    :param subnet_num: the number of subnetworks.

    :type  meta_info: dict
    :param meta_info: the meta information of the dataset.

    :type  subnet_arch: list
    :param subnet_arch: optional, default=(10, 6), the architecture of each subnetworks, the ith element represents the number of neurons in the ith layer.

    :type  task_type: string
    :param task_type: optional, one of {"Regression", "Classification"}, default="Regression". Only support binary classification at current version.

    :type  batch_size: int
    :param batch_size: optional, default=1000, size of minibatches for stochastic optimizers.

    :type  training_epochs: int
    :param training_epochs: optional, default=10000, maximum number of training epochs.

    :type  activation: tf object
    :param activation: optional, default=tf.tanh, activation function for the hidden layer of subnetworks. It can be any tensorflow activation function object.

    :type  lr_bp: float
    :param lr_bp: optional, default=0.001, learning rate for weight updates.

    :type  lr_cl: float
    :param lr_cl: optional, default=0.1, learning rate of Cayley Transform for updating the projection layer.

    :type  beta_threshold: float
    :param beta_threshold: optional, default=0.05, percentage threshold for pruning the subnetworks, which means the subnetworks that sum up to 95% of the total sclae will be kept.

    :type  tuning_epochs: int
    :param tuning_epochs: optional, default=500, the number of tunning epochs.

    :type  l1_proj: float
    :param l1_proj: optional, default=0.001, the strength of L1 penalty for projection layer.

    :type  l1_subnet: float
    :param l1_subnet: optional, default=0.001, the strength of L1 penalty for scaling layer.

    :type  l2_smooth: float
    :param l2_smooth: optional, default=0.000001, the strength of roughness penalty for subnetworks.

    :type  verbose: bool
    :param verbose: optional, default=False. If True, detailed messages will be printed.

    :type  val_ratio: float
    :param val_ratio: optional, default=0.2. The proportion of training data to set aside as validation set for early stopping. Must be between 0 and 1.

    :type  early_stop_thres: int
    :param early_stop_thres: optional, default=1000. Maximum number of epochs if no improvement occurs.

    :type  random_state: int
    :param random_state: optional, default=0, the random seed.

    References
    ----------
    .. Yang, Zebin, Aijun Zhang, and Agus Sudjianto. "Enhancing Explainability of Neural Networks through Architecture Constraints." TNNLS (2020).

    """

    def __init__(self, meta_info, subnet_num, subnet_arch=[10, 6], task_type="Regression",
                 activation_func=tf.tanh, batch_size=1000, training_epochs=10000, lr_bp=0.001, lr_cl=0.1,
                 beta_threshold=0.05, tuning_epochs=500, l1_proj=0.001, l1_subnet=0.001, l2_smooth=0.000001,
                 verbose=False, val_ratio=0.2, early_stop_thres=1000, random_state=0):

        super(ExNN, self).__init__(meta_info=meta_info,
                             subnet_num=subnet_num,
                             subnet_arch=subnet_arch,
                             task_type=task_type,
                             proj_method="orthogonal",
                             activation_func=activation_func,
                             bn_flag=True,
                             lr_bp=lr_bp,
                             l1_proj=l1_proj,
                             l1_subnet=l1_subnet,
                             l2_smooth=l2_smooth,
                             batch_size=batch_size,
                             training_epochs=training_epochs,
                             tuning_epochs=tuning_epochs,
                             beta_threshold=beta_threshold,
                             verbose=verbose,
                             val_ratio=val_ratio,
                             early_stop_thres=early_stop_thres,
                             random_state=random_state)
        self.lr_cl = lr_cl

    @tf.function
    def train_step_init(self, inputs, labels):
        with tf.GradientTape() as tape_cl:
            with tf.GradientTape() as tape_bp:
                pred = self.__call__(inputs, training=True)
                pred_loss = self.loss_fn(labels, pred)
                regularization_loss = tf.math.add_n(self.proj_layer.losses + self.output_layer.losses)
                cl_loss = pred_loss + regularization_loss
                bp_loss = pred_loss + regularization_loss
                if self.l2_smooth > 0:
                    smoothness_loss = self.subnet_blocks.smooth_loss
                    bp_loss += smoothness_loss

        train_weights_list = []
        trainable_weights_names = [self.trainable_weights[j].name for j in range(len(self.trainable_weights))]
        for i in range(len(self.trainable_weights)):
            if self.trainable_weights[i].name != self.proj_layer.weights[0].name:
                train_weights_list.append(self.trainable_weights[i])

        grad_proj = tape_cl.gradient(cl_loss, self.proj_layer.weights)
        grad_nets = tape_bp.gradient(bp_loss, train_weights_list)

        in_shape = self.proj_layer.weights[0].shape[0]
        matrix_a = (tf.matmul(grad_proj[0], tf.transpose(self.proj_layer.weights[0]))
                    - tf.matmul(self.proj_layer.weights[0], tf.transpose(grad_proj[0])))
        matrix_q = tf.matmul(tf.linalg.inv(tf.eye(in_shape) + tf.multiply(self.lr_cl / 2, matrix_a)),
                             (tf.eye(in_shape) - tf.multiply(self.lr_cl / 2, matrix_a)))
        self.proj_layer.weights[0].assign(tf.matmul(matrix_q, self.proj_layer.weights[0]))
        self.optimizer.apply_gradients(zip(grad_nets, train_weights_list))

    @tf.function
    def train_step_finetune(self, inputs, labels):

        with tf.GradientTape() as tape:
            pred = self.__call__(inputs, training=True)
            pred_loss = self.loss_fn(labels, pred)
            total_loss = pred_loss
            if self.l2_smooth > 0:
                smoothness_loss = self.subnet_blocks.smooth_loss
                total_loss += smoothness_loss

        train_weights_list = []
        trainable_weights_names = [self.trainable_weights[j].name for j in range(len(self.trainable_weights))]
        for i in range(len(self.trainable_weights)):
            if self.trainable_weights[i].name != self.proj_layer.weights[0].name:
                train_weights_list.append(self.trainable_weights[i])

        grads = tape.gradient(total_loss, train_weights_list)
        self.optimizer.apply_gradients(zip(grads, train_weights_list))
