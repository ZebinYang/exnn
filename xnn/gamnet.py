import tensorflow as tf
from .base import BaseNet


class GAMNet(BaseNet):
    """
    Generalized additive model vai neural network implementation. It is just a simplified version of sosxnn with identity projection layer.

    Parameters
    ----------
    :type input_num: int
    :param input_num: the length of input variables, excluding multi-class categorical variables.

    :type  input_dummy_num: int
    :param input_dummy_num: optional, default=0, the number of dummy variables.

    :type  subnet_layers: list
    :param subnet_layers: optional, default=(10, 6).
        The architecture of each subnetworks, the ith element represents the number of neurons in the ith layer.

    :type  task: string
    :param task: optional, one of {"Regression", "Classification"}, default="Regression". Only support binary classification at current version.

    :type  batch_size: int
    :param batch_size: optional, default=1000, size of minibatches for stochastic optimizers.

    :type  training_epochs: int
    :param training_epochs: optional, default=10000, maximum number of training epochs.

    :type  tune_epochs: int
    :param tune_epochs: optional, default=500, number of tuning epochs.

    :type  activation: tf object
    :param activation: optional, default=tf.tanh.
        Activation function for the hidden layer of subnetworks. It can be any tensorflow activation function object.

    :type  lr_BP: float
    :param lr_BP: optional, default=0.001, learning rate for weight updates.

    :type  beta_threshold: float
    :param beta_threshold: optional, default=0.01.
        Percentage threshold for pruning the subnetworks, which means the subnetworks that sum up to 95% of the total sclae will be kept.

    :type  l1_proj: float
    :param l1_proj: optional, default=0.001, the strength of L1 penalty for projection layer.

    :type  l1_subnet: float
    :param l1_subnet: optional, default=0.001, the strength of L1 penalty for scaling layer.

    :type  verbose: bool
    :param verbose: optional, default=False. If True, detailed messages will be printed.

    :type  val_ratio : float
    :param val_ratio : optional, default=0.2. 
        The proportion of training data to set aside as validation set for early stopping. Must be between 0 and 1.

    :type  early_stop_thres: int
    :param early_stop_thres: optional, default=1000. Maximum number of epochs if no improvement occurs.
    
    """

    def __init__(self, input_num, input_dummy_num=0, subnet_arch=[10, 6], task="Regression",
                 activation_func=tf.tanh, batch_size=1000, training_epochs=10000, lr_bp=0.001,
                 beta_threshold=0.05, tuning_epochs=500, l1_subnet=0.001, smooth_lambda=0.000001,
                 verbose=False, val_ratio=0.2, early_stop_thres=1000):

        super(GAMNet, self).__init__(input_num=input_num,
                                     input_dummy_num=input_dummy_num,
                                     subnet_num=input_num,
                                     subnet_arch=subnet_arch,
                                     task=task,
                                     proj_method="gam",
                                     activation_func=tf.tanh,
                                     bn_flag=True,
                                     lr_bp=lr_bp,
                                     l1_proj=0,
                                     l1_subnet=l1_subnet,
                                     smooth_lambda=smooth_lambda,
                                     batch_size=batch_size,
                                     training_epochs=training_epochs,
                                     tuning_epochs=tuning_epochs,
                                     beta_threshold=beta_threshold,
                                     verbose=verbose,
                                     val_ratio=val_ratio,
                                     early_stop_thres=early_stop_thres)

    @tf.function
    def train_step_init(self, inputs, labels):

        with tf.GradientTape() as tape:
            pred = self.apply(inputs, training=True)
            pred_loss = self.loss_fn(labels, pred)
            regularization_loss = tf.math.add_n(self.output_layer.losses)
            total_loss = pred_loss + regularization_loss
            if self.smooth_lambda > 0:
                smoothness_loss = self.subnet_blocks.smooth_loss
                total_loss += smoothness_loss

        variables = list(set(self.trainable_weights).difference(set(self.proj_layer.weights)))
        grads = tape.gradient(total_loss, variables)
        self.optimizer.apply_gradients(zip(grads, variables))

    @tf.function
    def train_step_finetune(self, inputs, labels):

        with tf.GradientTape() as tape:
            pred = self.apply(inputs, training=True)
            pred_loss = self.loss_fn(labels, pred)
            total_loss = pred_loss
            if self.smooth_lambda > 0:
                smoothness_loss = self.subnet_blocks.smooth_loss
                total_loss += smoothness_loss

        variables = list(set(self.trainable_weights).difference(set(self.proj_layer.weights)))
        grads = tape.gradient(total_loss, variables)
        self.optimizer.apply_gradients(zip(grads, variables))
